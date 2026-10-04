"""Offline Review experiment verification; real transports are blocked or replaced in every test."""
import csv
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from experiments.protocol import configuration, prepare, digest
from experiments.runner import execute, run_sample
from experiments.transport import (Journal, MockTransport, UnknownAttempt, read_journal,
    ResponseError, response_content, LiveTransport, HTTPFailure)
from experiments.analysis import analyze, metrics, gain
from run_review_experiment import main


class ReviewExperimentTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root/'run'
        self.config = configuration(ROOT/'examples/review_archive.json')
        self.fixtures = json.loads((ROOT/'examples/review_mock_responses.json').read_text())
        # Any accidental network contact fails the test, including SDK bypasses.
        self.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden'))
        self.network.start()

    def tearDown(self):
        self.network.stop()
        self.tmp.cleanup()

    def plan(self, **changes):
        return prepare(dict(self.config, **changes), self.output)

    def test_dry_run_does_not_construct_transport_or_read_credentials(self):
        with patch('experiments.runner.LiveTransport', side_effect=AssertionError('live constructor')), \
             patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'do-not-read'}):
            self.assertEqual(main(['--config', str(ROOT/'examples/review_archive.json'),
                                   '--output', str(self.output)]), 0)
        self.assertFalse((self.output/'execution.json').exists())
        self.assertFalse((self.output/'requests.jsonl').exists())
        self.assertNotIn('do-not-read', (self.output/'plan.json').read_text())

    def test_three_conditions_gain_coverage_costs_and_offline_recompute(self):
        plan = self.plan()
        report = execute(plan, self.output, 'simulate', self.fixtures)
        self.assertEqual(report['real_api_call_attempts'], 0)
        self.assertEqual(report['evidence'], 'synthetic_only')
        self.assertEqual(report['status'], 'complete_with_errors')
        self.assertEqual(report['triggered_count'], 4)
        self.assertAlmostEqual(report['trigger_rate_over_completed'], 4/7)
        self.assertAlmostEqual(report['trigger_rate_over_routed'], 4/6)
        summary = report['comparisons']['all']
        self.assertEqual((summary['gain']['corrected'], summary['gain']['corrupted'],
                          summary['gain']['net_gain_count']), (2, 1, 1))
        self.assertEqual(summary['gain']['lost_to_abstention'], 1)
        self.assertEqual(summary['gain']['correct_count_delta'], 0)
        self.assertEqual(summary['metrics']['first_only']['scored'], 6)
        self.assertEqual(summary['metrics']['full_two_layer']['scored'], 5)
        self.assertEqual(summary['metrics']['direct_llm']['scored'], 7)
        self.assertEqual(report['comparisons']['not_triggered']['gain']['net_gain_count'], 0)
        costs = report['costs']
        self.assertEqual(costs['first_only']['call_attempts'], 27)
        self.assertEqual(costs['review_increment']['call_attempts'], 21)
        self.assertEqual(costs['direct_llm']['call_attempts'], 8)
        self.assertEqual(costs['physical_total']['call_attempts'], 56)
        self.assertEqual(costs['physical_total']['retry_attempts'], 5)
        self.assertEqual(costs['physical_total']['failed_attempts'], 7)
        self.assertEqual(costs['physical_total']['observed_tokens']['total_tokens'], 50*15)
        self.assertIsNone(costs['physical_total']['token_totals']['total_tokens'])
        self.assertEqual(costs['physical_total']['token_usage_missing_attempts']['total_tokens'], 6)
        self.assertEqual(report, analyze(self.output))
        with (self.output/'stage_predictions.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 28)

    def test_resume_partial_and_complete_never_duplicates_requests(self):
        plan = self.plan()
        first = execute(plan, self.output, 'simulate', self.fixtures, max_samples=2)
        self.assertEqual((first['completed'], first['pending']), (2, 5))
        records = read_journal(self.output/'requests.jsonl')
        report = execute(plan, self.output, 'simulate', self.fixtures)
        full = read_journal(self.output/'requests.jsonl')
        self.assertEqual(full[:len(records)], records)
        before = (self.output/'requests.jsonl').read_bytes()
        with patch.object(MockTransport, '__call__', side_effect=AssertionError('duplicate request')):
            self.assertEqual(execute(plan, self.output, 'simulate', self.fixtures), report)
        self.assertEqual(before, (self.output/'requests.jsonl').read_bytes())

    def test_restart_mid_sample_replays_successful_calls(self):
        plan = self.plan()
        transport = MockTransport(self.fixtures)
        count = 0
        def interruption(payload, context):
            nonlocal count
            count += 1
            if count == 3:
                raise KeyboardInterrupt()
            return transport(payload, context)
        with self.assertRaises(KeyboardInterrupt):
            execute(plan, self.output, 'simulate', self.fixtures, transport=interruption)
        self.assertFalse((self.output/'run.lock').exists())
        self.assertEqual(analyze(self.output)['costs']['physical_total']['unknown_attempts'], 1)
        with self.assertRaises(UnknownAttempt):
            execute(plan, self.output, 'simulate', self.fixtures)
        report = execute(plan, self.output, 'simulate', self.fixtures, retry_unknown=True)
        self.assertEqual(report['costs']['physical_total']['call_attempts'], 57)
        records = read_journal(self.output/'requests.jsonl')
        first_scores = [r for r in records if r['kind']=='start' and r['sample_id']=='fix'
                        and r['stage']=='screening' and r['operation']=='Emotion/Sentiment']
        self.assertEqual(len(first_scores), 1)

    def test_plan_binding_rejects_data_config_code_and_fixture_mode_drift(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            prepare(dict(self.config, trigger_cap=.95), self.output)
        execute(plan, self.output, 'simulate', self.fixtures, max_samples=1)
        with self.assertRaisesRegex(ValueError, 'mode/fixture'):
            execute(plan, self.output, 'simulate', {})
        with self.assertRaisesRegex(ValueError, 'mode/fixture'):
            execute(plan, self.output, 'live', transport=MockTransport())
        raw = json.loads((self.output/'plan.json').read_text())
        raw['config']['model'] = 'tampered'
        (self.output/'plan.json').write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            analyze(self.output)

    def test_changed_input_and_source_fingerprint_refuse_reuse(self):
        source = self.root/'source.csv'
        source.write_text('text,label\nbody,0\n')
        config = dict(self.config, input_csv=str(source))
        prepare(config, self.output)
        source.write_text('text,label\nchanged body,0\n')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            prepare(config, self.output)
        source.write_text('text,label\nbody,0\n')
        with patch('experiments.protocol.file_hash', return_value='changed-source-fingerprint'):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                prepare(config, self.output)

    def test_all_requests_fail_without_confident_predictions(self):
        plan = self.plan()
        def failing(payload, context):
            raise TimeoutError('synthetic')
        report = execute(plan, self.output, 'simulate', transport=failing)
        self.assertEqual(report['routing_failed_count'], 7)
        self.assertEqual(report['triggered_count'], 0)
        self.assertIsNone(report['trigger_rate_over_routed'])
        for method in ('first_only', 'full_two_layer', 'direct_llm'):
            self.assertEqual(report['comparisons']['all']['metrics'][method]['scored'], 0)
            self.assertIsNone(report['comparisons']['all']['metrics'][method]['accuracy'])
        self.assertEqual(report['costs']['physical_total']['attempt_failure_rate'], 1.)
        self.assertEqual(report['costs']['physical_total']['request_failure_rate'], 1.)

    def test_success_before_sample_commit_replayed_without_network(self):
        plan = self.plan()
        journal = Journal(self.output, plan, 'simulate', MockTransport(self.fixtures), digest(self.fixtures))
        expected = run_sample(plan['samples'][0], plan['config'], journal)
        before = (self.output/'requests.jsonl').read_bytes()
        replay = Journal(self.output, plan, 'simulate', lambda *_: self.fail('replayed call'), digest(self.fixtures))
        self.assertEqual(run_sample(plan['samples'][0], plan['config'], replay), expected)
        self.assertEqual(before, (self.output/'requests.jsonl').read_bytes())

    def test_self_consistency_uses_median_not_last_response(self):
        plan = self.plan(self_consistency_n=3)
        normal = MockTransport()
        def varied(payload, context):
            response = normal(payload, context)
            if context['stage'] == 'screening':
                response['choices'][0]['message']['content'] = str([.2, .9, .3][context['ordinal']])
            return response
        journal = Journal(self.output, plan, 'simulate', varied, digest({}))
        result = run_sample(plan['samples'][0], plan['config'], journal)
        self.assertAlmostEqual(result['stages']['first_only']['p_positive'], .3)
        self.assertFalse(result['triggered'])

    def test_journal_corruption_and_torn_tail_refuse_resume(self):
        plan = self.plan()
        execute(plan, self.output, 'simulate', self.fixtures, max_samples=1)
        path = self.output/'requests.jsonl'
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            execute(plan, self.output, 'simulate', self.fixtures)
        path.write_bytes(original.replace(b'mock-model-v1', b'mock-model-v2', 1))
        with self.assertRaisesRegex(ValueError, 'hash chain'):
            analyze(self.output)

    def test_labels_ids_and_visible_inputs_do_not_leak_gold(self):
        source = self.root/'input.csv'
        source.write_text('id,text,label,teacher_reason\n5.5e17,body A,0,SECRET\n5.5e17,body B,1,SECRET\n')
        plan = self.plan(input_csv=str(source))
        self.assertEqual([s['target'] for s in plan['samples']], [1, 0])
        self.assertNotEqual(plan['samples'][0]['sample_id'], plan['samples'][1]['sample_id'])
        execute(plan, self.output, 'simulate')
        for record in read_journal(self.output/'requests.jsonl'):
            if record['kind'] == 'start':
                payload = json.dumps(record['payload'])
                self.assertNotIn('SECRET', payload)
                self.assertNotIn('5.5e17', payload)
                self.assertNotIn('source_label', payload)

    def test_input_rejection_duplicate_identity_empty_and_wrong_task_labels(self):
        for content in ('sample_id,text,label\nx,a,0\nx,b,1\n', 'text,label\n,0\n',
                        'text,label\na,rumour\n', 'text,label\n'):
            source = self.root/'bad.csv'
            source.write_text(content)
            with self.assertRaises(ValueError):
                prepare(dict(self.config, input_csv=str(source)), self.output)

    def test_pheme_is_explicit_adaptation_with_rumour_probabilities(self):
        plan = self.plan(label_schema='pheme_rumour_v1')
        self.assertEqual(plan['method'], 'reference_rumour_adaptation_v1')
        execute(plan, self.output, 'simulate', self.fixtures, max_samples=1)
        prediction = json.loads((self.output/'predictions.json').read_text())[0]
        self.assertEqual(prediction['target'], 0)
        self.assertIn('p_rumour', prediction['stages']['first_only'])
        self.assertNotIn('p_fake', prediction['stages']['first_only'])
        for record in read_journal(self.output/'requests.jsonl'):
            if record['kind'] == 'start':
                prompt = ' '.join(m['content'] for m in record['payload']['messages'])
                self.assertIn('rumour', prompt)

    def test_original_metadata_and_body_context_variants(self):
        plan = self.plan()
        execute(plan, self.output, 'simulate', self.fixtures, max_samples=1)
        records = read_journal(self.output/'requests.jsonl')
        prompt = [r['payload'] for r in records if r['kind']=='start' and r['operation']=='reference'][0]
        self.assertNotIn('Synthetic fabricated claim A', json.dumps(prompt))
        self.output = self.root/'with_content'
        plan = self.plan(reference_context='with_content')
        execute(plan, self.output, 'simulate', self.fixtures, max_samples=1)
        records = read_journal(self.output/'requests.jsonl')
        prompt = [r['payload'] for r in records if r['kind']=='start' and r['operation']=='reference'][0]
        self.assertIn('Synthetic fabricated claim A', json.dumps(prompt))

    def test_routing_boundaries_and_self_consistency_median(self):
        plan = self.plan(self_consistency_n=3)
        sample = plan['samples'][0]
        for score, triggered in ((.3, False), (.300001, True), (.7, True), (.89999, True), (.9, False)):
            out = self.root/str(score)
            out.mkdir()
            fixture = {'fix': {'first': score}}
            journal = Journal(out, plan, 'simulate', MockTransport(fixture), digest(fixture))
            result = run_sample(sample, plan['config'], journal)
            self.assertEqual(result['triggered'], triggered)
            self.assertAlmostEqual(result['stages']['first_only']['p_positive'], score)
            self.assertEqual(len([r for r in journal.records if r['kind']=='start' and r['stage']=='screening']), 12)

    def test_metrics_empty_single_class_ties_and_macro_f1(self):
        def row(y, p):
            return dict(target=y, stages={'first_only': dict(status='success', p_positive=p)})
        self.assertIsNone(metrics([], 'first_only')['accuracy'])
        self.assertIsNone(metrics([row(1, .7)], 'first_only')['roc_auc'])
        m = metrics([row(1, .5), row(0, .5)], 'first_only')
        self.assertEqual(m['roc_auc'], .5)
        self.assertAlmostEqual(m['class_macro_f1'], 1/3)
        m = metrics([row(1, .9), row(0, .2)], 'first_only')
        self.assertEqual((m['roc_auc'], m['binary_f1'], m['class_macro_f1']), (1., 1., 1.))

    def test_invalid_direct_response_and_truncation_not_accepted(self):
        for text in ('{"p_positive":"0.7","reason":"x"}', '{"p_positive":true,"reason":"x"}',
                     '{"p_positive":1.5,"reason":"x"}', '{"p_positive":NaN,"reason":"x"}', '{}'):
            with self.assertRaises(ResponseError):
                response_content(dict(choices=[dict(message=dict(content=text))]), 'direct')
        with self.assertRaises(ResponseError):
            response_content(dict(choices=[dict(message=dict(content='.5'), finish_reason='length')]), 'score')

    def test_malformed_response_envelope_still_archived_and_costed(self):
        plan = self.plan()
        invalid = [{'error': 'synthetic malformed envelope'}]
        report = execute(plan, self.output, 'simulate', max_samples=1, transport=lambda *_: invalid)
        self.assertEqual(report['costs']['physical_total']['call_attempts'], 6)
        self.assertEqual(report['costs']['physical_total']['failure_types'], {'ResponseError': 6})
        self.assertIsNone(report['costs']['physical_total']['token_totals']['total_tokens'])
        records = read_journal(self.output/'requests.jsonl')
        self.assertTrue(all(r['response'] == invalid for r in records if r['kind']=='finish'))

    def test_live_transport_one_post_and_http_error_body_retained(self):
        import io
        import urllib.error
        transport = LiveTransport(self.config)
        error = urllib.error.HTTPError('https://example.invalid', 429, 'limited', {}, io.BytesIO(b'{"error":"rate limited"}'))
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'not-a-real-key'}), \
             patch('urllib.request.build_opener') as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(HTTPFailure) as caught:
                transport({'model': 'offline'}, {})
            self.assertEqual(caught.exception.response['http_status'], 429)
            self.assertIn('rate limited', caught.exception.response['raw_body'])
            self.assertEqual(opener.return_value.open.call_count, 1)

    def test_config_rejects_invalid_and_unknown_fields(self):
        for key, value in (('max_attempts', 0), ('self_consistency_n', True), ('first_weights', [-1,1,1,1]),
                           ('trigger_cap', .6), ('timeout', 0), ('api_key', 'secret'),
                           ('base_url', 'https://user:secret@example.com'), ('reference_context', 'unknown')):
            path = self.root/'config.json'
            path.write_text(json.dumps(dict(input_csv='x', label_schema='archive_fake_v1', **{key:value})))
            with self.assertRaises(ValueError):
                configuration(path)

    def test_second_process_resume_and_analyze_only(self):
        # The subprocess uses --simulate only; no SDK and no live execution switch.
        args = [sys.executable, '-X', 'utf8', str(ROOT/'run_review_experiment.py'), '--config',
                str(ROOT/'examples/review_archive.json'), '--output', str(self.output), '--simulate']
        first = subprocess.run(args + ['--max-samples', '2'], capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        second = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        before = (self.output/'requests.jsonl').read_bytes()
        third = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'run_review_experiment.py'), '--output',
            str(self.output), '--analyze-only'], capture_output=True, text=True)
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual(before, (self.output/'requests.jsonl').read_bytes())

    def test_exclusive_writer_lock(self):
        plan = self.plan()
        (self.output/'run.lock').write_text('other process')
        with self.assertRaisesRegex(RuntimeError, 'locked'):
            execute(plan, self.output, 'simulate')
        self.assertEqual((self.output/'run.lock').read_text(), 'other process')


if __name__ == '__main__':
    unittest.main(verbosity=2)
