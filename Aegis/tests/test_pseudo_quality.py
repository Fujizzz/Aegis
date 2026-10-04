"""No external calls or training: identity, partial failures, lineage and release gates."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import pseudo_quality as q
from utils import freeze_pheme as f


def fixture(root):
    root.mkdir()
    rows = []
    for event in f.EVENTS:
        for split in f.SPLITS:
            for label in ('0', '1'):
                ident = str(100000000000000000 + len(rows))
                text = f'Unique {event} {split} {label} factual body @person https://example.org'
                row = dict(sample_id='pheme:' + ident, id=ident, thread_id=ident, reply_to_id='',
                           event_name=event, split=split, label=label, text=text,
                           text_sha256=f.digest(text), normalized_text_sha256=f.digest(f.normalize(text)),
                           source_member='source/' + ident, source_json_sha256=f.digest(ident),
                           group_id='group:' + ident, pseudo_source_records='',
                           pseudo_status='regeneration_required' if split == 'train' and label == '1' else q.READY)
                for dim in q.DIMS:
                    row.update({dim + '_label': '.2', dim + '_score_raw': '.2', dim + '_parse_ok': '1',
                                dim + '_parse_method': 'strict_json', dim + '_rationale': 'Slight degree.',
                                dim + '_raw': '{"score":0.2,"rationale":"Slight degree."}'})
                    if row['pseudo_status'] != q.READY:
                        for key in list(row):
                            if key.startswith(dim + '_'):
                                row[key] = ''
                rows.append(row)
    f.write_csv(root / 'canonical.csv', rows, list(rows[0]))
    mf = ['sample_id', 'id', 'event_name', 'split', 'group_id', 'pseudo_status', 'pseudo_source_records']
    f.write_csv(root / 'split_manifest.csv', [{k:r[k] for k in mf} for r in rows], mf)
    for event in f.EVENTS:
        for split in f.SPLITS:
            f.write_csv(root / ('split_' + f.STEMS[event]) / (split + '.csv'),
                        [r for r in rows if r['event_name'] == event and r['split'] == split], list(rows[0]))
    fields = ['sample_id', 'event_name', 'split', 'text', 'pseudo_status']
    f.write_csv(root / 'pseudo_regeneration_queue.csv',
                [{k:r[k] for k in fields} for r in rows if r['pseudo_status'] != q.READY], fields)
    f.write_csv(root / 'legacy_mapping.csv', [], ['source_kind', 'candidate_ids'])
    f.write_csv(root / 'group_edges.csv', [], ['a', 'b'])
    f.write_csv(root / 'mixed_label_group_review.csv', [], ['sample_id'])
    q.write_json(root / 'protocol.json', {'seed':42, 'version':'fixture', 'inputs':[]})
    q.write_json(root / 'audit_summary.json', {})
    q.write_json(root / 'freeze_manifest.json', {'version':'fixture', 'files':
        {p.relative_to(root).as_posix():f.file_hash(p) for p in root.rglob('*') if p.is_file()}})
    (root / 'FREEZE.sha256').write_text(f.file_hash(root / 'freeze_manifest.json') + ' freeze_manifest.json\n')
    return rows


def mock_response(*unused):
    return dict(id='test-only', model='mock-not-a-teacher', choices=[dict(finish_reason='stop',
        message=dict(content='{"score":0.2,"rationale":"Synthetic response, not research data."}'))])


def fill_synthetic_reviews(audit, path):
    rows = f.read_csv(audit / 'human_review_blind.csv')
    for row in rows:
        row.update(score='.2', reviewer_id='SYNTHETIC_TEST_' + row['annotator_slot'],
                   rationale='Synthetic fixture only', critical_issue='0')
    f.write_csv(path, rows, list(rows[0]))


def teacher_decision(audit, path):
    q.write_json(path, dict(schema='teacher_only_release_decision_v1', release_policy='teacher_only',
        user_instruction='SYNTHETIC TEST: retain original teacher scores, no human review.',
        audit_sha256=f.file_hash(audit / 'audit_manifest.json')))


class PseudoQualityTests(unittest.TestCase):
    def test_teacher_only_release_preserves_scores_without_claiming_human_review(self):
        from utils.data_loader import NewsDataset
        from utils.label_schema import get_label_schema
        from utils.pseudo_policy import TEACHER_ONLY, training_release_ready
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'TEST_TEACHER_KEY':'synthetic'}):
            root = Path(temp)
            rows = fixture(root / 'source')
            plan = q.make_plan(root / 'source', rows)
            log = root / 'responses.jsonl'
            q.generate(plan, log, 'https://example.org', execute=True,
                       api_key_env='TEST_TEACHER_KEY', transport=mock_response)
            q.create_audit(root / 'source', root / 'audit', responses=log)
            teacher_decision(root / 'audit', root / 'decision.json')
            accepted, _ = q.load_responses(log, plan)
            expected = q.apply_responses(rows, plan, accepted)
            q.build_release(root / 'source', root / 'audit', None, root / 'teacher_v2',
                            release_policy='teacher_only', decision_record=root / 'decision.json')
            actual = f.read_csv(root / 'teacher_v2/canonical.csv')
            for before, after in zip(expected, actual):
                self.assertEqual(before, {k:after[k] for k in before})
                if after['split'] == 'train':
                    self.assertEqual(after['pseudo_quality_status'], TEACHER_ONLY)
            self.assertTrue(training_release_ready(actual))
            self.assertFalse(q.read_json(root / 'teacher_v2/protocol.json')['human_review_completed'])
            quality = q.read_json(root / 'teacher_v2/pseudo_provenance/quality_review.json')
            self.assertEqual(quality['status'], 'accepted_teacher_only_not_human_validated')
            self.assertFalse((root / 'teacher_v2/pseudo_provenance/human_annotations.csv').exists())
            dataset = NewsDataset(root / 'teacher_v2/split_charliehebdo/train.csv', None,
                                  label_schema=get_label_schema('pheme_rumour_v1'))
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]['emotion_label'], .2)
            f.verify(root / 'source', recompute_near=False)

    def test_teacher_only_still_requires_complete_responses_and_bound_decision(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'TEST_TEACHER_KEY':'synthetic'}):
            root = Path(temp)
            rows = fixture(root / 'source')
            plan = q.make_plan(root / 'source', rows)
            log = root / 'responses.jsonl'
            q.generate(plan, log, 'https://example.org', execute=True, limit=1,
                       api_key_env='TEST_TEACHER_KEY', transport=mock_response)
            q.create_audit(root / 'source', root / 'partial', responses=log)
            teacher_decision(root / 'partial', root / 'partial_decision.json')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                q.build_release(root / 'source', root / 'partial', None, root / 'blocked',
                    release_policy='teacher_only', decision_record=root / 'partial_decision.json')
            q.generate(plan, log, 'https://example.org', execute=True,
                       api_key_env='TEST_TEACHER_KEY', transport=mock_response)
            q.create_audit(root / 'source', root / 'audit', responses=log)
            with self.assertRaises(ValueError):
                q.build_release(root / 'source', root / 'audit', None, root / 'blocked',
                    release_policy='teacher_only', decision_record=root / 'partial_decision.json')
            with self.assertRaises(ValueError):
                q.build_release(root / 'source', root / 'audit', None, root / 'blocked')
            with self.assertRaises(ValueError):
                q.build_release(root / 'source', root / 'audit', None, root / 'blocked', release_policy='teacher_only')
            self.assertFalse((root / 'blocked').exists())

    def test_unknown_or_pending_markers_cannot_enable_teacher_supervision(self):
        from utils.pseudo_policy import TEACHER_ONLY, HUMAN_REVIEWED, training_release_ready
        rows = [dict(split='train', pseudo_status=q.GENERATED, pseudo_quality_status=TEACHER_ONLY)]
        self.assertTrue(training_release_ready(rows))
        self.assertFalse(training_release_ready([]))
        self.assertFalse(training_release_ready(rows + [dict(rows[0], pseudo_quality_status=HUMAN_REVIEWED)]))
        self.assertFalse(training_release_ready([dict(rows[0], pseudo_status='regeneration_required')]))
        self.assertFalse(training_release_ready([dict(rows[0], pseudo_quality_status='approved')]))

    def test_parallel_failure_drains_inflight_and_resume_skips_successes(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'TEST_TEACHER_KEY':'synthetic'}):
            root = Path(temp)
            rows = fixture(root / 'source')
            plan = q.make_plan(root / 'source', rows)
            log = root / 'responses.jsonl'
            barrier = threading.Barrier(3)
            error_committed = threading.Event()
            lock = threading.Lock()
            calls = []
            real_fsync = os.fsync
            def transport(payload, *unused):
                with lock:
                    index = len(calls)
                    calls.append(payload)
                barrier.wait(timeout=5)
                if index == 0:
                    raise RuntimeError('secret must not appear in log')
                if not error_committed.wait(timeout=5):
                    raise AssertionError('The failed request was not committed')
                return mock_response()
            def fsync_and_signal(fd):
                real_fsync(fd)
                error_committed.set()
            with patch.object(q.os, 'fsync', side_effect=fsync_and_signal):
                result = q.generate(plan, log, 'https://example.org', execute=True,
                    api_key_env='TEST_TEACHER_KEY', workers=3, transport=transport)
            self.assertEqual(len(calls), 3)
            self.assertEqual(result['counts'], {'api_error':1, 'ok':2})
            accepted, records = q.load_responses(log, plan)
            self.assertEqual(len(records), 3)
            self.assertNotIn('secret must not appear', log.read_text())
            completed_payloads = {q.canonical(r['payload']) for r in plan['requests'] if r['request_id'] in accepted}
            def resume_transport(payload, *unused):
                self.assertNotIn(q.canonical(payload), completed_payloads)
                return mock_response()
            result = q.generate(plan, log, 'https://example.org', execute=True,
                api_key_env='TEST_TEACHER_KEY', workers=3, transport=resume_transport)
            self.assertEqual(result['completed_calls'], 20)
            accepted, records = q.load_responses(log, plan)
            self.assertEqual(len(records), 21)
            before = log.read_bytes()
            q.generate(plan, log, 'https://example.org', execute=True,
                api_key_env='TEST_TEACHER_KEY', workers=3, transport=transport)
            self.assertEqual(log.read_bytes(), before)

    def test_parallel_limit_and_worker_validation(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'TEST_TEACHER_KEY':'synthetic'}):
            root = Path(temp)
            rows = fixture(root / 'source')
            plan = q.make_plan(root / 'source', rows)
            log = root / 'responses.jsonl'
            for workers in (0, 9, True, 1.5):
                with self.assertRaises(ValueError):
                    q.generate(plan, log, 'https://example.org', workers=workers)
            result = q.generate(plan, log, 'https://example.org', execute=True,
                api_key_env='TEST_TEACHER_KEY', workers=4, limit=8, transport=mock_response)
            self.assertEqual(result['completed_calls'], 8)
            accepted, records = q.load_responses(log, plan)
            self.assertEqual(len(records), 8)
            self.assertEqual(sum(r['pseudo_status'] == q.GENERATED for r in q.apply_responses(rows, plan, accepted)), 2)

    def test_strict_rejects_coercion_duplicates_and_bad_completion(self):
        for raw in ['{"score":true,"rationale":"a"}', '{"score":1.2,"rationale":"a"}',
                    '{"score":"0.2","rationale":"a"}', '{"score":NaN,"rationale":"a"}',
                    '{"score":0.3,"rationale":"a"}', '{"score":0.2,"rationale":""}',
                    '{"score":0,"score":0.2,"rationale":"a"}', 'score=0.2',
                    '{"score":0.2,"rationale":"a","extra":1}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                q.strict_score(raw)
        response = mock_response()
        response['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(ValueError):
            q.response_score({'response':response})

    def test_default_dry_run_makes_no_network_call_and_no_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixture(root / 'source')
            plan = q.make_plan(root / 'source', rows)
            def forbidden(*args):
                self.fail('Network called in dry run')
            result = q.generate(plan, root / 'responses', 'https://example.org', transport=forbidden)
            self.assertFalse(result['network_called'])
            self.assertEqual(result['planned_calls'], 20)
            self.assertFalse((root / 'responses').exists())
            self.assertTrue(all(r['split'] == 'train' and 'label' not in r for r in plan['requests']))

    def test_partial_failure_resume_and_hash_tampering(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'TEST_TEACHER_KEY':'synthetic'}):
            root = Path(temp)
            rows = fixture(root / 'source')
            plan = q.make_plan(root / 'source', rows)
            log = root / 'responses.jsonl'
            progress = q.generate(plan, log, 'https://example.org', execute=True, api_key_env='TEST_TEACHER_KEY',
                                  limit=3, transport=mock_response)
            self.assertEqual(progress['status'], 'partial')
            self.assertEqual(progress['remaining_calls'], 17)
            accepted, _ = q.load_responses(log, plan)
            self.assertEqual(q.apply_responses(rows, plan, accepted), rows)
            def fail(*args):
                raise RuntimeError('secret text must not be saved')
            q.generate(plan, log, 'https://example.org', execute=True, api_key_env='TEST_TEACHER_KEY', transport=fail)
            self.assertNotIn('secret text', log.read_text())
            progress = q.generate(plan, log, 'https://example.org', execute=True, api_key_env='TEST_TEACHER_KEY', transport=mock_response)
            self.assertEqual(progress['status'], 'complete')
            accepted, records = q.load_responses(log, plan)
            self.assertEqual(len(accepted), 20)
            self.assertEqual(len(records), 21)
            merged = q.apply_responses(rows, plan, accepted)
            self.assertEqual(sum(r['pseudo_status'] == q.GENERATED for r in merged), 5)
            before = log.read_bytes()
            q.generate(plan, log, 'https://example.org', execute=True, api_key_env='TEST_TEACHER_KEY', transport=fail)
            self.assertEqual(before, log.read_bytes())
            changed = copy.deepcopy(plan)
            changed['requests'][0]['text_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                q.load_responses(log, changed)
            log.write_text(log.read_text().replace('Synthetic response', 'Changed response'), encoding='utf-8')
            with self.assertRaises(ValueError):
                q.load_responses(log, plan)

    def test_review_gate_release_membership_and_parent_immutable(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'TEST_TEACHER_KEY':'synthetic'}):
            root = Path(temp)
            rows = fixture(root / 'source')
            source = root / 'source'
            parent_hash = f.file_hash(source / 'freeze_manifest.json')
            plan = q.make_plan(source, rows)
            log = root / 'responses.jsonl'
            q.generate(plan, log, 'https://example.org', execute=True, api_key_env='TEST_TEACHER_KEY', transport=mock_response)
            audit = root / 'audit'
            q.create_audit(source, audit, responses=log)
            with self.assertRaises(ValueError):
                q.review_metrics(audit, audit / 'human_review_blind.csv')
            reviews = root / 'reviews.csv'
            fill_synthetic_reviews(audit, reviews)
            self.assertEqual(q.review_metrics(audit, reviews)['status'], 'passed')
            result = q.build_release(source, audit, reviews, root / 'v2')
            self.assertEqual(result['status'], 'passed')
            built = f.read_csv(root / 'v2' / 'canonical.csv')
            immutable = ['sample_id', 'label', 'text', 'split', 'group_id', 'source_member', 'text_sha256']
            self.assertEqual([[r[k] for k in immutable] for r in rows], [[r[k] for k in immutable] for r in built])
            self.assertEqual(f.file_hash(source / 'freeze_manifest.json'), parent_hash)
            with self.assertRaises(FileExistsError):
                q.build_release(source, audit, reviews, root / 'v2')

    def test_incomplete_and_failed_review_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'source')
            q.create_audit(root / 'source', root / 'audit')
            reviews = root / 'reviews.csv'
            fill_synthetic_reviews(root / 'audit', reviews)
            with self.assertRaises(ValueError):
                q.build_release(root / 'source', root / 'audit', reviews, root / 'v2')
            self.assertFalse((root / 'v2').exists())
            entries = f.read_csv(reviews)
            entries[0]['critical_issue'] = '1'
            f.write_csv(reviews, entries, list(entries[0]))
            self.assertEqual(q.review_metrics(root / 'audit', reviews)['status'], 'failed')
            entries[0]['display_sha256'] = 'wrong'
            f.write_csv(reviews, entries, list(entries[0]))
            with self.assertRaises(ValueError):
                q.review_metrics(root / 'audit', reviews)

    def test_data_loader_requires_reviewed_generated_labels(self):
        from utils.data_loader import NewsDataset
        from utils.label_schema import get_label_schema
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixture(root / 'source')[:1]
            rows[0]['pseudo_status'] = q.GENERATED
            path = root / 'row.csv'
            f.write_csv(path, rows, list(rows[0]))
            kwargs = dict(tokenizer=None, label_schema=get_label_schema('pheme_rumour_v1'))
            with self.assertRaisesRegex(ValueError, 'need verified'):
                NewsDataset(path, **kwargs)
            rows[0]['pseudo_quality_status'] = 'human_screen_passed_v1'
            f.write_csv(path, rows, list(rows[0]))
            self.assertEqual(len(NewsDataset(path, **kwargs)), 1)

    def test_source_cannot_be_used_as_audit_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'source')
            with self.assertRaises(ValueError):
                q.create_audit(root / 'source', root / 'source' / 'audit')
            self.assertFalse((root / 'source' / 'audit').exists())


if __name__ == '__main__':
    unittest.main()
