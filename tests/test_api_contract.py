"""Offline API contract tests; SDK transport is stubbed, never calls a service."""
import csv
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'Agent'))
from mafnd.detector import MultiAgentDetector, DEFAULT_WEIGHTS
from mafnd.models import NewsItem, VerificationResult
from mafnd.providers import DeepSeekProvider
from mafnd.cli import main as cli
from pipeline import detect_item
from probability import parse_probability, normalize_comparison
from reporter.report_generator import ReportGenerator

# Only constructor/transport stand-ins. Production parsers and orchestration run.
with patch.dict(sys.modules, {'openai': types.SimpleNamespace(OpenAI=Mock()),
                             'httpx': types.SimpleNamespace(HTTPTransport=Mock(), Client=Mock())}):
    from agent.deepseek_agent import DeepSeekAgent
    from detector.first_layer_detector import FirstLayerDetector
    from detector.second_layer_detector import SecondLayerDetector


class APIContractTests(unittest.TestCase):
    def test_strict_numbers_both_implementations(self):
        for parse in (parse_probability, DeepSeekProvider._number):
            for text, expected in [('0', 0), ('1', 1), ('80.5%', .805), ('1e-2', .01), (' 0.25 ', .25)]:
                self.assertEqual(parse(text), expected)
            for text in ('-0.2', '1.5', '-20%', '101%', 'NaN', 'inf', 'score 0.8', '0.2 or 0.9', '80.5'):
                with self.subTest(text=text), self.assertRaises(ValueError): parse(text)

    def test_weights_thresholds_and_counts(self):
        for bad in (-.1, float('nan'), float('inf'), True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MultiAgentDetector(Mock(), weights=dict(DEFAULT_WEIGHTS, emotion=bad))
        for options in (dict(weights={}), dict(weights=dict.fromkeys(DEFAULT_WEIGHTS, 0)),
                        dict(real_threshold=float('nan')), dict(fake_threshold=1.1), dict(real_threshold=.55),
                        dict(second_layer_fake_cap=.6), dict(second_layer_fake_cap=float('inf')),
                        dict(self_consistency=True), dict(self_consistency=1.5)):
            with self.assertRaises(ValueError): MultiAgentDetector(Mock(), **options)

    def test_release_stage_probability_confidence_and_failures(self):
        provider = Mock()
        for score, verified, stage, label, p in ((.2, .8, 'screening', 'real', .2),
                                               (.5, .2, 'verification', 'real', .2),
                                               (.8, .9, 'verification', 'fake', .9),
                                               (.95, .2, 'screening', 'fake', .95)):
            provider.score.return_value = score
            provider.verify.return_value = VerificationResult(verified, 'test')
            result = MultiAgentDetector(provider).detect(NewsItem('news')).to_dict()
            self.assertEqual((result['stage'], result['label'], result['status']), (stage, label, 'success'))
            self.assertAlmostEqual(result['p_fake'], p)
            self.assertAlmostEqual(result['class_confidence'], p if label == 'fake' else 1-p)
        for bad in (-.2, 1.5, float('nan'), float('inf')):
            provider.score.return_value = bad
            result = MultiAgentDetector(provider).detect(NewsItem('news'))
            self.assertEqual((result.status, result.stage), ('error', 'screening'))
            self.assertIsNone(result.p_fake)
            provider.score.return_value = .5
            provider.verify.return_value = VerificationResult(bad, '')
            result = MultiAgentDetector(provider).detect(NewsItem('news'))
            self.assertEqual((result.status, result.stage), ('error', 'verification'))

    def test_release_verifier_missing_probability_is_error(self):
        provider = DeepSeekProvider('offline-placeholder')
        for payload in ('{}', '{"fake_probability": 1.5}', '{"fake_probability": NaN}'):
            with patch.object(provider, '_chat', return_value=payload), self.assertRaises((KeyError, ValueError)):
                provider.verify(NewsItem('news'), {})

    def test_original_weights_and_failure_propagation(self):
        aspects = {'Emotion/Sentiment': 'x', 'Exaggeration': 'x', 'Bias': 'x', 'Commonsense Conflict': 'x'}
        weights = {'情绪煽动': .3, '夸张程度': .25, '偏见倾向': .3, '常识冲突': .15}
        with patch.dict(FirstLayerDetector.__init__.__globals__, DeepSeekAgent=Mock()):
            first = FirstLayerDetector('offline', aspects, weights=weights)
        self.assertEqual(list(first.weights.values()), [.3, .25, .3, .15])
        for value in first.agents.values(): value.analyze_multi.side_effect = RuntimeError('offline failure')
        with self.assertRaises(RuntimeError): first.analyze_news('', 'news')
        for bad in ({}, dict(weights, 情绪煽动=-1), dict(weights, 情绪煽动=float('nan'))):
            with self.assertRaises(ValueError): FirstLayerDetector('offline', aspects, weights=bad)

    def test_original_sdk_failure_is_not_a_score(self):
        agent = DeepSeekAgent.__new__(DeepSeekAgent)
        agent.prompt_template = '{content}'; agent.role = 'test'; agent.model = 'offline'
        agent.temperature = .1; agent.max_tokens = 512; agent.logger = Mock(); agent.client = Mock()
        agent.client.chat.completions.create.side_effect = RuntimeError('offline failure')
        with self.assertRaises(RuntimeError): agent.analyze('', 'news')
        with self.assertRaises(RuntimeError): agent.compare_closeness('news', 'true', 'fake')
        for payload in ({}, {'closer': 'fake', 'proximity_true': -1, 'proximity_fake': 2},
                        {'closer': 'tie', 'proximity_true': 0, 'proximity_fake': 0}):
            with self.assertRaises((ValueError, KeyError)): normalize_comparison(payload)

    def test_original_second_layer_returns_p_fake(self):
        second = SecondLayerDetector.__new__(SecondLayerDetector)
        second.logger = Mock(); second.aspects = ['a']; second.aspect_weights = {'a': 1}
        second._generate_true_fake = Mock(return_value=('true', 'fake'))
        second.compare_agents = {'a': Mock()}
        second.compare_agents['a'].compare_closeness.return_value = dict(closer='true', proximity_true=.8, proximity_fake=.2)
        _, label, p = second.analyze_news('', 'news')
        self.assertEqual(label, 'real'); self.assertLess(p, .5)
        second._generate_true_fake.side_effect = RuntimeError('offline failure')
        with self.assertRaises(RuntimeError): second.analyze_news('', 'news')

    def test_original_pipeline_csv_keeps_precision_and_abstentions(self):
        first, second = Mock(), Mock()
        first.analyze_news.return_value = (.5, 'uncertain')
        second.analyze_news.return_value = ('test', 'real', .123456789012345)
        good = detect_item({}, first, second)
        self.assertEqual(good['p_fake'], .123456789012345)
        self.assertEqual(good['class_confidence'], 1-good['p_fake'])
        second.analyze_news.side_effect = RuntimeError('offline failure')
        bad = detect_item({}, first, second)
        self.assertEqual((bad['status'], bad['stage']), ('error', 'verification'))
        self.assertIsNone(bad['p_fake'])
        first.analyze_news.side_effect = RuntimeError('offline failure')
        self.assertEqual(detect_item({}, first, second)['stage'], 'screening')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'results.csv'; report = ReportGenerator(path)
            report.save_results([dict(r, sample_id=str(i), title='', actual_label='real') for i, r in enumerate((good, bad))])
            with path.open(encoding='utf-8') as stream: rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]['p_fake']), good['p_fake']); self.assertEqual(rows[1]['p_fake'], '')
            with self.assertRaises(FileExistsError): report.save_results([])

    def test_release_cli_error_export_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'input.csv', Path(tmp)/'output.csv'
            source.write_text('content,label\nnews,1\n', encoding='utf-8')
            provider = Mock(); provider.score.side_effect = RuntimeError('offline failure')
            with patch('mafnd.cli._provider', return_value=provider):
                self.assertEqual(cli(['--input', str(source), '--output', str(output)]), 1)
            with output.open() as stream: row = next(csv.DictReader(stream))
            self.assertEqual(row['status'], 'error'); self.assertEqual(row['p_fake'], '')

    def test_existing_release_tests(self):
        for filename in ('test_detector.py', 'test_cli.py'):
            path = ROOT/'tests'/filename
            spec = importlib.util.spec_from_file_location('legacy_'+path.stem, path)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            for name in dir(module):
                if name.startswith('test_'):
                    function = getattr(module, name)
                    if function.__code__.co_argcount:
                        with tempfile.TemporaryDirectory() as tmp: function(Path(tmp))
                    else: function()


if __name__ == '__main__':
    unittest.main(verbosity=2)
