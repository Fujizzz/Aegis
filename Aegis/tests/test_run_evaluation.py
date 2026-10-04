import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.metrics import binary_metrics, event_metrics
from utils.run_archive import model_manifest, prepare_run, resolve_config, write_json
from evaluate import REQUIRED, evaluate_files, write_report


class MetricProtocolTests(unittest.TestCase):
    def test_known_confusion_and_tied_auc(self):
        m = binary_metrics([0, 0, 1, 1], [.1, .5, .5, .9])
        self.assertEqual([m[k] for k in ('tp', 'fp', 'tn', 'fn')], [2, 1, 1, 0])
        self.assertAlmostEqual(m['binary_f1'], .8)
        self.assertAlmostEqual(m['class_macro_f1'], (.8 + 2/3)/2)
        self.assertEqual(m['roc_auc'], .875)

    def test_auc_matches_independent_pairwise_definition(self):
        rng = np.random.default_rng(102)
        for _ in range(30):
            y = rng.integers(0, 2, size=31)
            p = rng.integers(0, 6, size=31) / 5
            pairs = [float(a > b) + .5 * float(a == b) for a in p[y == 1] for b in p[y == 0]]
            self.assertAlmostEqual(binary_metrics(y, p)['roc_auc'], sum(pairs)/len(pairs))

    def test_invalid_and_empty_inputs_fail(self):
        for y, p in [([], []), ([0], []), ([.7], [.1]), ([2], [.2]), ([0], [np.nan]),
                     ([0], [np.inf]), ([0], [-.1]), ([0], [1.1]), ([[0]], [[.2]])]:
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                binary_metrics(y, p)
        with self.assertRaises(ValueError):
            binary_metrics([0], [.2], np.nan)

    def test_single_class_fixed_two_class_macro_and_json_null(self):
        m = binary_metrics([0, 0], [.1, .2])
        self.assertEqual(m['binary_f1'], 0)
        self.assertEqual(m['class_macro_f1'], .5)
        self.assertIsNone(m['roc_auc'])
        json.dumps(m, allow_nan=False)

    def test_event_mean_and_pooled_are_distinct(self):
        result = event_metrics(['a', 'a', 'b', 'b', 'b', 'b'], [0, 1, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0])
        self.assertEqual(result['event_equal_mean']['accuracy'], .5)
        self.assertAlmostEqual(result['sample_pooled']['accuracy'], 1/3)
        single = event_metrics(['a', 'b'], [0, 1], [0, 1])
        self.assertIsNone(single['event_equal_mean']['roc_auc'])
        self.assertEqual(single['sample_pooled']['roc_auc'], 1)

    def rows(self):
        return [dict(sample_id=str(i), event='event', target=str(y), p_positive=str(p),
                     predicted_target=str(int(p >= .5)), label_schema='pheme_rumour_v1',
                     variant='full', seed='42', split='test', checkpoint_id='sha256:fixture')
                for i, (y, p) in enumerate([(0, .1), (1, .9)])]

    def write_predictions(self, path, rows):
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED))
            writer.writeheader()
            writer.writerows(rows)

    def test_csv_round_trip_and_no_report_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pred.csv'
            self.write_predictions(path, self.rows())
            result = evaluate_files([path])
            self.assertEqual(result['sample_pooled']['roc_auc'], 1)
            write_report(result, Path(tmp)/'report')
            self.assertEqual(json.loads((Path(tmp)/'report/metrics.json').read_text())['protocol']['threshold'], .5)
            with self.assertRaises(FileExistsError): write_report(result, Path(tmp)/'report')

    def test_csv_rejects_ambiguous_or_mixed_results(self):
        for key, value in [('sample_id', '0'), ('seed', '43'), ('variant', 'other'),
                           ('split', 'val'), ('label_schema', 'archive_fake_v1'),
                           ('predicted_target', '0'), ('p_positive', 'nan'), ('checkpoint_id', 'other')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                rows = self.rows()
                rows[1][key] = value
                path = Path(tmp)/'pred.csv'
                self.write_predictions(path, rows)
                with self.assertRaises(ValueError): evaluate_files([path])


class RunArchiveTests(unittest.TestCase):
    def fixture(self, root):
        (root/'model').mkdir()
        (root/'source').mkdir()
        (root/'model/config.json').write_text('{}')
        (root/'model/model.bin').write_bytes(b'tiny fixture, not a loadable model')
        (root/'source/train.py').write_text('print("fixture")')
        (root/'train.csv').write_text('label,text\n0,hello\n1,world\n')
        write_json(root/'model.json', model_manifest(root/'model'))
        return {'model': {'base_model': 'model', 'manifest': 'model.json'},
                'data': {'label_schema': 'pheme_rumour_v1', 'train_file': 'train.csv', 'val_file': 'train.csv'},
                'experiment': {'dataset': 'fixture', 'event': 'test', 'variant': 'full', 'method_id': 'test'},
                'runtime': {'runs_root': 'runs', 'seed': 42, 'run_id': 'fixed'}}

    def test_archive_isolated_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            resolved, run = prepare_run(cfg, root/'config.json', root/'source')
            self.assertEqual(run.relative_to(root).as_posix(), 'runs/fixture/test/full/seed_42/fixed')
            self.assertEqual(json.loads((run/'status.json').read_text())['status'], 'prepared')
            self.assertEqual(resolved['runtime']['select_metric'], 'binary_f1')
            self.assertTrue((run/'source/train.py').is_file())
            self.assertTrue((run/'environment/installed.txt').is_file())
            (root/'train.csv').write_text('changed')
            self.assertIn('hello', Path(resolved['data']['train_file']).read_text())
            with self.assertRaises(FileExistsError): prepare_run(cfg, root/'config.json', root/'source')
            cfg['runtime']['run_id'] = 'second'
            _, second = prepare_run(cfg, root/'config.json', root/'source')
            self.assertNotEqual(second, run)

    def test_model_tamper_and_secret_rejected_before_run_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            secret = copy.deepcopy(cfg)
            secret['api_key'] = 'private fixture'
            with self.assertRaises(ValueError): prepare_run(secret, root/'config.json', root/'source')
            (root/'model/model.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError): prepare_run(cfg, root/'config.json', root/'source')
            self.assertFalse((root/'runs').exists())

    def test_unsafe_names_threshold_and_invalid_accumulation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            for section, key, value in [('experiment', 'event', '../escape'), ('runtime', 'seed', -1),
                                        ('evaluation', 'threshold', .6)]:
                bad = copy.deepcopy(cfg)
                bad.setdefault(section, {})[key] = value
                with self.assertRaises(ValueError): resolve_config(bad, root/'config.json')
            for value in (0, -1, True, False, 1.0, 1.5, '2', None):
                bad = copy.deepcopy(cfg)
                bad['training'] = {'grad_accum_steps': value}
                with self.subTest(value=value), self.assertRaises(ValueError):
                    resolve_config(bad, root/'config.json')

    def test_accumulation_default_and_positive_integers_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            self.assertEqual(resolve_config(cfg, root/'config.json')['training']['grad_accum_steps'], 1)
            for value in (1, 2, 7):
                cfg['training'] = {'grad_accum_steps': value}
                resolved, run = prepare_run(cfg, root/'config.json', root/'source')
                self.assertEqual(resolved['training']['grad_accum_steps'], value)
                self.assertEqual(json.loads((run/'config.resolved.json').read_text())['training']['grad_accum_steps'], value)
                cfg['runtime']['run_id'] = f'next-{value}'


if __name__ == '__main__':
    unittest.main()
