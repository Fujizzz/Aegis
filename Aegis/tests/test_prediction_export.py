import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from test_training_core import fixture_trainer
from evaluate import evaluate_files
from utils.predictions import sample_identity


class PredictionExportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.silent = contextlib.redirect_stdout(io.StringIO()); self.silent.__enter__()
        self.progress = patch('training.trainer.tqdm', lambda data, **kw: data); self.progress.start()

    def tearDown(self):
        self.progress.stop(); self.silent.__exit__(None, None, None)

    def test_identity_preserves_string_id_and_ignores_broken_legacy_id(self):
        self.assertEqual(sample_identity({'sample_id': '000123'}, 0, 'abc'), ('000123', 'source_sample_id'))
        self.assertEqual(sample_identity({'id': '5.53e17'}, 2, 'abc'), ('csv:abc:row:3', 'csv_sha256_row_v1'))

    def test_all_metrics_roundtrip_without_model_and_csv_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; trainer = fixture_trainer(root, n=5, epochs=1)
            trainer.train_loop()
            for split in ('val', 'test'):
                path = root/f'{split}_predictions.csv'
                result = evaluate_files([path])['sample_pooled']
                stored = json.loads((root/f'{split}_metrics.json').read_text())
                for key, value in result.items(): self.assertEqual(value, stored[key], key)
                with path.open() as stream: rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 5)
                self.assertEqual([r['sample_id'] for r in rows], [f'synthetic-{i}' for i in range(5)])
                self.assertTrue(all(r['checkpoint_id'] == stored['checkpoint_id'] for r in rows))
            # Separate process proves that evaluation does not instantiate the model.
            child = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'evaluate.py'),
                                    str(root/'test_predictions.csv'), '--output', str(root/'recomputed')],
                                   capture_output=True, text=True, timeout=60)
            self.assertEqual(child.returncode, 0, child.stderr)
            path = root/'test_predictions.csv'; path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): evaluate_files([path])

    def test_duplicate_identity_fails_export_and_marks_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; trainer = fixture_trainer(root, n=4, epochs=1)
            trainer.val_loader.dataset[1]['metadata']['sample_id'] = 'synthetic-0'
            with self.assertRaisesRegex(ValueError, 'duplicate'): trainer.train_loop()
            self.assertEqual(json.loads((root/'status.json').read_text())['status'], 'failed')
            self.assertFalse((root/'test_metrics.json').exists())
            self.assertFalse((root/'val_predictions.csv').exists())

    def test_event_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; trainer = fixture_trainer(root, n=4, epochs=1)
            trainer.val_loader.dataset[0]['metadata']['event'] = 'foreign'
            with self.assertRaisesRegex(ValueError, 'event'): trainer.train_loop()
            self.assertEqual(json.loads((root/'status.json').read_text())['status'], 'failed')


if __name__ == '__main__': unittest.main()
