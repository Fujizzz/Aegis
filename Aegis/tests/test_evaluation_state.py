import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from test_training_core import fixture_trainer, collate
from training.trainer import Trainer


class EvaluationStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.output = contextlib.redirect_stdout(io.StringIO()); self.output.__enter__()
        self.progress = patch('training.trainer.tqdm', lambda data, **kw: data); self.progress.start()

    def tearDown(self):
        self.progress.stop(); self.output.__exit__(None, None, None)

    def state(self, trainer):
        return json.loads((Path(trainer.save_dir)/'status.json').read_text())

    def test_validation_failure_marks_run_and_never_selects_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = fixture_trainer(Path(tmp)/'run', n=4, epochs=1)
            with patch.object(trainer, 'evaluate', side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError): trainer.train_loop()
            state = self.state(trainer)
            self.assertEqual((state['status'], state['val']['status']), ('failed', 'failed'))
            self.assertIsNone(state['best'])
            self.assertFalse((Path(trainer.save_dir)/'best').exists())
            self.assertFalse((Path(trainer.save_dir)/'test_metrics.json').exists())

    def test_test_failure_is_not_success_and_keeps_valid_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = fixture_trainer(Path(tmp)/'run', n=4, epochs=1)
            original = trainer.evaluate_loader
            def injected(loader, split='val', **kwargs):
                if split == 'test': raise RuntimeError('injected')
                return original(loader, split, **kwargs)
            with patch.object(trainer, 'evaluate_loader', side_effect=injected):
                with self.assertRaises(RuntimeError): trainer.train_loop()
            state = self.state(trainer)
            self.assertEqual((state['status'], state['test']['status']), ('failed', 'failed'))
            self.assertEqual(state['best']['epoch'], 1)
            self.assertFalse((Path(trainer.save_dir)/'test_metrics.json').exists())

    def test_empty_and_single_class_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = fixture_trainer(Path(tmp)/'run', n=4, epochs=1)
            for row in trainer.val_loader.dataset: row['label'] = torch.tensor(0)
            metrics = trainer.evaluate()
            self.assertIsNone(metrics['roc_auc'])
            self.assertEqual(self.state(trainer)['val']['auc_reason'], 'single_class')
            trainer.val_loader = DataLoader([], collate_fn=collate)
            with self.assertRaises(ValueError): trainer.evaluate()
            self.assertEqual(self.state(trainer)['status'], 'failed')

    def test_undefined_selection_and_foreign_best_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = fixture_trainer(Path(tmp)/'run', n=4, epochs=1)
            trainer.select_metric = 'roc_auc'
            for row in trainer.val_loader.dataset: row['label'] = torch.tensor(0)
            with self.assertRaisesRegex(ValueError, 'undefined'): trainer.train_loop()
            self.assertIsNone(self.state(trainer)['best'])
        with tempfile.TemporaryDirectory() as tmp:
            trainer = fixture_trainer(Path(tmp)/'run', n=4, epochs=1)
            trainer.train_loop()
            path = Path(trainer.save_dir)/'best/checkpoint_manifest.json'
            data = json.loads(path.read_text()); data['run_id'] = 'foreign'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'different run'): trainer._load_all('best')

    def test_missing_test_is_failure_but_explicit_disable_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = fixture_trainer(Path(tmp)/'run', n=4, epochs=1)
            # Reuse only a prepared config, before any training outputs exist.
            trainer.cfg['data'].pop('test_file')
            (Path(trainer.save_dir)/'config.resolved.json').write_text(json.dumps(trainer.cfg))
            with self.assertRaisesRegex(ValueError, 'Test requested'): Trainer(trainer.cfg)
            self.assertEqual(self.state(trainer)['test']['reason'], 'missing_test_file')
        with tempfile.TemporaryDirectory() as tmp:
            trainer = fixture_trainer(Path(tmp)/'run', n=4, epochs=1)
            trainer.run_test_after_train = False
            trainer.run_state['test'] = {'status': 'not_requested'}
            trainer.train_loop()
            self.assertEqual(self.state(trainer)['test']['status'], 'not_requested')
            self.assertEqual(self.state(trainer)['status'], 'completed')

    def test_ties_keep_first_epoch_and_previous_outputs_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = fixture_trainer(Path(tmp)/'run', n=4, epochs=2)
            # Zero learning rate creates real tied predictions, without changing metrics.
            for group in trainer.optim.param_groups: group['lr'] = 0
            trainer.sched.base_lrs = [0 for _ in trainer.sched.base_lrs]
            trainer.train_loop()
            self.assertEqual(self.state(trainer)['best']['epoch'], 1)
            self.assertEqual(self.state(trainer)['test']['status'], 'success')
            with self.assertRaisesRegex(ValueError, 'already contains'): Trainer(trainer.cfg)


if __name__ == '__main__': unittest.main()
