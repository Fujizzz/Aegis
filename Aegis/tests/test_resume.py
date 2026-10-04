"""Offline recovery tests: synthetic CPU state only, no Qwen/API/formal training."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_training_core import TinyAgent, collate, fixture_trainer, records
from training.trainer import Trainer
from utils.checkpoints import COMPONENTS
from utils.resume import (MODULE_ATTRS, capture_rng, load_training_state, prepare_resume_run,
                          restore_rng, verify_checkpoint, verify_resume_inputs)
from utils.run_archive import model_manifest, prepare_run, sha256, write_json
from server_acceptance import state_digest


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch('training.trainer.tqdm', lambda data, **kw: data))
        template = fixture_trainer(self.root / 'template', n=5, epochs=3, warmup=1,
                                   model_options={'attn_dropout': .2, 'ff_dropout': .25})
        self.rows = template.train_loader.dataset
        cfg = copy.deepcopy(template.cfg)
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'weights.bin').write_bytes(b'offline identity fixture, never loaded')
        write_json(self.root / 'model.json', model_manifest(model))
        cfg['model'].update(base_model=str(model), manifest=str(self.root / 'model.json'))
        cfg['experiment'].update(dataset='fixture', method_id='aegis_neural_v1')
        cfg['runtime'].update(runs_root=str(self.root / 'runs'), select_metric='binary_f1', save_each_epoch=True)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'fixture.py').write_text('# offline fixture\n')
        # Include an original freeze layout distinct from the archive's train.csv names.
        freeze = self.root / 'template' / 'freeze.json'
        write_json(freeze, {'files': {'fixture.csv': sha256(self.root / 'template' / 'fixture.csv')}})
        cfg['data']['freeze_manifest'] = str(freeze)
        self.cfg = cfg
        self.stack.enter_context(patch('training.trainer.BaseAgent', TinyAgent))
        self.stack.enter_context(patch('training.trainer.AutoTokenizer.from_pretrained',
                                      return_value=type('Tokenizer', (), {'pad_token': 'pad'})()))
        self.stack.enter_context(patch('training.trainer.build_dataset', side_effect=lambda *a, **k: copy.deepcopy(self.rows)))
        self.stack.enter_context(patch('training.trainer.collate_fn', collate))

    def fresh(self, name):
        cfg = copy.deepcopy(self.cfg)
        cfg['runtime'].update(save_dir=str(self.root / name), run_id=name)
        cfg, _ = prepare_run(cfg, self.root / 'cfg.json', self.source)
        return Trainer(cfg)

    def state_fixture(self, name='original', epoch=1):
        """Populate real AdamW moments without model forward/backward or a training loop."""
        trainer = self.fresh(name)
        for _ in range(epoch * trainer.updates_per_epoch):
            for p in trainer._trainable_params():
                p.grad = torch.randn_like(p) * .01
            trainer.optim.step()
            trainer.sched.step()
            trainer.optimizer_steps += 1
        trainer.optim.zero_grad(set_to_none=True)
        trainer.current_epoch = trainer.best_epoch = epoch
        trainer.best_score = .5
        trainer.best_record = dict(epoch=epoch, metric='binary_f1', score=.5, split='val', run_id=name,
                                   validation_metrics={'binary_f1': .5})
        trainer._save_all('best')
        trainer.best_record['checkpoint_manifest_sha256'] = sha256(Path(trainer.save_dir) / 'best/checkpoint_manifest.json')
        trainer._save_all('last')
        return trainer

    def resume_config(self, trainer, name):
        cfg = copy.deepcopy(trainer.cfg)
        cfg['runtime'].update(save_dir=str(self.root / name), run_id=name)
        cfg['model']['manifest'] = str(Path(trainer.save_dir) / 'model_manifest.json')
        cfg['data']['freeze_manifest'] = str(Path(trainer.save_dir) / 'data/freeze_manifest.json')
        return cfg

    def prepare(self, trainer, name='resumed', checkpoint='last', cfg=None):
        return prepare_resume_run(cfg or self.resume_config(trainer, name), self.root / 'cfg.json',
                                  Path(trainer.save_dir) / checkpoint, source_root=self.source)

    def draws(self):
        return random.random(), float(np.random.rand()), torch.rand(5).tolist()

    def hashes(self, directory):
        return {p.relative_to(directory).as_posix(): sha256(p) for p in Path(directory).rglob('*') if p.is_file()}

    def test_full_state_restore_and_source_immutable(self):
        original = self.state_fixture()
        source_hashes = self.hashes(original.save_dir)
        expected = self.draws()
        cfg, run = self.prepare(original)
        random.seed(901); np.random.seed(902); torch.manual_seed(903)
        resumed = Trainer(cfg)
        self.assertEqual(self.draws(), expected)
        for attr in MODULE_ATTRS:
            self.assertEqual(state_digest(getattr(original, attr).state_dict()),
                             state_digest(getattr(resumed, attr).state_dict()))
        self.assertEqual(state_digest(original.optim.state_dict()), state_digest(resumed.optim.state_dict()))
        self.assertEqual(original.sched.state_dict(), resumed.sched.state_dict())
        self.assertEqual((resumed.current_epoch, resumed.start_epoch, resumed.best_epoch), (1, 2, 1))
        self.assertEqual(resumed.optimizer_steps, original.optimizer_steps)
        self.assertEqual(resumed.best_score, .5)
        self.assertEqual(json.loads((run / 'resume.json').read_text())['source_run_id'], 'original')
        self.assertFalse((run / 'metrics.jsonl').exists())
        self.assertEqual(source_hashes, self.hashes(original.save_dir))
        resumed._load_all('best')  # inherited best passes normal prediction provenance checks
        with self.assertRaises(FileExistsError): self.prepare(original)
        with self.assertRaisesRegex(ValueError, 'already contains'): Trainer(cfg)

    def test_interrupted_shuffle_dropout_matches_uninterrupted_cpu_fixture(self):
        baseline = self.fresh('baseline')
        baseline.train_loop()
        expected_draws = self.draws()
        interrupted = self.fresh('interrupted')
        save = interrupted._save_all
        def stop_after_epoch(subdir=''):
            save(subdir)
            if subdir == 'last' and interrupted.current_epoch == 1:
                raise KeyboardInterrupt('offline simulated interruption')
        with patch.object(interrupted, '_save_all', side_effect=stop_after_epoch):
            with self.assertRaises(KeyboardInterrupt): interrupted.train_loop()
        before = self.hashes(interrupted.save_dir)
        cfg, run = self.prepare(interrupted)
        resumed = Trainer(cfg)
        resumed.train_loop()
        self.assertEqual(self.draws(), expected_draws)
        for name in (*COMPONENTS, 'training_state'):
            a = torch.load(Path(baseline.save_dir) / 'last' / (name + '.pth'), weights_only=True)
            b = torch.load(run / 'last' / (name + '.pth'), weights_only=True)
            if name == 'training_state':
                a.pop('run_id'); b.pop('run_id')
            self.assertEqual(state_digest(a), state_digest(b), name)
        self.assertEqual(records(resumed.optimizer_steps_path), records(baseline.optimizer_steps_path)[2:])
        self.assertEqual([r['epoch'] for r in records(resumed.metrics_path)], [2, 3])
        self.assertEqual(resumed.best_epoch, baseline.best_epoch)
        self.assertEqual(before, self.hashes(interrupted.save_dir))
        self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'completed')

    def test_completed_checkpoint_runs_only_final_evaluation_and_preserves_earliest_best(self):
        original = self.fresh('finished')
        # Fixed model guarantees tied selection scores; earliest epoch must survive.
        for group in original.optim.param_groups: group['lr'] = 0.
        original.sched.base_lrs = [0.]
        original.train_loop()
        self.assertEqual(original.best_epoch, 1)
        cfg, run = self.prepare(original)
        resumed = Trainer(cfg)
        with patch.object(resumed.optim, 'step', side_effect=AssertionError('unexpected update')):
            resumed.train_loop()
        self.assertEqual((resumed.start_epoch, resumed.best_epoch), (4, 1))
        self.assertFalse((run / 'optimizer_steps.jsonl').exists())
        self.assertTrue((run / 'test_predictions.csv').is_file())
        self.assertEqual(json.loads((run / 'best_selection.json').read_text())['epoch'], 1)

    def test_config_and_input_mismatches_rejected(self):
        original = self.state_fixture()
        changes = [('runtime', 'seed', 43), ('training', 'batch_size', 4),
                   ('training', 'grad_accum_steps', 1), ('training', 'learning_rate', .01),
                   ('training', 'collab_epochs', 7), ('model', 'comm_rounds', 0),
                   ('data', 'max_length', 31), ('runtime', 'select_metric', 'accuracy'),
                   ('data', 'label_schema', 'archive_fake_v1')]
        for i, (section, key, value) in enumerate(changes):
            cfg = self.resume_config(original, f'bad{i}')
            cfg[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'mismatch'):
                self.prepare(original, cfg=cfg)
        cfg = self.resume_config(original, 'changed_data')
        data = self.root / 'different.csv'; data.write_text('changed data\n')
        cfg['data']['train_file'] = str(data)
        cfg['data'].pop('freeze_manifest')
        with self.assertRaisesRegex(ValueError, 'mismatch'): self.prepare(original, cfg=cfg)
        cfg = self.resume_config(original, 'changed_model')
        (self.root / 'model/weights.bin').write_bytes(b'changed')
        write_json(self.root / 'model.json', model_manifest(self.root / 'model'))
        cfg['model']['manifest'] = str(self.root / 'model.json')
        with self.assertRaisesRegex(ValueError, 'mismatch'): self.prepare(original, cfg=cfg)

    def test_tampered_or_missing_checkpoint_files_and_legacy_refused_before_load(self):
        original = self.state_fixture()
        checkpoint = Path(original.save_dir) / 'last'
        for name in ('training_state.pth', 'agent_emotion.pth', 'config.resolved.json',
                     'resume_contract.json', 'selection.json'):
            path = checkpoint / name
            content = path.read_bytes()
            path.write_bytes(content + b'tamper')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'hash mismatch'), \
                 patch('torch.load', side_effect=AssertionError('loaded before hash check')):
                self.prepare(original)
            path.write_bytes(content)
        path = checkpoint / 'team_layer.pth'; content = path.read_bytes(); path.unlink()
        with self.assertRaises(FileNotFoundError): self.prepare(original)
        path.write_bytes(content)
        path = checkpoint / 'checkpoint_manifest.json'
        doc = json.loads(path.read_text()); doc['format_version'] = 1; write_json(path, doc)
        with self.assertRaisesRegex(ValueError, 'legacy resume refused'): self.prepare(original)

    def test_prepare_only_cli_imports_no_torch_and_revalidates_local_snapshot(self):
        original = self.state_fixture()
        run = self.root / 'cli'
        script = Path(__file__).resolve().parents[1] / 'train.py'
        code = """import builtins, runpy, sys
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'transformers', 'peft'):
        raise AssertionError('prepare-only imported model dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
sys.argv = ['train.py', '--resume', sys.argv[1], '--save_dir', sys.argv[2], '--run_id', 'cli', '--prepare_only']
runpy.run_path('train.py', run_name='__main__')
"""
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code,
                                 str(Path(original.save_dir) / 'last'), str(run)],
                                cwd=script.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = json.loads((run / 'config.resolved.json').read_text())
        verify_resume_inputs(cfg)
        self.assertFalse((run / 'best').exists())
        self.assertFalse((run / 'console.log').exists())
        Path(cfg['data']['train_file']).write_text('changed after prepare')
        with patch('training.trainer.BaseAgent', side_effect=AssertionError('model initialized')):
            with self.assertRaisesRegex(ValueError, 'data hash mismatch'): Trainer(cfg)

    def test_malformed_state_epoch_scheduler_rng_and_parameter_order_refused(self):
        original = self.state_fixture()
        directory = Path(original.save_dir) / 'last'
        state = torch.load(directory / 'training_state.pth', weights_only=True)
        changes = [('epoch', 4), ('optimizer_steps', 999), ('parameter_names', []),
                   ('cuda_rng', [torch.zeros(8, dtype=torch.uint8)]), ('torch_version', 'different')]
        for key, value in changes:
            altered = copy.deepcopy(state); altered[key] = value
            torch.save(altered, directory / 'training_state.pth')
            manifest = json.loads((directory / 'checkpoint_manifest.json').read_text())
            manifest['sha256']['training_state.pth'] = sha256(directory / 'training_state.pth')
            write_json(directory / 'checkpoint_manifest.json', manifest)
            with self.subTest(key=key), self.assertRaises(ValueError): load_training_state(directory, original)

    def test_cuda_rng_capture_restore_mock_does_not_claim_gpu_execution(self):
        states = [torch.arange(8, dtype=torch.uint8), torch.arange(9, dtype=torch.uint8)]
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_rng_state_all', return_value=states), \
             patch('torch.cuda.set_rng_state_all') as setter:
            saved = capture_rng()
            restore_rng(saved)
            self.assertIs(setter.call_args.args[0], states)

    def test_failure_while_staging_does_not_corrupt_previous_checkpoint(self):
        trainer = self.state_fixture()
        before = self.hashes(Path(trainer.save_dir) / 'last')
        with patch('torch.save', side_effect=OSError('injected write failure')):
            with self.assertRaises(OSError): trainer._save_all('last')
        self.assertEqual(before, self.hashes(Path(trainer.save_dir) / 'last'))
        verify_checkpoint(Path(trainer.save_dir) / 'last')
        self.assertFalse(list(Path(trainer.save_dir).glob('.checkpoint-*')))

    def test_old_epoch_retains_its_selected_best_after_later_best_changes(self):
        trainer = self.state_fixture()
        old_best = trainer.best_record['checkpoint_manifest_sha256']
        # Epoch 2 has no improvement; epoch 3 replaces the run's visible best.
        for ep in (2, 3):
            for _ in range(trainer.updates_per_epoch):
                for p in trainer._trainable_params(): p.grad = torch.full_like(p, .01)
                trainer.optim.step(); trainer.sched.step(); trainer.optimizer_steps += 1
            trainer.optim.zero_grad(set_to_none=True)
            trainer.current_epoch = ep
            if ep == 3:
                trainer.best_epoch, trainer.best_score = 3, .75
                trainer.best_record = dict(epoch=3, metric='binary_f1', score=.75, split='val',
                    run_id=trainer.cfg['runtime']['run_id'], validation_metrics={'binary_f1': .75})
                trainer._save_all('best')
                trainer.best_record['checkpoint_manifest_sha256'] = sha256(Path(trainer.save_dir) / 'best/checkpoint_manifest.json')
            trainer._save_all('epoch_' + str(ep))
        cfg, run = self.prepare(trainer, checkpoint='epoch_2')
        resumed = Trainer(cfg)
        self.assertEqual((resumed.current_epoch, resumed.best_epoch, resumed.best_score), (2, 1, .5))
        self.assertEqual(json.loads((run / 'resume.json').read_text())['best_manifest_sha256'], old_best)
        path = Path(trainer.save_dir) / 'best_history/epoch_1/final_classifier.pth'
        path.write_bytes(b'changed old best')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.prepare(trainer, name='broken_old_best', checkpoint='epoch_2')

    def test_base_model_and_recovery_snapshot_rechecked_before_initialization(self):
        original = self.state_fixture()
        cfg, run = self.prepare(original)
        path = run / 'recovery/checkpoint/training_state.pth'
        content = path.read_bytes(); path.write_bytes(b'changed')
        with patch('training.trainer.BaseAgent', side_effect=AssertionError('model initialized')):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): Trainer(cfg)
        path.write_bytes(content)
        (self.root / 'model/weights.bin').write_bytes(b'changed external model after preparation')
        with patch('training.trainer.BaseAgent', side_effect=AssertionError('model initialized')):
            with self.assertRaisesRegex(ValueError, 'base model content hash mismatch'): Trainer(cfg)

    def test_publish_interruption_keeps_previous_epoch_recoverable(self):
        trainer = self.state_fixture()
        original_rename = Path.rename
        def fail_publish(path, target):
            if path.name.startswith('.checkpoint-'):
                raise OSError('injected interruption between directory renames')
            return original_rename(path, target)
        with patch.object(Path, 'rename', fail_publish):
            with self.assertRaises(OSError): trainer._save_all('last')
        self.assertFalse((Path(trainer.save_dir) / 'last').exists())
        verify_checkpoint(Path(trainer.save_dir) / 'last.previous')
        cfg, _ = self.prepare(trainer, checkpoint='last.previous')
        self.assertEqual(Trainer(cfg).start_epoch, 2)

    def test_export_after_inference_reload_cannot_mix_model_and_optimizer_epochs(self):
        trainer = self.state_fixture()
        trainer._load_all('best')
        trainer._save_all('roundtrip')
        with self.assertRaisesRegex(ValueError, 'verified recovery metadata'):
            verify_checkpoint(Path(trainer.save_dir) / 'roundtrip')


if __name__ == '__main__':
    unittest.main()
