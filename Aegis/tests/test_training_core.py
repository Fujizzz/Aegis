"""Real CPU autograd/optimizer tests; only pretrained encoders and input I/O are replaced."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from communication.comm_module import CommunicationModule
from training.trainer import Trainer
from training.options import grad_accum_steps, resolve_variant


class TinyAgent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 1)

    def forward(self, ids, mask):
        h = torch.tanh(self.encoder(ids.to(self.encoder.weight.dtype)))
        return self.classifier(h), h

    def head_on_embed(self, h):
        return self.classifier(h)


def collate(rows, *args):
    return {key: [row[key] for row in rows] if key == 'metadata' else torch.stack([row[key] for row in rows]) for key in rows[0]}


def fixture_trainer(root, n=9, batch_size=2, accum=2, epochs=2, weights=None, warmup=0, declared_epochs=None,
                    variant='custom', model_options=None, weight_decay=0.):
    """Exercise production initialization, loop, evaluation, checkpoint save/reload."""
    root.mkdir()
    (root/'environment').mkdir()
    dims = ('emotion', 'exaggeration', 'bias', 'commonsense')
    loss_weights = dict(final=1., cons=.3, pred=.2, div=0., gate=.1,
                        **{name: .4 for name in dims})
    if weights is not None:
        loss_weights = weights
    cfg = {
        'runtime': {'save_dir': str(root), 'run_id': root.name, 'seed': 42,
                    'deterministic': False, 'debug_sync': False, 'log_steps': 1000},
        'model': {'base_model': 'local-test-double', 'comm_heads': 2, 'comm_rounds': 2,
                  'quantization': 'none', 'use_lora': False, 'use_fp16': False},
        'data': {'train_file': 'fixture', 'val_file': 'fixture', 'test_file': 'fixture',
                 'label_schema': 'pheme_rumour_v1'},
        'hardware': {'device_map': {**{f'agent_{name}': 'cpu' for name in dims}, 'communication': 'cpu'}},
        'evaluation': {'threshold': .5},
        'training': {'batch_size': batch_size, 'grad_accum_steps': accum,
                     'epochs': epochs if declared_epochs is None else declared_epochs,
                     'warmup_epochs': warmup, 'collab_epochs': epochs - warmup,
                     'learning_rate': .02, 'weight_decay': weight_decay, 'grad_clip': .15,
                     'loss_weights': loss_weights},
    }
    cfg['experiment'] = {'variant': variant, 'event': 'synthetic'}
    data_path = root/'fixture.csv'
    data_path.write_text('synthetic fixture provenance\n')
    cfg['data'].update({f'{split}_file': str(data_path) for split in ('train', 'val', 'test')})
    cfg['model'].update(model_options or {})
    cfg = resolve_variant(cfg)
    (root/'run.json').write_text('{}')
    (root/'config.resolved.json').write_text(json.dumps(cfg))
    generator = torch.Generator().manual_seed(64)
    rows = []
    for i in range(n):
        row = {'input_ids': torch.randn(4, generator=generator), 'attention_mask': torch.ones(4),
               'label': torch.tensor(i % 2),
               'metadata': dict(sample_id=f'synthetic-{i}', event='synthetic', identity_source='synthetic_fixture')}
        for name in dims:
            row[f'{name}_label'] = torch.rand((), generator=generator)
            row[f'{name}_mask'] = torch.tensor(1.)
        rows.append(row)
    tokenizer = type('Tokenizer', (), {'pad_token': 'pad'})()
    with patch('training.trainer.BaseAgent', TinyAgent), \
         patch('training.trainer.AutoTokenizer.from_pretrained', return_value=tokenizer), \
         patch('training.trainer.build_dataset', return_value=rows), \
         patch('training.trainer.collate_fn', collate):
        trainer = Trainer(cfg)
    # Deterministic ordering makes small-batch and merged-batch objectives comparable.
    trainer.train_loader = DataLoader(rows, batch_size=batch_size, collate_fn=collate)
    trainer.val_loader = DataLoader(rows, batch_size=batch_size, collate_fn=collate)
    trainer.test_loader = trainer.val_loader
    return trainer


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class GateActivationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        torch.set_num_threads(1)

    def test_activation_average_all_rounds_and_gradient_flow(self):
        comm = CommunicationModule(4, rounds=3, attn_heads=2)
        activations = []
        hook = comm.gate.register_forward_hook(lambda module, inputs, out: activations.append(out.sigmoid()))
        x = torch.randn(3, 4, 4, requires_grad=True)
        comm(x)
        hook.remove()
        expected = torch.stack(activations).abs().mean()
        torch.testing.assert_close(comm.gate_l1(), expected)
        self.assertTrue(0 < expected.item() < 1)
        comm.gate_l1().backward()
        for grad in (comm.gate.weight.grad, comm.gate.bias.grad, x.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_penalty_is_activation_not_parameter_norm(self):
        comm = CommunicationModule(4, rounds=1, attn_heads=2)
        x = torch.randn(3, 4, 4)
        comm(x)
        before = comm.gate_l1().detach().clone()
        with torch.no_grad():
            for p in comm.ff.parameters():
                p.add_(20.)
        comm(x)
        torch.testing.assert_close(comm.gate_l1(), before)
        comm.gate_l1().backward()
        self.assertTrue(all(p.grad is None for p in comm.ff.parameters()))
        with torch.no_grad():
            comm.gate.weight.zero_()
            comm.gate.bias.fill_(math.log(.25 / .75))
        comm(x)
        self.assertAlmostEqual(comm.gate_l1().item(), .25, places=6)

    def test_zero_coefficient_and_latest_forward(self):
        comm = CommunicationModule(4, rounds=2, attn_heads=2)
        self.assertEqual(comm.gate_l1().item(), 0)
        x = torch.randn(2, 4, 4)
        for _ in range(2):
            comm.zero_grad(set_to_none=True)
            comm(x)
            (comm.gate_l1() * 0).backward()
            self.assertTrue(all(p.grad is None or not p.grad.any() for p in comm.parameters()))
        # A new forward must not reuse a freed graph or evaluation activations.
        with torch.no_grad():
            comm(x)
        comm(x[:1])
        comm.gate_l1().backward()
        self.assertGreater(comm.gate.bias.grad.abs().sum().item(), 0)


class TrainerAccumulationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.progress = patch('training.trainer.tqdm', lambda loader, **kwargs: loader)
        self.progress.start()

    def tearDown(self):
        self.progress.stop()
        self.output.__exit__(None, None, None)

    def test_invalid_values_fail_before_model_loading(self):
        for value in (0, -2, True, False, 2.0, '2', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Trainer({'training': {'grad_accum_steps': value}})
        self.assertEqual(grad_accum_steps({}), 1)

    def test_update_frequency_tail_lr_clipping_and_checkpoints(self):
        for n, batch_size, accum in ((9, 2, 1), (9, 2, 2), (11, 2, 2), (3, 2, 8), (8, 2, 2)):
            with self.subTest(n=n, accum=accum), tempfile.TemporaryDirectory() as tmp:
                trainer = fixture_trainer(Path(tmp)/'run', n, batch_size, accum)
                updates = math.ceil(math.ceil(n / batch_size) / accum)
                total = 2 * updates
                seen_grad_norms = []
                def capture(optimizer, args, kwargs):
                    seen_grad_norms.append(torch.linalg.vector_norm(torch.cat([
                        p.grad.reshape(-1) for p in trainer._trainable_params() if p.grad is not None])).item())
                trainer.optim.register_step_pre_hook(capture)
                with patch('training.trainer.torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
                    trainer.train_loop()
                    self.assertEqual(clip.call_count, total)
                log = records(trainer.optimizer_steps_path)
                self.assertEqual(len(log), total)
                self.assertEqual(trainer.optimizer_steps, total)
                self.assertTrue(all(norm <= .150001 for norm in seen_grad_norms))
                for epoch in (1, 2):
                    rows = [row for row in log if row['epoch'] == epoch]
                    self.assertEqual(sum(row['effective_batch_size'] for row in rows), n)
                    self.assertEqual(sum(row['micro_batches'] for row in rows), math.ceil(n / batch_size))
                    self.assertEqual(rows[-1]['effective_batch_size'], n - (updates - 1) * batch_size * accum)
                warmup = int(total * .1)
                def lr(step):
                    return .02 * (step / max(1, warmup) if step < warmup else
                                  max(0., (total - step) / max(1, total - warmup)))
                for step, row in enumerate(log):
                    self.assertAlmostEqual(row['lr_used'][0], lr(step))
                    self.assertAlmostEqual(row['lr_next'][0], lr(step + 1))
                self.assertTrue(all(p.grad is None for p in trainer._trainable_params()))
                state = torch.load(Path(trainer.save_dir)/'last/training_state.pth', weights_only=False)
                self.assertEqual(state['optimizer_steps'], total)
                self.assertEqual(state['scheduler']['last_epoch'], total)
                self.assertEqual(state['grad_accum_steps'], accum)
                self.assertTrue((Path(trainer.save_dir)/'test_metrics.json').is_file())

    def test_gradients_and_losses_match_merged_batches_including_short_samples(self):
        # Diversity is batch-dependent; positive unit dimension weights are used here.
        for n, accum in ((9, 2), (11, 2), (3, 8)):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                runs, gradients = [], []
                for name, bs, k in (('accum', 2, accum), ('merged', 2 * accum, 1)):
                    trainer = fixture_trainer(Path(tmp)/name, n, bs, k, epochs=1)
                    # FP64 prevents Adam's epsilon from amplifying roundoff in
                    # mathematically zero attention key-bias gradients. Other
                    # integration tests exercise the production FP32 path.
                    for module in (trainer.agent_e, trainer.agent_x, trainer.agent_b, trainer.agent_c,
                                   trainer.comm, trainer.final_head, trainer.team_layer, trainer.pred_heads):
                        module.double()
                    captured = []
                    trainer.optim.register_step_pre_hook(lambda opt, args, kwargs: captured.append([
                        p.grad.detach().clone() if p.grad is not None else None for p in trainer._trainable_params()]))
                    trainer.train_loop()
                    runs.append(trainer)
                    gradients.append(captured)
                for actual_step, expected_step in zip(*gradients):
                    for actual, expected in zip(actual_step, expected_step):
                        if actual is None or expected is None:
                            self.assertIs(actual, expected)
                        else:
                            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-4)
                for actual, expected in zip(runs[0]._trainable_params(), runs[1]._trainable_params()):
                    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
                a = records(runs[0].metrics_path)[0]
                b = records(runs[1].metrics_path)[0]
                for key in a['train_losses']:
                    self.assertAlmostEqual(a['train_losses'][key], b['train_losses'][key], places=5)

    def test_trainer_uses_activation_gate_loss_and_warmup_switch(self):
        for gate_weight in (0., 2.):
            with self.subTest(gate_weight=gate_weight), tempfile.TemporaryDirectory() as tmp:
                trainer = fixture_trainer(Path(tmp)/'run', n=3, accum=2, epochs=2, warmup=1,
                                          weights={'final': 0., 'gate': gate_weight})
                trainer.cfg['training']['grad_clip'] = 100.
                # With zero gate weights the activations are exactly .5 in every round.
                nn.init.zeros_(trainer.comm.gate.weight)
                nn.init.zeros_(trainer.comm.gate.bias)
                biases = []
                trainer.optim.register_step_pre_hook(lambda opt, args, kwargs:
                                                       biases.append(trainer.comm.gate.bias.grad.item()))
                trainer.train_loop()
                log = records(trainer.metrics_path)
                self.assertAlmostEqual(log[0]['gate_activation_mean'], .5)
                self.assertEqual(log[0]['train_losses']['gate'], 0.)
                self.assertAlmostEqual(log[1]['train_losses']['gate'], .5 * gate_weight)
                self.assertAlmostEqual(biases[0], 0.)
                self.assertAlmostEqual(biases[1], .25 * gate_weight, places=6)

    def test_scheduler_uses_executed_phase_epochs(self):
        # Explicit phases define the loop even if the legacy epochs field disagrees.
        with tempfile.TemporaryDirectory() as tmp:
            trainer = fixture_trainer(Path(tmp)/'run', n=8, accum=2, epochs=3, warmup=1, declared_epochs=99)
            trainer.train_loop()
            self.assertEqual(trainer.optimizer_steps, 6)
            self.assertEqual(records(trainer.optimizer_steps_path)[-1]['lr_next'], [0.])


if __name__ == '__main__':
    unittest.main()
