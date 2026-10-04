"""T11/T12: real autograd, data masking, ablation behavior and checkpoint parity."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from communication.comm_module import CommunicationModule
from inference.predictor import Predictor
from training.losses import bce_logits_weighted
from training.options import resolve_variant, VARIANTS, DIMENSIONS
from utils.data_loader import NewsDataset, collate_fn
from utils.label_schema import get_label_schema
from utils.run_archive import resolve_config
from test_training_core import TinyAgent, fixture_trainer, records


class Tokenizer:
    pad_token = 'pad'

    def __call__(self, texts, **kwargs):
        n = 1 if isinstance(texts, str) else len(texts)
        return {'input_ids': torch.ones(n, 4), 'attention_mask': torch.ones(n, 4)}


class SupervisionTests(unittest.TestCase):
    def test_all_masked_is_graph_zero_even_with_nan_inputs(self):
        logits = torch.tensor([[float('nan')], [2.]], requires_grad=True)
        loss = bce_logits_weighted(logits, torch.full_like(logits, float('nan')), torch.zeros(2, 1))
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_partial_mask_and_nonunit_weight_match_independent_reference(self):
        logits = torch.tensor([[-1.], [.4], [2.], [.9]], requires_grad=True)
        targets = torch.tensor([[.2], [float('nan')], [.8], [1.]])
        weights = torch.tensor([[2.], [0.], [3.], [0.]])
        loss = bce_logits_weighted(logits, targets, weights)
        selected = logits.detach()[[0, 2]].clone().requires_grad_()
        expected = (F.binary_cross_entropy_with_logits(selected, targets[[0, 2]], reduction='none') * weights[[0, 2]]).sum() / 5
        torch.testing.assert_close(loss, expected)
        loss.backward(); expected.backward()
        torch.testing.assert_close(logits.grad[[0, 2]], selected.grad)
        torch.testing.assert_close(logits.grad[[1, 3]], torch.zeros(2, 1))

    def test_broadcast_denominator_counts_every_active_element(self):
        logits = torch.randn(3, 4, requires_grad=True)
        targets = torch.rand(3, 1)
        expected = F.binary_cross_entropy_with_logits(logits, targets.expand_as(logits))
        for weights in (None, torch.tensor(2.), torch.ones(3, 1)):
            torch.testing.assert_close(bce_logits_weighted(logits, targets, weights), expected)

    def test_invalid_supervision_fails_instead_of_silent_fallback(self):
        logits = torch.zeros(1, requires_grad=True)
        for value in (-1., float('inf'), float('nan')):
            with self.subTest(weight=value), self.assertRaises(ValueError):
                bce_logits_weighted(logits, torch.ones(1), torch.tensor([value]))
        for value in (-.1, 1.1, float('nan'), float('inf')):
            with self.subTest(target=value), self.assertRaises(ValueError):
                bce_logits_weighted(logits, torch.tensor([value]))

    def test_csv_to_collation_missing_label_weight_and_explicit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'input.csv'
            path.write_text('title,text,subject,date,label,emotion_label,emotion_weight\n'
                            ',one,,,0,0.8,\n,two,,,1,,5\n,three,,,0,0.2,0\n,four,,,1,0.6,2\n')
            dataset = NewsDataset(str(path), Tokenizer(), label_schema=get_label_schema('pheme_rumour_v1'))
            batch = collate_fn([dataset[i] for i in range(4)], Tokenizer(), 4)
            self.assertEqual(batch['emotion_mask'].tolist(), [1., 0., 1., 1.])
            self.assertEqual(batch['emotion_weight'].tolist(), [1., 5., 0., 2.])
            self.assertEqual(batch['emotion_wmask'].tolist(), [0., 1., 1., 1.])
            logits = torch.zeros(4, requires_grad=True)
            bce_logits_weighted(logits, batch['emotion_label'], batch['emotion_mask'] * batch['emotion_weight']).backward()
            torch.testing.assert_close(logits.grad, torch.tensor([-.3/3, 0., 0., -.2/3]))

    def test_collation_nan_absent_and_bad_values(self):
        base = dict(input_text='text', label=1)
        batch = collate_fn([dict(base, emotion_label=float('nan')), base], Tokenizer(), 4)
        self.assertNotIn('emotion_label', batch)
        batch = collate_fn([dict(base, emotion_label=.5), dict(base, emotion_label=.8)], Tokenizer(), 4)
        self.assertNotIn('emotion_weight', batch)  # trainer defaults to the label mask
        for key, value in (('emotion_label', 1.2), ('emotion_label', float('inf')),
                           ('emotion_weight', -1), ('emotion_weight', float('inf'))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                collate_fn([dict(base, **{key: value})], Tokenizer(), 4)


class CommunicationAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)

    def test_zero_rounds_is_identity_mean_with_no_parameter_gradients(self):
        for fusion in ('mean', 'attn'):
            comm = CommunicationModule(4, rounds=0, attn_heads=2, fuse_mode=fusion)
            x = torch.randn(3, 4, 4, requires_grad=True)
            states, fused = comm(x)
            torch.testing.assert_close(states, x, rtol=0, atol=0)
            torch.testing.assert_close(fused, x.mean(1), rtol=0, atol=0)
            (fused.sum() + comm.gate_l1()).backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, .25))
            self.assertTrue(all(p.grad is None for p in comm.parameters()))

    def test_no_gate_matches_unit_gate_and_does_not_train_gate(self):
        reference = CommunicationModule(4, rounds=2, attn_heads=2)
        ungated = CommunicationModule(4, rounds=2, attn_heads=2, use_gate=False)
        ungated.load_state_dict(reference.state_dict())
        with torch.no_grad():
            reference.gate.weight.zero_(); reference.gate.bias.fill_(100.)
        x = torch.randn(3, 4, 4, requires_grad=True)
        with patch.object(ungated.gate, 'forward', side_effect=AssertionError('gate must be bypassed')):
            actual, _ = ungated(x)
        expected, _ = reference(x)
        torch.testing.assert_close(actual, expected)
        (actual[..., 0].sum() + ungated.gate_l1()).backward()
        self.assertIsNone(ungated.gate.weight.grad)
        self.assertGreater(ungated.v_proj.weight.grad.abs().sum().item(), 0.)
        self.assertEqual(ungated.gate_l1().item(), 0.)

    def test_stale_gate_graph_is_cleared_and_invalid_rounds_rejected(self):
        comm = CommunicationModule(4, attn_heads=2)
        x = torch.randn(2, 4, 4, requires_grad=True)
        comm(x); self.assertIsNotNone(comm._latest_gates)
        comm.rounds = 0
        comm(x); self.assertIsNone(comm._latest_gates)
        self.assertEqual(comm.gate_l1().item(), 0.)
        for rounds in (-1, True, 1.5, '0'):
            with self.assertRaises(ValueError): CommunicationModule(4, rounds=rounds)


def run_variant(root, variant):
    """Retain synthetic artifacts and return machine-readable acceptance evidence."""
    weights = dict(final=1., cons=.3, pred=.2, div=.1, gate=.1,
                   **{d: .4 for d in DIMENSIONS})
    trainer = fixture_trainer(root, n=5, epochs=1, variant=variant, weights=weights, weight_decay=.1)
    # Deliberately retain dimension labels in the test batch, even for no_dim.
    # This proves the trainer itself ignores them, independent of data loading.
    if variant == 'no_dim':
        for row in trainer.train_loader.dataset:
            for dim in DIMENSIONS:
                row[f'{dim}_label'] = torch.tensor(float('nan'))
    modules = {'comm': trainer.comm, 'gate': trainer.comm.gate, 'team': trainer.team_layer,
               'pred': trainer.pred_heads, 'final': trainer.final_head}
    initial = {name: [p.detach().clone() for p in module.parameters()] for name, module in modules.items()}
    grad_seen = {name: False for name in modules}
    def capture(opt, args, kwargs):
        for name, module in modules.items():
            grad_seen[name] |= any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in module.parameters())
    trainer.optim.register_step_pre_hook(capture)
    with patch('training.trainer.tqdm', lambda loader, **kwargs: loader):
        trainer.train_loop()
    log = records(trainer.metrics_path)[0]
    disabled = {'no_cons': ['cons'], 'no_pred': ['pred'], 'no_div': ['div'],
                'no_gate_loss': ['gate'], 'no_collab': ['cons', 'pred', 'div', 'gate'],
                'no_dim': ['dim'], 'no_comm': ['gate'], 'no_gate': ['gate']}.get(variant, [])
    assert all(log['train_losses'][key] == 0 for key in disabled)
    inactive = {'no_comm': ['comm'], 'no_gate': ['gate'], 'no_cons': ['team'],
                'no_pred': ['pred'], 'no_collab': ['team', 'pred']}.get(variant, [])
    last = {name: torch.load(root/'last'/file, weights_only=True) for name, file in
            [('comm', 'communication_module.pth'), ('team', 'team_layer.pth'), ('pred', 'pred_heads.pth')]}
    for name in inactive:
        assert not grad_seen[name], (variant, name)
        state = last['comm'] if name == 'gate' else last[name]
        keys = list(modules[name].state_dict())
        for key, before in zip(keys, initial[name]):
            torch.testing.assert_close(state['gate.' + key if name == 'gate' else key], before, atol=0, rtol=0)
    assert grad_seen['final']
    if variant in ('no_comm', 'no_gate'):
        assert log['gate_activation_mean'] is None
    paths = {name: str(root/'best'/f'{name}.pth') for name in
             ('agent_emotion', 'agent_exaggeration', 'agent_bias', 'agent_commonsense',
              'communication_module', 'final_classifier')}
    with patch('inference.predictor.BaseAgent', TinyAgent), \
         patch('inference.predictor.AutoTokenizer.from_pretrained', return_value=Tokenizer()):
        predictor = Predictor(trainer.cfg, paths)
    output = predictor.predict('fixture')
    batch = Tokenizer()('fixture')
    with torch.no_grad():
        _, embeds = trainer._forward_agents(batch)
        _, fused = trainer.comm(list(embeds))
        expected = trainer.final_head(fused).sigmoid().item()
    assert abs(output['p_positive'] - expected) < 1e-7
    assert (root/'test_metrics.json').is_file()
    return {'variant': variant, 'losses': log['train_losses'], 'nonzero_gradient_seen': grad_seen,
            'inactive_parameters_unchanged_with_weight_decay': inactive,
            'predictor_matches_best': True, 'optimizer_steps': trainer.optimizer_steps,
            'gate_activation_mean': log['gate_activation_mean']}


class VariantIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_every_named_variant_trains_evaluates_saves_and_predicts(self):
        # This fixture inspects four-agent gates and cross-agent predictors.
        # Single-agent and non-neural implementations have their own integration tests.
        from training.options import FOUR_AGENT_VARIANTS
        for variant in FOUR_AGENT_VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp, \
                 contextlib.redirect_stdout(io.StringIO()):
                run_variant(Path(tmp)/variant, variant)

    def test_resolution_is_immutable_idempotent_and_preserves_other_weights(self):
        for variant in VARIANTS:
            cfg = {'experiment': {'variant': variant}, 'training': {'loss_weights': {'final': 1., 'cons': .7}},
                   'data': {'label_schema': 'pheme_rumour_v1'}}
            original = copy.deepcopy(cfg)
            resolved = resolve_variant(cfg)
            self.assertEqual(cfg, original)
            self.assertEqual(resolve_variant(resolved), resolved)
            self.assertEqual(resolved['training']['loss_weights']['final'], 1.)
            cfg['experiment'].update(dataset='fixture', event='test', method_id='aegis')
            self.assertEqual(resolve_config(cfg, 'config.json')['model'], resolved['model'])
            if variant == 'no_dim':
                self.assertFalse(resolved['data']['use_pseudo_labels'])
                self.assertTrue(all(resolved['training']['loss_weights'][d] == 0 for d in DIMENSIONS))

    def test_custom_combined_switches_and_invalid_options(self):
        cfg = resolve_variant({'model': {'comm_rounds': 0, 'fuse_mode': 'attn'},
                               'data': {'use_pseudo_labels': False}})
        self.assertEqual(cfg['model']['fuse_mode'], 'mean')
        self.assertTrue(all(v == 0 for v in cfg['training']['loss_weights'].values()))
        for cfg in ({'model': {'comm_rounds': -.1}}, {'model': {'use_gate': 'false'}},
                    {'data': {'use_pseudo_labels': 0}}, {'experiment': {'variant': 'no_gtae'}},
                    {'training': {'loss_weights': {'cons': float('nan')}}}):
            with self.assertRaises(ValueError): resolve_variant(cfg)

    def test_archive_records_requested_and_effective_variant(self):
        from test_run_evaluation import RunArchiveTests
        from utils.run_archive import prepare_run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = RunArchiveTests().fixture(root)
            cfg['experiment']['variant'] = 'no_dim'
            cfg['data']['use_pseudo_labels'] = True
            cfg['training'] = {'loss_weights': {'emotion': .6, 'cons': .3}}
            resolved, run = prepare_run(cfg, root/'config.json', root/'source')
            requested = json.loads((run/'config.requested.json').read_text())
            saved = json.loads((run/'config.resolved.json').read_text())
            self.assertTrue(requested['data']['use_pseudo_labels'])
            self.assertFalse(saved['data']['use_pseudo_labels'])
            self.assertEqual(saved, resolved)
            self.assertEqual(saved['training']['loss_weights']['cons'], .3)
            self.assertIn('no_dim', run.parts)

    def test_disabled_branches_are_not_called(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            trainer = fixture_trainer(Path(tmp)/'run', epochs=1, variant='no_collab')
            with patch.object(trainer.team_layer, 'forward', side_effect=AssertionError('cons used')), \
                 patch('training.trainer.diversity_loss', side_effect=AssertionError('div used')):
                trainer.train_loop()

    def test_trainer_dimension_gradient_respects_masks_and_zero_weights(self):
        for case in ('partial', 'all_masked', 'missing'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp, \
                 contextlib.redirect_stdout(io.StringIO()):
                trainer = fixture_trainer(Path(tmp)/'run', n=3, batch_size=3, accum=1,
                                          epochs=1, weights={'final': 1., 'emotion': 1.})
                for i, row in enumerate(trainer.train_loader.dataset):
                    row['emotion_label'] = torch.tensor(.8)
                    row['emotion_mask'] = torch.tensor(float(i > 0 and case == 'partial'))
                    row['emotion_weight'] = torch.tensor(0. if i == 1 else 2.)
                    if case == 'missing':
                        del row['emotion_label']; del row['emotion_mask']
                logits, observed = [], []
                def capture(module, inputs, out):
                    if torch.is_grad_enabled():
                        out.retain_grad(); logits.append(out)
                hook = trainer.agent_e.classifier.register_forward_hook(capture)
                def check(opt, args, kwargs):
                    # One pre-communication head and one post-communication head.
                    actual = logits[1].grad
                    expected = torch.zeros(3, 1)
                    if case == 'partial':
                        expected[2] = logits[1].detach()[2].sigmoid() - .8
                    torch.testing.assert_close(actual, expected)
                    observed.append(True)
                trainer.optim.register_step_pre_hook(check)
                trainer.train_loop()
                hook.remove()
                self.assertEqual(observed, [True])


if __name__ == '__main__':
    unittest.main()
