"""T13/T10/T09: real local Qwen2 + PEFT; no pretrained model downloads."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Qwen2Config, Qwen2Model, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.agent_base import BaseAgent
from training.trainer import Trainer
from training.options import DIMENSIONS, resolve_variant
from inference.predictor import Predictor
from utils.checkpoints import checkpoint_paths, checkpoint_config, validate_paths
from utils.data_loader import NewsDataset, collate_fn
from utils.label_schema import get_label_schema
from utils.text_input import build_news_text, tokenize_texts
from utils.run_archive import write_json


def local_model(root):
    root.mkdir()
    torch.manual_seed(29)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        max_position_embeddings=128, pad_token_id=0)
    Qwen2Model(config).save_pretrained(root)
    vocab = {w: i for i, w in enumerate(['[PAD]', '[UNK]', '[EOS]', '[', ']',
             'TITLE', 'SUBJECT', 'DATE', 'CONTENT', 'news', 'one', 'two', 'three',
             'four', 'report', 'today', 'world', 'body', 'long', '2026', '-', '09', '30'])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]',
                                  unk_token='[UNK]', eos_token='[EOS]')
    tok.save_pretrained(root)
    return tok


def real_run(root, dual_cuda=False, streams=False):
    """Reusable acceptance entry point; --dual-cuda requires two real GPUs."""
    if dual_cuda and torch.cuda.device_count() < 2:
        raise RuntimeError('Dual-CUDA acceptance requires at least two visible CUDA devices')
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    tok = local_model(root/'base_model')
    rows = [dict(title='news '+word, text='body '+word, subject='world', date=None,
                 label=i % 2, **{f'{d}_label': .2 + .15*i for d in DIMENSIONS})
            for i, word in enumerate(('one', 'two', 'three', 'four'))]
    data = root/'synthetic.csv'
    pd.DataFrame(rows).to_csv(data, index=False)
    run = root/'run'; run.mkdir(); (run/'environment').mkdir()
    devices = {f'agent_{d}': ('cuda:0' if i < 2 else 'cuda:1') if dual_cuda else 'cpu'
               for i, d in enumerate(DIMENSIONS)}
    devices['communication'] = 'cuda:1' if dual_cuda else 'cpu'
    cfg = resolve_variant({
        'runtime': dict(save_dir=str(run.resolve()), run_id='synthetic-qwen2-lora', seed=42,
                        deterministic=False, debug_sync=False, log_steps=1000,
                        sequential_agents=not streams, use_cuda_streams=streams),
        'model': dict(base_model=str((root/'base_model').resolve()), use_lora=True,
                      lora_r=2, lora_alpha=4, lora_dropout=0., target_modules=['q_proj', 'v_proj'],
                      quantization='none', use_fp16=False, gradient_checkpointing=True,
                      comm_heads=2, comm_rounds=2),
        'data': dict(train_file=str(data), val_file=str(data), test_file=str(data),
                     label_schema='pheme_rumour_v1', max_length=24),
        'hardware': dict(device_map=devices), 'experiment': dict(variant='full', event='synthetic'),
        'evaluation': dict(threshold=.5),
        'training': dict(batch_size=2, grad_accum_steps=1, epochs=2, warmup_epochs=0,
                         collab_epochs=2, learning_rate=.01, weight_decay=.01,
                         loss_weights=dict(final=1., cons=.3, pred=.2, div=.01, gate=.1,
                                           **{d: .4 for d in DIMENSIONS}))})
    write_json(run/'config.resolved.json', cfg)
    write_json(run/'run.json', {'provenance': 'random tiny Qwen2, synthetic news; implementation validation only'})
    trainer = Trainer(cfg)
    modules = dict(zip(DIMENSIONS, (trainer.agent_e, trainer.agent_x, trainer.agent_b, trainer.agent_c)))
    modules.update(comm=trainer.comm, final=trainer.final_head, team=trainer.team_layer, pred=trainer.pred_heads)
    snapshots = {k: {n: p.detach().cpu().clone() for n, p in m.named_parameters()} for k, m in modules.items()}
    pointers = [{(str(p.device), p.data_ptr()) for p in modules[d].parameters()} for d in DIMENSIONS]
    assert all(not a & b for i, a in enumerate(pointers) for b in pointers[i+1:])
    evidence = {'provenance': 'random tiny Qwen2 + real PEFT + production Trainer, no mocks',
                'torch': torch.__version__, 'devices': devices, 'streams': streams, 'gradients': [], 'parameters': {}}
    for key, module in modules.items():
        evidence['parameters'][key] = {'total': sum(p.numel() for p in module.parameters()),
                                      'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad),
                                      'device_dtype': sorted({f'{p.device}/{p.dtype}' for p in module.parameters()})}
        if key in DIMENSIONS:
            assert all(p.requires_grad == ('lora_' in n or n.startswith('classifier.'))
                       for n, p in module.named_parameters())
    def gradients(opt, args, kwargs):
        totals = {}
        for key, module in modules.items():
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    assert p.grad is None
                    continue
                assert p.grad is not None and torch.isfinite(p.grad).all(), (key, name)
                group = 'lora_A' if 'lora_A' in name else 'lora_B' if 'lora_B' in name else 'head'
                tag = key + '/' + group
                totals[tag] = totals.get(tag, 0.) + p.grad.abs().sum().item()
        # LoRA B starts at zero, so A may have zero gradient on update one only.
        for key, total in totals.items():
            if len(evidence['gradients']) > 0 or not key.endswith('/lora_A'):
                assert total > 0, key
        evidence['gradients'].append(totals)
    trainer.optim.register_step_pre_hook(gradients)
    # Inspect last training state before best is reloaded by train_loop.
    saved = trainer._save_all
    def save_with_update_check(subdir=''):
        if subdir == 'last':
            for key, module in modules.items():
                changed = set()
                for name, p in module.named_parameters():
                    same = torch.equal(p.detach().cpu(), snapshots[key][name])
                    if not p.requires_grad:
                        assert same, (key, name)
                    elif not same:
                        changed.add('lora_A' if 'lora_A' in name else 'lora_B' if 'lora_B' in name else 'head')
                assert changed == ({'lora_A', 'lora_B', 'head'} if key in DIMENSIONS else {'head'}), (key, changed)
                evidence['parameters'][key]['updated_groups'] = sorted(changed)
        return saved(subdir)
    trainer._save_all = save_with_update_check
    trainer.train_loop()
    evidence['parity'] = {}
    for boundary in ('best', 'last'):
        errors = []
        trainer._load_all(boundary)
        for module in modules.values(): module.eval()
        predictor = Predictor(cfg, checkpoint_paths(run/boundary))
        for i, row in enumerate(rows):
            batch = collate_fn([trainer.val_loader.dataset[i]], tok, 24)
            expected_input = tokenize_texts(tok, [build_news_text(row)], 24)
            for key in ('input_ids', 'attention_mask'):
                assert torch.equal(batch[key], expected_input[key])
            with torch.no_grad():
                _, embeddings = trainer._forward_agents(batch)
                _, fused = trainer.comm(list(embeddings))
                expected = trainer.final_head(fused).sigmoid().item()
            actual = predictor.predict(row)['p_positive']
            assert abs(actual - expected) < 1e-6, (actual, expected)
            errors.append(abs(actual-expected))
        evidence['parity'][boundary] = {'max_tolerance': 1e-6, 'max_error': max(errors), 'samples': len(rows)}
    # Save -> mutate all trainable parameters -> reload: match probability before save.
    trainer._save_all('roundtrip')
    reference = predictor.predict(rows[0])['p_positive']
    with torch.no_grad():
        for module in modules.values():
            for p in module.parameters():
                if p.requires_grad: p.add_(.5)
    trainer._load_all('roundtrip')
    with torch.no_grad():
        batch = collate_fn([trainer.val_loader.dataset[0]], tok, 24)
        _, embeddings = trainer._forward_agents(batch)
        _, fused = trainer.comm(list(embeddings))
        restored = trainer.final_head(fused).sigmoid().item()
    assert abs(reference-restored) < 1e-6
    evidence['roundtrip_error'] = abs(reference-restored)
    write_json(root/'input.json', rows[0])
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve().parents[1]/'infer.py'),
               '--checkpoint_dir', str(run/'last'), '--input_json', str(root/'input.json'),
               '--output', str(root/'prediction.json')]
    child = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=120)
    (root/'cli.stdout.txt').write_text(child.stdout, encoding='utf-8')
    (root/'cli.stderr.txt').write_text(child.stderr, encoding='utf-8')
    assert child.returncode == 0, child.stderr
    assert abs(json.loads((root/'prediction.json').read_text(encoding='utf-8'))['p_positive']-reference) < 1e-6
    evidence['cli_exit_code'] = child.returncode
    write_json(root/'evidence.json', evidence)
    return evidence


class InputTests(unittest.TestCase):
    def test_null_priority_body_and_no_label_leakage(self):
        self.assertEqual(build_news_text({'title': pd.NA, 'subject': float('nan'), 'date': None,
                                         'text': '  body  ', 'content': 'ignored', 'label': 1}), '[CONTENT]\nbody')
        self.assertEqual(build_news_text({'text': ' ', 'content': 'body'}), build_news_text('body'))
        self.assertEqual(build_news_text({'text': 'nan'}), '[CONTENT]\nnan')
        for value in ('', {}, {'text': None}, {'label': 1}):
            with self.assertRaises(ValueError): build_news_text(value)

    def test_csv_structured_inference_token_ids_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tok = local_model(root/'model')
            rows = [dict(title='news', text='body '*40, content='ignored', subject=None, date=None, label=1),
                    dict(title=None, text=None, content='body two', subject=None, date=None, label=0),
                    dict(title='NA', text='nan', subject=None, date='0012', label=1)]
            path=root/'data.csv'; pd.DataFrame(rows).to_csv(path, index=False)
            dataset=NewsDataset(path, tok, use_pseudo=False, label_schema=get_label_schema('pheme_rumour_v1'))
            for i, news in enumerate(rows):
                train=collate_fn([dataset[i]], tok, 12)
                inference=tokenize_texts(tok, [build_news_text(news)], 12)
                for key in ('input_ids', 'attention_mask'): self.assertTrue(torch.equal(train[key], inference[key]))
            self.assertEqual(train['input_ids'].shape[0], 1)
            self.assertEqual(len(tokenize_texts(tok, [build_news_text(rows[0])], 12)['input_ids'][0]), 12)

    def test_requested_lora_cannot_silently_fall_back(self):
        with patch('agents.agent_base.LoraConfig', None):
            with self.assertRaisesRegex(RuntimeError, 'LoRA requested'):
                BaseAgent('unused', use_lora=True, lora_config={})

    def test_encode_preserves_gradient_and_non_lora_freezing(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/'model'; tok = local_model(model)
            batch = tokenize_texts(tok, [build_news_text('body one')], 16)
            for ckpt in (False, True):
                agent = BaseAgent(str(model), use_lora=True, lora_config={'r': 2},
                                  quantization='none', use_fp16=False, gradient_checkpointing=ckpt)
                agent.train()
                agent.encode(batch['input_ids'], batch['attention_mask']).square().sum().backward()
                grads = [p.grad for n, p in agent.named_parameters() if 'lora_B' in n]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
            frozen = BaseAgent(str(model), quantization='none', use_fp16=False, gradient_checkpointing=False)
            self.assertTrue(all(not p.requires_grad for p in frozen.base_model.parameters()))
            self.assertEqual({id(p) for p in frozen.trainable_parameters()},
                             {id(p) for p in frozen.classifier.parameters()})

    def test_real_lora_training_reload_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'acceptance'
            real_run(root)
            cfg = checkpoint_config(root/'run/best')
            paths = checkpoint_paths(root/'run/best')
            for section, key, value in [('model', 'comm_rounds', 0), ('data', 'max_length', 1)]:
                bad = copy.deepcopy(cfg); bad[section][key] = value
                with self.assertRaises(ValueError): validate_paths(paths, bad)
            paths['team_layer'] = str(root/'run/last/team_layer.pth')
            with self.assertRaises(ValueError): validate_paths(paths, cfg)
            paths = checkpoint_paths(root/'run/best')
            with Path(paths['agent_emotion']).open('ab') as f: f.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): validate_paths(paths, cfg)


if __name__ == '__main__':
    unittest.main()
