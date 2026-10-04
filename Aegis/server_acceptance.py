"""Future T10/T15 server check. Default only writes a plan; --execute loads/trains.

Runs eight training and four validation rows for two optimizer updates, preserving
the parent identities. These acceptance results are never formal experiment scores.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

from utils import freeze_pheme as f
from utils.pseudo_quality import load_source, write_json
from utils.pseudo_policy import training_release_ready
from utils.experiment_protocol import CROSS_EVENT, scope, split_relative


def balanced_subset(rows, count):
    pools = {label: sorted([r for r in rows if r['label'] == label], key=lambda r:r['sample_id'])
             for label in ('0', '1')}
    if count % 2 or any(len(pool) < count // 2 for pool in pools.values()):
        raise ValueError('Acceptance needs balanced nonempty subsets')
    return pools['0'][:count // 2] + pools['1'][:count // 2]


def acceptance_subset(rows, count, *, cross=False):
    if not cross:return balanced_subset(rows,count)
    # Exercise all four source events and both classes without using held-out data.
    return [r for event in sorted({r['event_name'] for r in rows})
            for r in balanced_subset([r for r in rows if r['event_name']==event],2)]


def state_digest(value):
    """Content digest for nested optimizer/scheduler state, independent of pickle bytes."""
    import torch
    h = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu().contiguous()
            h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                h.update(repr(key).encode()); visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            h.update(str(type(obj)).encode())
            for item in obj:
                visit(item)
        else:
            h.update(repr(obj).encode())
    visit(value)
    return h.hexdigest()


def execute(config_path, output):
    import numpy as np
    import torch
    import yaml
    import bitsandbytes as bnb
    from check_environment import check
    from training.trainer import Trainer
    from utils.run_archive import resolve_config, prepare_run
    from training.metrics import SCORES
    from utils.resume import restore_rng

    environment = check('training')
    if torch.cuda.device_count() < 2:
        raise RuntimeError('T10 requires two visible real GPUs')
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    cfg = resolve_config(raw, config_path)
    if cfg['experiment']['variant'] != 'full' or cfg['model']['quantization'] != '4bit':
        raise ValueError('Acceptance requires full Aegis with explicit 4bit quantization')
    if not cfg['model'].get('gradient_checkpointing') or not cfg['model'].get('use_lora') or not cfg['model'].get('use_fp16'):
        raise ValueError('Acceptance must exercise LoRA, FP16 and gradient checkpointing')
    devices = set(cfg['hardware']['device_map'].values())
    if len(devices) != 2 or any(not str(d).startswith('cuda:') for d in devices):
        raise ValueError('Acceptance config must use exactly two distinct CUDA devices')
    freeze = Path(cfg['data']['freeze_manifest'])
    cross = scope(cfg)==CROSS_EVENT
    if cross:
        from utils.cross_event import verify
        check_data=verify(freeze.parent)
        rows=f.read_csv(freeze.parent/'parent_canonical.csv')
    else:
        rows, check_data = load_source(freeze.parent)
    if not training_release_ready(rows):
        raise ValueError('Publish Teacher supervision labels under an explicit release policy before server acceptance')
    data_protocol = json.loads((freeze.parent / 'protocol.json').read_text(encoding='utf-8'))
    inputs = output / 'acceptance_inputs'
    inputs.mkdir()
    hashes = {}
    for split, count in (('train', 8), ('val', 4)):
        path = Path(cfg['data'][split + '_file'])
        relative = path.relative_to(freeze.parent).as_posix()
        if f.file_hash(path) != json.loads(freeze.read_text(encoding='utf-8'))['files'].get(relative):
            raise ValueError('Configured acceptance source differs from frozen data')
        selected = acceptance_subset(f.read_csv(path), count, cross=cross)
        dest = inputs / (split + '.csv')
        f.write_csv(dest, selected, list(selected[0]))
        hashes[dest.name] = f.file_hash(dest)
        if cross:hashes[split_relative(cfg,split)]=hashes[dest.name]
        cfg['data'][split + '_file'] = str(dest.resolve())
    cfg['data'].pop('test_file', None)
    write_json(inputs / 'freeze_manifest.json', dict(files=hashes,
        version=CROSS_EVENT if cross else 'acceptance_subset',
        parent_sha256=check_data['freeze_sha256'], purpose='acceptance_subset_only_not_formal_experiment'))
    cfg['data']['freeze_manifest'] = str((inputs / 'freeze_manifest.json').resolve())
    cfg['experiment']['dataset'] += '_acceptance_subset'
    cfg['runtime'].update(save_dir=str((output / 'run').resolve()), run_test_after_train=False,
                          save_each_epoch=False)
    cfg['training'].update(batch_size=2, grad_accum_steps=2, warmup_epochs=0, collab_epochs=1, epochs=1)
    cfg, run = prepare_run(cfg, config_path)
    write_json(output / 'environment.json', environment)
    trainer = Trainer(cfg)
    agents = dict(zip(f.DIMS, (trainer.agent_e, trainer.agent_x, trainer.agent_b, trainer.agent_c)))
    modules = dict(agents, communication=trainer.comm, final=trainer.final_head,
                   team=trainer.team_layer, prediction=trainer.pred_heads)
    evidence = dict(status='running', formal_experiment=False, devices=cfg['hardware']['device_map'],
                    source_config_sha256=f.file_hash(config_path), data_freeze_sha256=check_data['freeze_sha256'],
                    pseudo_release_policy=data_protocol.get('pseudo_release_policy'),
                    human_review_completed=data_protocol.get('human_review_completed', False),
                    fp16_requested=cfg['model'].get('use_fp16'), parameter_dtypes={},
                    gradients=[], quantized_layers={}, frozen_parameter_digests={}, trainable_updates={})
    pointers = []
    for dim, agent in agents.items():
        quantized_modules = [m for m in agent.modules() if isinstance(m, bnb.nn.Linear4bit)]
        quantized = len(quantized_modules)
        if not quantized:
            raise RuntimeError('4bit requested but no Linear4bit layer found: ' + dim)
        evidence['quantized_layers'][dim] = quantized
        if any(m.compute_dtype != torch.float16 for m in quantized_modules):
            raise RuntimeError('4bit compute dtype is not the requested FP16')
        evidence['parameter_dtypes'][dim] = sorted({str(p.dtype) for p in agent.parameters()})
        if str(trainer._agent_device(agent)) != cfg['hardware']['device_map']['agent_' + dim]:
            raise RuntimeError('Agent placed on unexpected device')
        if any(p.requires_grad != ('lora_' in name or name.startswith('classifier.'))
               for name, p in agent.named_parameters()):
            raise RuntimeError('Unexpected backbone/adapter freeze state')
        pointers.append({(str(p.device), p.data_ptr()) for p in agent.parameters()})
    if any(a & b for i, a in enumerate(pointers) for b in pointers[i+1:]):
        raise RuntimeError('Agent parameter storage is shared')
    def frozen_digest(module):
        return state_digest({n:p for n,p in module.named_parameters() if not p.requires_grad})
    frozen = {name:frozen_digest(module) for name,module in agents.items()}
    before = {name:{n:p.detach().cpu().clone() for n,p in module.named_parameters() if p.requires_grad}
              for name,module in modules.items()}
    def gradient_hook(opt, args, kwargs):
        totals = {}
        for name, module in modules.items():
            for param_name, param in module.named_parameters():
                if not param.requires_grad:
                    if param.grad is not None:
                        raise RuntimeError('Frozen backbone received gradient')
                    continue
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise RuntimeError('Missing/nonfinite gradient: ' + name + '/' + param_name)
                group = 'lora_A' if 'lora_A' in param_name else 'lora_B' if 'lora_B' in param_name else 'head'
                tag = name + '/' + group
                totals[tag] = totals.get(tag, 0.) + float(param.grad.abs().sum())
        for name, total in totals.items():
            if (evidence['gradients'] or not name.endswith('/lora_A')) and total <= 0:
                raise RuntimeError('Zero gradient group: ' + name)
        evidence['gradients'].append(totals)
    hook = trainer.optim.register_step_pre_hook(gradient_hook)
    trainer.train_loop()
    hook.remove()
    if trainer.optimizer_steps != 2:
        raise RuntimeError('Expected exactly two acceptance optimizer updates')
    trainer._load_all('last')
    for name, module in agents.items():
        digest = frozen_digest(module)
        if digest != frozen[name]:
            raise RuntimeError('Frozen parameter changed: ' + name)
        evidence['frozen_parameter_digests'][name] = digest
    for name, module in modules.items():
        changed = set()
        for param_name, param in module.named_parameters():
            if param.requires_grad and not torch.equal(param.detach().cpu(), before[name][param_name]):
                changed.add('lora_A' if 'lora_A' in param_name else 'lora_B' if 'lora_B' in param_name else 'head')
        expected = {'lora_A', 'lora_B', 'head'} if name in agents else {'head'}
        if changed != expected:
            raise RuntimeError('Expected parameter group did not update: ' + name)
        evidence['trainable_updates'][name] = sorted(changed)
    del before
    checkpoint = run / 'last'
    manifest = json.loads((checkpoint / 'checkpoint_manifest.json').read_text(encoding='utf-8'))
    if f.file_hash(checkpoint / 'training_state.pth') != manifest['sha256']['training_state.pth']:
        raise ValueError('Training state content hash mismatch')
    state = torch.load(checkpoint / 'training_state.pth', map_location='cpu', weights_only=True)
    def restore_state():
        trainer.optim.load_state_dict(state['optimizer'])
        trainer.sched.load_state_dict(state['scheduler'])
        restore_rng(state)
        trainer.optimizer_steps = state['optimizer_steps']
        trainer.current_epoch = state['epoch']
    def draws():
        return [random.random(), float(np.random.rand()), float(torch.rand(())),
                *[float(torch.rand((), device=d)) for d in sorted(devices)]]
    restore_state()
    expected_draws = draws()
    optimizer_digest = state_digest(trainer.optim.state_dict())
    scheduler_digest = state_digest(trainer.sched.state_dict())
    for group in trainer.optim.param_groups:
        group['lr'] = 999.
    trainer.optim.state.clear()
    restore_state()
    if (draws() != expected_draws or state_digest(trainer.optim.state_dict()) != optimizer_digest
            or state_digest(trainer.sched.state_dict()) != scheduler_digest):
        raise RuntimeError('Optimizer/scheduler/RNG restore mismatch')
    evidence['state_restore'] = dict(optimizer=True, scheduler=True, rng=True,
        boundary='same-process epoch-state roundtrip; separate-process --resume still requires server acceptance')
    trainer._load_all('best')
    batch = next(iter(trainer.val_loader))
    for module in modules.values():
        module.eval()
    with torch.no_grad():
        _, embeddings = trainer._forward_agents(batch)
        _, fused = trainer.comm(list(embeddings))
        reference = trainer.final_head(fused).sigmoid()[0].item()
        for module in modules.values():
            for param in module.parameters():
                if param.requires_grad:
                    param.add_(.1)
    trainer._load_all('best')
    with torch.no_grad():
        _, embeddings = trainer._forward_agents(batch)
        _, fused = trainer.comm(list(embeddings))
        restored = trainer.final_head(fused).sigmoid()[0].item()
    if abs(reference-restored) > 1e-5:
        raise RuntimeError('Checkpoint probability roundtrip mismatch')
    news = f.read_csv(Path(cfg['data']['val_file']))[0]
    write_json(output / 'probe.json', {k:news[k] for k in ('title','text','subject','date') if k in news})
    # Release all GPU references before the independent inference process.
    del module, param, agent, trainer, agents, modules, batch, embeddings, fused, state, hook, group, quantized_modules
    gc.collect(); torch.cuda.empty_cache()
    command = [sys.executable, str(Path(__file__).with_name('infer.py')), '--checkpoint_dir', str(run / 'best'),
               '--input_json', str(output / 'probe.json'), '--output', str(output / 'probe_prediction.json')]
    with (output / 'independent_inference.log').open('w', encoding='utf-8') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    actual = json.loads((output / 'probe_prediction.json').read_text(encoding='utf-8'))['p_positive']
    if abs(actual-reference) > 1e-5:
        raise RuntimeError('Independent inference probability mismatch')
    subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate.py')), str(run / 'val_predictions.csv'),
                    '--output', str(output / 'independent_metrics')], check=True)
    recomputed = json.loads((output / 'independent_metrics/metrics.json').read_text(encoding='utf-8'))
    original = json.loads((run / 'val_metrics.json').read_text(encoding='utf-8'))
    if any(original[k] != recomputed['sample_pooled'][k] for k in SCORES):
        raise RuntimeError('Independent metric mismatch')
    evidence.update(status='passed', checkpoint_max_error=abs(reference-restored),
                    independent_inference_error=abs(actual-reference), metrics_recomputed=True)
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.execute:
        write_json(args.output / 'acceptance.json', dict(status='planned_not_executed',
            config=str(args.config.resolve()), config_sha256=f.file_hash(args.config),
            requirements=['explicit-policy Teacher supervision release', 'pinned Linux environment', 'two CUDA GPUs',
                          '4bit Qwen + LoRA + gradient checkpointing'],
            checks=['two updates, all gradients, parameter independence, frozen backbone unchanged',
                    'model/optimizer/scheduler/RNG roundtrip', 'independent inference and metrics'],
            model_loaded=False, training_started=False))
        print('Plan saved; no model loaded and no training started.')
        return
    try:
        evidence = execute(args.config.resolve(), args.output.resolve())
    except BaseException as error:
        write_json(args.output / 'acceptance.json', dict(status='failed', error_type=type(error).__name__))
        raise
    write_json(args.output / 'acceptance.json', evidence)


if __name__ == '__main__':
    main()
