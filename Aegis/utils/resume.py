"""Verified epoch-boundary recovery. Preparation does not import torch or unpickle.

Version 1 states predate the input/selection binding and are deliberately refused.
All version 2 tensor files can be read with weights_only=True.
"""
import copy
import json
import math
from pathlib import Path
import random
import shutil
import uuid

from training.options import epoch_counts
from utils.checkpoints import COMPONENTS, checkpoint_config
from utils.label_schema import require_schema_metadata, schema_from_config
from utils.run_archive import model_manifest, prepare_run, sha256, write_json
from utils.text_input import INPUT_FORMAT

MODULE_ATTRS = ('agent_e', 'agent_x', 'agent_b', 'agent_c', 'comm', 'final_head', 'team_layer', 'pred_heads')
METADATA = ('config.resolved.json', 'label_schema.json', 'resume_contract.json', 'selection.json')
FILES = tuple(name + '.pth' for name in COMPONENTS) + ('training_state.pth',) + METADATA


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def config_identity(config):
    """Only storage locations and logging/checkpoint frequency may change."""
    cfg = copy.deepcopy(config)
    for section, keys in {
        'runtime': ('save_dir', 'runs_root', 'run_id', 'log_steps', 'save_each_epoch', 'stop_after_epoch'),
        'data': ('train_file', 'val_file', 'test_file', 'freeze_manifest'),
        'model': ('base_model', 'manifest'),
    }.items():
        for key in keys:
            cfg.get(section, {}).pop(key, None)
    return cfg


def input_contract(config):
    root = Path(config['runtime']['save_dir'])
    data = {split: sha256(config['data'][split + '_file']) for split in ('train', 'val', 'test')
            if config['data'].get(split + '_file')}
    manifest_path = root / 'data_manifest.json'
    if manifest_path.exists():
        declared = {k: v['sha256'] for k, v in read_json(manifest_path).items()}
        if data != declared:
            raise ValueError('Archived data hash mismatch')
    model_path = root / 'model_manifest.json'
    freeze_path = root / 'data' / 'freeze_manifest.json'
    return dict(config=config_identity(config), data=data,
                model=read_json(model_path) if model_path.is_file() else None,
                freeze_sha256=sha256(freeze_path) if freeze_path.is_file() else None)


def verify_checkpoint(directory):
    directory = Path(directory).resolve(strict=True)
    manifest = read_json(directory / 'checkpoint_manifest.json')
    if (manifest.get('format_version') != 2 or not manifest.get('automatic_resume_supported') or
            manifest.get('recovery_boundary') != 'end_of_epoch'):
        raise ValueError('Checkpoint lacks version 2 verified recovery metadata; legacy resume refused')
    if set(manifest.get('sha256', {})) != set(FILES):
        raise ValueError('Incomplete recovery checkpoint manifest')
    for name in FILES:
        if sha256(directory / name) != manifest['sha256'][name]:
            raise ValueError('Recovery checkpoint hash mismatch: ' + name)
    cfg = checkpoint_config(directory)
    require_schema_metadata(directory / 'label_schema.json', schema_from_config(cfg))
    contract = read_json(directory / 'resume_contract.json')
    if contract['config'] != config_identity(cfg) or manifest['run_id'] != cfg['runtime']['run_id']:
        raise ValueError('Recovery config/run identity mismatch')
    selection = read_json(directory / 'selection.json')
    total = sum(epoch_counts(cfg['training']))
    ep = manifest['epoch']
    if type(ep) is not int or not 1 <= ep <= total:
        raise ValueError('Invalid recovery epoch')
    if (type(selection.get('epoch')) is not int or not 1 <= selection['epoch'] <= ep or
            selection.get('run_id') != manifest['run_id'] or selection.get('split') != 'val' or
            selection.get('metric') != cfg['runtime'].get('select_metric', 'binary_f1') or
            not isinstance(selection.get('score'), (int, float)) or not math.isfinite(selection['score']) or
            selection['validation_metrics'].get(selection['metric']) != selection['score']):
        raise ValueError('Invalid recovery best selection')
    return manifest, cfg, contract, selection


def find_selected_best(checkpoint, manifest, selection):
    if selection['epoch'] == manifest['epoch']:
        return checkpoint
    # Old best directories are retained so older epoch checkpoints remain usable.
    run = checkpoint.parent
    expected = selection['checkpoint_manifest_sha256']
    for path in (run / 'best', run / 'best_history' / f"epoch_{selection['epoch']}"):
        if (path / 'checkpoint_manifest.json').is_file() and sha256(path / 'checkpoint_manifest.json') == expected:
            best_manifest, _, _, best_selection = verify_checkpoint(path)
            if (best_manifest['run_id'] == manifest['run_id'] and
                    best_manifest['epoch'] == selection['epoch'] and
                    best_selection['validation_metrics'] == selection['validation_metrics']):
                return path
    raise ValueError('Selected historical best checkpoint is missing or changed')


def prepare_resume_run(config, config_path, checkpoint, **kwargs):
    """Verify the source, create a fresh archive and snapshot recovery inputs."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    manifest, _, contract, selection = verify_checkpoint(checkpoint)
    if contract['model'] is None:
        raise ValueError('Resume requires the archived base model content manifest')
    best = find_selected_best(checkpoint, manifest, selection)
    cfg, run = prepare_run(config, config_path, recovery_contract=contract, **kwargs)
    try:
        if cfg['runtime']['run_id'] == manifest['run_id']:
            raise ValueError('Resume requires a new run_id')
        if input_contract(cfg) != contract:
            raise ValueError('Resume config/data/model/freeze hash mismatch')
        recovery = run / 'recovery'
        recovery.mkdir()
        copied = {}
        for name, source in (('checkpoint', checkpoint), ('best', best)):
            destination = recovery / name
            destination.mkdir()
            # Copy only verified manifest members; never copy arbitrary extra outputs.
            source_hash = sha256(source / 'checkpoint_manifest.json')
            for file in (*FILES, 'checkpoint_manifest.json'):
                shutil.copyfile(source / file, destination / file)
            verify_checkpoint(destination)
            if sha256(destination / 'checkpoint_manifest.json') != source_hash:
                raise ValueError('Checkpoint changed during snapshot')
            copied[name] = source_hash
        lineage = dict(format_version=1, source_checkpoint=str(checkpoint),
                       source_run_id=manifest['run_id'], source_epoch=manifest['epoch'],
                       start_epoch=manifest['epoch'] + 1, recovery_boundary='end_of_epoch',
                       checkpoint_manifest_sha256=copied['checkpoint'],
                       best_manifest_sha256=copied['best'], source_best_checkpoint=str(best),
                       inherited_best_epoch=selection['epoch'])
        write_json(run / 'resume.json', lineage)
        info = read_json(run / 'run.json')
        info['resume'] = lineage
        write_json(run / 'run.json', info)
        write_json(run / 'status.json', dict(status='prepared', run_id=cfg['runtime']['run_id'], resume=lineage))
    except BaseException as error:
        write_json(run / 'status.json', dict(status='preparation_failed', error_type=type(error).__name__))
        raise
    return cfg, run


def verify_resume_inputs(config):
    root = Path(config['runtime']['save_dir'])
    lineage = read_json(root / 'resume.json')
    if read_json(root / 'run.json').get('resume') != lineage:
        raise ValueError('Resume lineage differs from run record')
    manifest, _, contract, selection = verify_checkpoint(root / 'recovery' / 'checkpoint')
    best_manifest, _, best_contract, best_selection = verify_checkpoint(root / 'recovery' / 'best')
    for name, key in (('checkpoint', 'checkpoint_manifest_sha256'), ('best', 'best_manifest_sha256')):
        if sha256(root / 'recovery' / name / 'checkpoint_manifest.json') != lineage[key]:
            raise ValueError('Resume snapshot identity mismatch')
    if (contract != input_contract(config) or best_contract != contract or
            manifest['run_id'] != lineage['source_run_id'] or manifest['epoch'] != lineage['source_epoch'] or
            lineage['start_epoch'] != manifest['epoch'] + 1 or
            best_manifest['run_id'] != manifest['run_id'] or best_manifest['epoch'] != selection['epoch'] or
            best_selection['validation_metrics'] != selection['validation_metrics']):
        raise ValueError('Resume config/data/model/selection mismatch')
    if model_manifest(config['model']['base_model']) != contract['model']:
        raise ValueError('Resume base model content hash mismatch')
    if (selection['epoch'] != manifest['epoch'] and
            selection['checkpoint_manifest_sha256'] != lineage['best_manifest_sha256']):
        raise ValueError('Resume best identity mismatch')
    return lineage


def parameter_names(trainer):
    return [f'{attr}.{name}' for attr in MODULE_ATTRS
            for name, p in getattr(trainer, attr).named_parameters() if p.requires_grad]


def capture_rng():
    import numpy as np
    import torch
    state = np.random.get_state()
    return dict(python_rng=random.getstate(),
                numpy_rng=(state[0], torch.tensor(state[1].astype('int64')), *state[2:]),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import numpy as np
    import torch
    numpy_state = state['numpy_rng']
    random.setstate(state['python_rng'])
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype('uint32'), *numpy_state[2:]))
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def write_manifest(directory, epoch, run_id, *, recoverable=True):
    write_json(directory / 'checkpoint_manifest.json', dict(
        format_version=2, epoch=epoch, run_id=run_id,
        sha256={name: sha256(directory / name) for name in FILES}, input_format=INPUT_FORMAT,
        recovery_boundary='end_of_epoch', automatic_resume_supported=recoverable))


def save_checkpoint(trainer, subdir):
    """Stage complete files, publish manifest last, retain previous bests."""
    import torch
    root = Path(trainer.save_dir)
    destination = root / subdir
    stage = root / ('.checkpoint-' + uuid.uuid4().hex)
    stage.mkdir()
    try:
        for name, attr in zip(COMPONENTS, MODULE_ATTRS):
            torch.save(getattr(trainer, attr).state_dict(), stage / (name + '.pth'))
        state = dict(format_version=2, epoch=trainer.current_epoch, best_epoch=trainer.best_epoch,
                     best_score=trainer.best_score, optimizer=trainer.optim.state_dict(),
                     scheduler=trainer.sched.state_dict(), optimizer_steps=trainer.optimizer_steps,
                     grad_accum_steps=trainer.grad_accum_steps, updates_per_epoch=trainer.updates_per_epoch,
                     parameter_names=parameter_names(trainer), torch_version=str(torch.__version__),
                     run_id=trainer.cfg['runtime']['run_id'], **capture_rng())
        torch.save(state, stage / 'training_state.pth')
        write_json(stage / 'config.resolved.json', trainer.cfg)
        write_json(stage / 'label_schema.json', trainer.label_schema.metadata())
        write_json(stage / 'resume_contract.json', input_contract(trainer.cfg))
        selection = copy.deepcopy(trainer.best_record)
        if selection['epoch'] == trainer.current_epoch:
            selection.pop('checkpoint_manifest_sha256', None)  # This checkpoint itself is the best.
        write_json(stage / 'selection.json', selection)
        # Loading best for inference changes model weights without rolling back the
        # optimizer. Such later ad-hoc exports must never advertise resumability.
        write_manifest(stage, trainer.current_epoch, state['run_id'],
                       recoverable=getattr(trainer, 'loaded_checkpoint_id', None) is None)
        if destination.exists():
            if subdir == 'best':
                previous = root / 'best_history' / f"epoch_{read_json(destination / 'checkpoint_manifest.json')['epoch']}"
                previous.parent.mkdir(exist_ok=True)
            else:
                previous = root / (subdir + '.previous')
                if previous.exists():
                    shutil.rmtree(previous)
            destination.rename(previous)
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_training_state(directory, trainer):
    """Hash and semantics are checked before optimizer/model mutation."""
    import numpy as np
    import torch
    manifest, _, _, selection = verify_checkpoint(directory)
    state = torch.load(Path(directory) / 'training_state.pth', map_location='cpu', weights_only=True)
    if (state['format_version'] != 2 or state['epoch'] != manifest['epoch'] or
            state['run_id'] != manifest['run_id'] or state['best_epoch'] != selection['epoch'] or
            state['best_score'] != selection['score'] or state['grad_accum_steps'] != trainer.grad_accum_steps or
            state['updates_per_epoch'] != trainer.updates_per_epoch or
            state['optimizer_steps'] != state['epoch'] * trainer.updates_per_epoch or
            state['scheduler']['last_epoch'] != state['optimizer_steps'] or
            state['parameter_names'] != parameter_names(trainer) or
            state['torch_version'] != str(torch.__version__)):
        raise ValueError('Incompatible training state/epoch/optimizer/scheduler/runtime')
    if len(state['cuda_rng']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('CUDA RNG device count differs; exact resume refused')
    # Validate RNG on private generators, without perturbing the process state.
    random.Random().setstate(state['python_rng'])
    ns = state['numpy_rng']
    np.random.RandomState().set_state((ns[0], ns[1].numpy().astype('uint32'), *ns[2:]))
    torch.Generator().set_state(state['torch_rng'])
    for rng in state['cuda_rng']:
        if rng.dtype != torch.uint8 or rng.ndim != 1 or not rng.numel():
            raise ValueError('Malformed CUDA RNG state')
    return state


def restore_training(trainer):
    import torch
    root = Path(trainer.save_dir)
    lineage = verify_resume_inputs(trainer.cfg)
    source = root / 'recovery' / 'checkpoint'
    state = load_training_state(source, trainer)
    best_source = root / 'recovery' / 'best'
    best_state = load_training_state(best_source, trainer)
    for name, attr in zip(COMPONENTS, MODULE_ATTRS):
        getattr(trainer, attr).load_state_dict(torch.load(source / (name + '.pth'),
                                                       map_location='cpu', weights_only=True), strict=True)
    # PyTorch maps each optimizer tensor to its own parameter device (step remains
    # on CPU for non-capturable AdamW); do not move all state to communication GPU.
    trainer.optim.load_state_dict(state['optimizer'])
    trainer.sched.load_state_dict(state['scheduler'])
    trainer.optim.zero_grad(set_to_none=True)
    trainer.current_epoch = state['epoch']
    trainer.start_epoch = state['epoch'] + 1
    trainer.optimizer_steps = state['optimizer_steps']
    trainer.best_epoch, trainer.best_score = state['best_epoch'], state['best_score']
    # Bind a copy of the inherited best to this run; original bytes stay under recovery/.
    best = root / 'best'
    shutil.copytree(best_source, best)
    best_state['run_id'] = trainer.cfg['runtime']['run_id']
    torch.save(best_state, best / 'training_state.pth')
    selection = read_json(best / 'selection.json')
    selection.update(run_id=trainer.cfg['runtime']['run_id'])
    selection.pop('checkpoint_manifest_sha256', None)
    write_json(best / 'selection.json', selection)
    write_json(best / 'config.resolved.json', trainer.cfg)
    write_json(best / 'resume_contract.json', input_contract(trainer.cfg))
    write_manifest(best, trainer.best_epoch, trainer.cfg['runtime']['run_id'])
    trainer.best_record = dict(selection, checkpoint_manifest_sha256=sha256(best / 'checkpoint_manifest.json'))
    write_json(root / 'best_selection.json', trainer.best_record)
    from utils.run_status import update_status
    update_status(trainer, phase='resumed', resume=lineage, best=trainer.best_record,
                  training=dict(status='resumed', completed_epoch=state['epoch'], next_epoch=trainer.start_epoch))
    # Initialization, validation and file I/O must finish before restoring global RNG.
    restore_rng(state)
