"""Prepare an immutable run, then train from its frozen inputs."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys

from utils.run_archive import SOURCE_ROOT, model_manifest, prepare_run, write_json
from utils.label_schema import SCHEMAS
from utils.resume import prepare_resume_run, verify_checkpoint


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)
    def flush(self):
        self.console.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Config file; with --resume, defaults to the source checkpoint config')
    parser.add_argument('--resume', metavar='CHECKPOINT_DIR', help='Resume a verified epoch checkpoint into a new run')
    for flag in ('train_file', 'val_file', 'test_file', 'save_dir', 'runs_root', 'run_id', 'base_model', 'model_manifest', 'dataset', 'event', 'variant'):
        parser.add_argument('--' + flag)
    parser.add_argument('--device', help='Assign all modules to cpu/cuda:N; overrides config device_map')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--stop_after_epoch', type=int, help='Save a complete epoch and return interrupted; used for recovery acceptance')
    parser.add_argument('--label_schema', choices=sorted(SCHEMAS))
    parser.add_argument('--run_test_after_train', type=int, choices=[0, 1])
    parser.add_argument('--prepare_only', action='store_true', help='Archive and hash inputs without importing torch or loading weights')
    parser.add_argument('--pin_model', metavar='NEW_JSON', help='Create content manifest for --base_model, then exit')
    args = parser.parse_args()
    if args.pin_model and args.resume:
        parser.error('--pin_model cannot be combined with --resume')
    if args.pin_model:
        if not args.base_model:
            parser.error('--pin_model requires --base_model')
        # Exclusive creation prevents accidental replacement of a pinned model identity.
        manifest = model_manifest(args.base_model)
        with Path(args.pin_model).open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2)
        print('Pinned local model:', manifest['identity'])
        return
    if args.resume:
        verify_checkpoint(args.resume)
    path = Path(args.config or (Path(args.resume) / 'config.resolved.json' if args.resume else SOURCE_ROOT / 'config.yaml')).resolve()
    if path.suffix == '.json':
        cfg = json.loads(path.read_text(encoding='utf-8'))
    else:
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if args.resume:
        # Output identity is always new unless explicitly supplied on this CLI.
        cfg.setdefault('runtime', {}).pop('stop_after_epoch', None)
        cfg.setdefault('runtime', {}).pop('save_dir', None)
        cfg['runtime'].pop('run_id', None)
        if not args.config:
            # These immutable run snapshots survive original input relocation.
            source_run = Path(args.resume).resolve().parent
            for split in ('train', 'val', 'test'):
                archived = source_run / 'data' / (split + '.csv')
                if cfg['data'].get(split + '_file') and archived.is_file():
                    cfg['data'][split + '_file'] = str(archived)
            if (source_run / 'model_manifest.json').is_file():
                cfg['model']['manifest'] = str(source_run / 'model_manifest.json')
            freeze = source_run / 'data' / 'freeze_manifest.json'
            if cfg['data'].get('freeze_manifest') and freeze.is_file():
                # prepare_run verifies split paths against the archived content map
                # below using hashes, because archive names differ from freeze paths.
                cfg['data']['freeze_manifest'] = str(freeze)
    sections = {'data': ('train_file', 'val_file', 'test_file', 'label_schema'),
                'runtime': ('save_dir', 'runs_root', 'run_id', 'seed', 'run_test_after_train', 'stop_after_epoch'),
                'model': ('base_model',), 'experiment': ('dataset', 'event', 'variant')}
    for section, names in sections.items():
        for name in names:
            value = getattr(args, name)
            if value is not None:
                cfg.setdefault(section, {})[name] = value
    if args.model_manifest:
        cfg.setdefault('model', {})['manifest'] = args.model_manifest
    if args.device:
        cfg.setdefault('hardware', {})['device_map'] = {key: args.device for key in
            ('agent_emotion', 'agent_exaggeration', 'agent_bias', 'agent_commonsense', 'communication')}
    if cfg.get('experiment', {}).get('variant') in ('tfidf_nb', 'teacher_direct'):
        parser.error('Use run_baseline.py for text/API baselines')
    if args.resume:
        cfg, run = prepare_resume_run(cfg, path, args.resume)
    else:
        cfg, run = prepare_run(cfg, path)
    print('Run directory:', run)
    if args.prepare_only:
        return
    with (run / 'console.log').open('x', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            write_json(run / 'status.json', {'status': 'running'})
            try:
                from check_environment import check
                write_json(run / 'environment' / 'preflight.json', check('training'))
                from training.trainer import Trainer
                trainer = Trainer(cfg)
                trainer.train_loop()
            except BaseException as error:
                # Do not persist exception text: external libraries may include credentials.
                status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
                status.update(status='failed', error_type=type(error).__name__)
                status.setdefault('failed_phase', 'preflight_or_initialization')
                write_json(run / 'status.json', status)
                raise


if __name__ == '__main__':
    main()
