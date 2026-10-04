"""T19 text/API baselines. Dry-run by default; --execute is required for fitting/calls."""
import argparse
import copy
import json
from pathlib import Path
from utils.run_archive import prepare_run, sha256, write_json, resolve_config
from baselines.common import run_text
from baselines import teacher_direct


def execute(config, config_path, output, *, limit=None, resume=False):
    config = copy.deepcopy(config)
    config['runtime']['save_dir'] = str(Path(output).resolve())
    variant = config['experiment']['variant']
    if variant not in ('tfidf_nb', 'teacher_direct'):
        raise ValueError('Use train.py for neural variants')
    if resume:
        if variant != 'teacher_direct':
            raise ValueError('Text baseline fits are atomic runs; only API requests resume here')
        root = Path(output)
        cfg = json.loads((root / 'config.resolved.json').read_text(encoding='utf-8'))
        if json.loads((root / 'config.requested.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Requested baseline config changed')
        inputs = json.loads((root / 'data_manifest.json').read_text(encoding='utf-8'))
        for split, info in inputs.items():
            if sha256(cfg['data'][split + '_file']) != info['sha256']:
                raise ValueError('Archived baseline data changed')
        # Revalidate the complete response set and exported predictions even for
        # a finished run. generate() skips every accepted request, so this costs no API calls.
    else:
        cfg, root = prepare_run(config, config_path, require_model=False)
    try:
        write_json(root / 'status.json', dict(status='running', variant=variant))
        result = teacher_direct.run(cfg, limit=limit) if variant == 'teacher_direct' else run_text(cfg)
        status = 'partial' if result.get('status') == 'partial' else 'completed'
        write_json(root / 'status.json', dict(status=status, variant=variant, result=result))
        return dict(status=status, output=str(root))
    except BaseException as error:
        write_json(root / 'status.json', dict(status='failed', error_type=type(error).__name__))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if not args.execute:
        if config['experiment']['variant'] not in ('tfidf_nb','teacher_direct'):
            parser.error('Expected a non-neural baseline config')
        print(json.dumps(dict(status='planned_not_executed', variant=config['experiment']['variant'],
            config_sha256=sha256(args.config), training_started=False, network_called=False)))
        return
    print(json.dumps(execute(config, args.config, args.output, limit=args.limit, resume=args.resume)))


if __name__ == '__main__':
    main()
