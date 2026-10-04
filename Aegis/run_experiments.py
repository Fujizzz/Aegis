"""Portable T10 -> T15 -> T17 -> T19/T20/T21 runner. Default: inspect only.

Re-running the same command resumes the ledger. It never silently retries a
failed API request within one invocation, and never overwrites a neural run.
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import subprocess
import sys

from prepare_experiments import CORE, BASELINES
from utils.pseudo_quality import object_hash
from utils.run_archive import sha256, write_json, resolve_config
from utils.experiment_protocol import IN_EVENT, CROSS_EVENT, scope, split_relative

SOURCE = Path(__file__).resolve().parent
STAGES = ('acceptance', 'recovery', 'pilot', 'baselines', 'ablations', 'replicates', 'all')
NON_NEURAL = ('tfidf_nb', 'teacher_direct')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_matrix(path):
    path = Path(path).resolve()
    matrix = read(path)
    body = {k:v for k,v in matrix.items() if k != 'matrix_sha256'}
    if matrix.get('schema') != 'aegis_experiment_matrix_v1' or object_hash(body) != matrix.get('matrix_sha256'):
        raise ValueError('Invalid experiment matrix hash/schema')
    seen = set()
    for item in matrix['runs']:
        cfg_path = (path.parent / item['config']).resolve()
        if not cfg_path.is_relative_to(path.parent) or sha256(cfg_path) != item['config_sha256']:
            raise ValueError('Configuration path/hash differs from the matrix')
        cfg = read(cfg_path)
        if scope(cfg) != matrix.get('evaluation_scope',IN_EVENT):
            raise ValueError('Configuration evaluation scope differs from matrix')
        key = run_key(item)
        if key in seen or item['variant'] not in CORE + BASELINES:
            raise ValueError('Duplicate or unsupported matrix run')
        seen.add(key)
        if (cfg['experiment']['event'], cfg['experiment']['variant'], cfg['runtime']['seed'], cfg['experiment']['dataset']) != (
                item['event'], item['variant'], item['seed'], matrix['dataset']):
            raise ValueError('Configuration identity differs from the matrix')
    pilot_record(matrix)
    return matrix


def run_key(item):
    return f"{item['event']}.{item['variant']}.seed{item['seed']}"


def pilot_record(matrix):
    matches = [r for r in matrix['runs'] if r['event'] == matrix['pilot_event'] and
               r['seed'] == matrix['pilot_seed'] and r['variant'] == 'full']
    if len(matches) != 1:
        raise ValueError('Matrix requires one full pilot')
    return matches[0]


def select(matrix, stage):
    pilot = pilot_record(matrix)
    if stage == 'pilot':
        return [pilot]
    if stage == 'baselines':
        return [r for r in matrix['runs'] if r['variant'] in BASELINES and r['seed'] == matrix['pilot_seed']]
    if stage == 'ablations':
        return [r for r in matrix['runs'] if r['variant'] in CORE and r != pilot and r['seed'] == matrix['pilot_seed']]
    if stage == 'replicates':
        return [r for r in matrix['runs'] if r['seed'] != matrix['pilot_seed']]
    if stage == 'all':
        return [pilot] + select(matrix, 'baselines') + select(matrix, 'ablations') + select(matrix, 'replicates')
    return []


def source_identity():
    files = list(SOURCE.glob('*.py'))
    for folder in ('agents','communication','inference','training','utils','baselines','environment'):
        files.extend(p for p in (SOURCE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return object_hash({p.relative_to(SOURCE).as_posix():sha256(p) for p in sorted(files)})


@contextmanager
def locked(output):
    """OS lock auto-releases on crash; a stale lock file does not block recovery."""
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.pipeline.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if sys.platform == 'win32':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == 'win32':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def invoke(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, cwd=SOURCE)


def verify_run(root, item, matrix, matrix_path):
    """Recompute metrics and bind predictions to archived data/config/checkpoint."""
    from evaluate import evaluate_files
    from utils.freeze_pheme import read_csv
    from utils.label_schema import schema_from_config
    if read(root/'status.json')['status'] != 'completed':
        raise ValueError('Run has not completed')
    cfg = read(root/'config.resolved.json')
    expected = resolve_config(read(matrix_path.parent/item['config']), matrix_path.parent/item['config'])
    def comparable(config):
        value = copy.deepcopy(config)
        for k in ('save_dir','run_id','runs_root','stop_after_epoch'):
            value['runtime'].pop(k, None)
        for k in ('train_file','val_file','test_file','freeze_manifest'):
            value['data'].pop(k, None)
        value['model'].pop('manifest', None)
        return value
    if comparable(cfg) != comparable(expected):
        raise ValueError('Completed run configuration differs from the matrix')
    if sha256(root/'data/freeze_manifest.json') != matrix['source_manifest_sha256']:
        raise ValueError('Completed run belongs to a different frozen dataset')
    if item['variant'] not in NON_NEURAL and read(root/'model_manifest.json') != read(expected['model']['manifest']):
        raise ValueError('Completed run used a different base model')
    archive = read(root/'data_manifest.json')
    frozen = read(root/'data/freeze_manifest.json')['files']
    for split, info in archive.items():
        path = root/'data'/(split+'.csv')
        if sha256(path) != info['sha256'] or frozen.get(split_relative(cfg,split)) != info['sha256']:
            raise ValueError('Archived data integrity failure')
    schema = schema_from_config(cfg)
    reports = {}
    for split in ('val','test'):
        prediction = root/(split+'_predictions.csv')
        manifest = read(prediction.with_suffix('.manifest.json'))
        if manifest['source_sha256'] != archive[split]['sha256']:
            raise ValueError('Predictions reference different data')
        checkpoint = Path(manifest['checkpoint_manifest']).resolve()
        if not checkpoint.is_relative_to(root.resolve()) or sha256(checkpoint) != manifest['checkpoint_manifest_sha256']:
            raise ValueError('Prediction checkpoint provenance differs')
        if item['variant'] not in NON_NEURAL:
            from utils.resume import verify_checkpoint
            verify_checkpoint(checkpoint.parent)
        elif item['variant'] == 'tfidf_nb':
            for name, digest in read(checkpoint)['sha256'].items():
                if sha256(checkpoint.parent/name) != digest:
                    raise ValueError('Text model checksum failure')
        else:
            teacher = read(checkpoint)
            if sha256(root/'teacher_responses.jsonl') != teacher['responses_sha256']:
                raise ValueError('Teacher response archive checksum failure')
            from baselines.teacher_direct import make_plan, load_log
            plan = make_plan(cfg)
            if read(root/'teacher_request_plan.json') != plan or teacher['plan_sha256'] != plan['plan_sha256']:
                raise ValueError('Teacher classification prompt/input binding differs')
            accepted, _ = load_log(root/'teacher_responses.jsonl', plan)
            if len(accepted) != len(plan['requests']):
                raise ValueError('Incomplete teacher classification responses')
        original = read_csv(root/'data'/(split+'.csv'))
        predicted = read_csv(prediction)
        if [(r['sample_id'], schema.encode_source(r['label']),r['event_name']) for r in original] != [
                (r['sample_id'], int(r['target']),r['event']) for r in predicted]:
            raise ValueError('Prediction population/labels differ from the frozen split')
        if any((r['variant'],int(r['seed']),r.get('fold',r['event']),r['split']) != (
                item['variant'],item['seed'],item['event'],split) for r in predicted):
            raise ValueError('Prediction experiment identity differs')
        if any(r.get('evaluation_scope',IN_EVENT)!=scope(cfg) for r in predicted):
            raise ValueError('Prediction evaluation scope differs')
        reports[split] = evaluate_files([prediction])
    return reports


def require_gate(path, matrix, *, config_sha=None, acceptance_sha=None):
    evidence = read(path/'acceptance.json')
    if evidence['status'] != 'passed' or evidence.get('data_freeze_sha256') != matrix['source_manifest_sha256']:
        raise ValueError('Required real-server acceptance has not passed for this data version')
    if config_sha is not None and evidence.get('source_config_sha256') != config_sha:
        raise ValueError('T10 acceptance used a different configuration')
    if acceptance_sha is not None and evidence.get('acceptance_sha256') != acceptance_sha:
        raise ValueError('T15 recovery belongs to a different T10 acceptance')
    return evidence


def execute(matrix_path, output, stage):
    matrix_path, output = Path(matrix_path).resolve(), Path(output).resolve()
    matrix = load_matrix(matrix_path)
    if not matrix['data_ready_for_formal_training']:
        raise ValueError('The matrix was generated before data release')
    pilot = pilot_record(matrix)
    config_path = matrix_path.parent/pilot['config']
    resolved = resolve_config(read(config_path), config_path)
    if sha256(resolved['data']['freeze_manifest']) != matrix['source_manifest_sha256']:
        raise ValueError('Server data freeze differs from the planned release')
    with locked(output):
        ledger_path = output/'ledger.json'
        identity = dict(matrix_sha256=matrix['matrix_sha256'], source_sha256=source_identity())
        ledger = read(ledger_path) if ledger_path.exists() else dict(**identity, runs={}, gates={})
        if any(ledger.get(k) != v for k,v in identity.items()):
            raise ValueError('Pipeline source/matrix changed; use a new output directory')
        write_json(ledger_path, ledger)
        def gate(name, script, args, **binding):
            previous = ledger['gates'].get(name, [])
            if previous and (Path(previous[-1])/'acceptance.json').exists():
                if read(Path(previous[-1])/'acceptance.json').get('status') == 'passed':
                    require_gate(Path(previous[-1]), matrix, **binding)
                    return Path(previous[-1])
            path = output/name/f'attempt_{len(previous)+1:03d}'
            ledger['gates'].setdefault(name, []).append(str(path))
            write_json(ledger_path, ledger)
            invoke([sys.executable,str(SOURCE/script), *args,'--output',str(path),'--execute'],
                   output/'logs'/f'{name}_{len(previous)+1:03d}.log')
            require_gate(path, matrix, **binding)
            return path
        if stage in ('acceptance','all'):
            t10 = gate('T10','server_acceptance.py',['--config',str(config_path)], config_sha=pilot['config_sha256'])
        else:
            t10 = Path(ledger['gates']['T10'][-1])
            require_gate(t10, matrix, config_sha=pilot['config_sha256'])
        if stage == 'acceptance':
            return ledger
        if stage in ('recovery','all'):
            gate('T15','resume_acceptance.py',['--acceptance-dir',str(t10)], acceptance_sha=sha256(t10/'acceptance.json'))
        else:
            require_gate(Path(ledger['gates']['T15'][-1]), matrix, acceptance_sha=sha256(t10/'acceptance.json'))
        if stage == 'recovery':
            return ledger
        if stage in ('baselines','ablations','replicates'):
            previous = ledger['runs'].get(run_key(pilot), [])
            if not previous:
                raise ValueError('Complete the pilot before batch experiments')
            verify_run(Path(previous[-1]), pilot, matrix, matrix_path)
        if stage == 'replicates':
            for item in select(matrix,'baselines') + select(matrix,'ablations'):
                previous = ledger['runs'].get(run_key(item), [])
                if not previous:
                    raise ValueError('Complete the first-seed comparisons before seed replicates')
                verify_run(Path(previous[-1]), item, matrix, matrix_path)
        for item in select(matrix, stage):
            key = run_key(item)
            attempts = ledger['runs'].setdefault(key, [])
            old = Path(attempts[-1]) if attempts else None
            if old and (old/'status.json').exists() and read(old/'status.json')['status'] == 'completed':
                verify_run(old, item, matrix, matrix_path)
                continue
            api_resume = item['variant'] == 'teacher_direct' and old and (old/'config.resolved.json').exists()
            root = old if api_resume else output/'runs'/key/f'attempt_{len(attempts)+1:03d}'
            if not api_resume:
                attempts.append(str(root))
                write_json(ledger_path, ledger)
            cfg_path = matrix_path.parent/item['config']
            if item['variant'] in NON_NEURAL:
                command = [sys.executable,str(SOURCE/'run_baseline.py'),'--config',str(cfg_path),
                           '--output',str(root),'--execute'] + (['--resume'] if api_resume else [])
            else:
                recoverable = next((Path(p)/'last' for p in reversed(attempts[:-1])
                                    if (Path(p)/'last/checkpoint_manifest.json').exists()), None)
                command = ([sys.executable,str(SOURCE/'train.py'),'--resume',str(recoverable),'--save_dir',str(root)]
                           if recoverable else [sys.executable,str(SOURCE/'train.py'),'--config',str(cfg_path),'--save_dir',str(root)])
            invoke(command, output/'logs'/(key+f'_{len(attempts):03d}.log'))
            report = verify_run(root, item, matrix, matrix_path)
            write_json(root/'pipeline_verification.json',dict(matrix_sha256=matrix['matrix_sha256'], metrics=report))
        write_json(output/'last_stage.json', dict(stage=stage, status='completed',matrix_sha256=matrix['matrix_sha256']))
        return ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=STAGES, default='all')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    matrix = load_matrix(args.matrix)
    if not args.execute:
        print(json.dumps(dict(status='planned_not_executed',stage=args.stage,
            source_manifest_sha256=matrix['source_manifest_sha256'],
            order=['T10','T15'] if args.stage == 'all' else [args.stage],
            runs=[run_key(r) for r in select(matrix,args.stage)],training_started=False,network_called=False),indent=2))
        return
    execute(args.matrix, args.output, args.stage)
    print(json.dumps(dict(status='completed',stage=args.stage,output=str(args.output))))


if __name__ == '__main__':
    main()
