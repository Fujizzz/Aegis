"""T15 separate-process continuous/interrupted/resumed comparison; dry-run by default."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from utils.run_archive import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare(continuous, resumed):
    import torch
    from server_acceptance import state_digest
    from utils.resume import verify_checkpoint
    from utils.checkpoints import COMPONENTS
    from utils.freeze_pheme import read_csv
    for run in (continuous, resumed):
        if read(run / 'status.json')['status'] != 'completed':
            raise ValueError('Both recovery comparison runs must complete')
        verify_checkpoint(run / 'last')
    components = {}
    for name in COMPONENTS:
        left = torch.load(continuous/'last'/(name+'.pth'), weights_only=True, map_location='cpu')
        right = torch.load(resumed/'last'/(name+'.pth'), weights_only=True, map_location='cpu')
        components[name] = state_digest(left) == state_digest(right)
    left = torch.load(continuous/'last/training_state.pth', weights_only=True, map_location='cpu')
    right = torch.load(resumed/'last/training_state.pth', weights_only=True, map_location='cpu')
    keys = ('epoch','best_epoch','best_score','optimizer','scheduler','optimizer_steps',
            'python_rng','numpy_rng','torch_rng','cuda_rng','parameter_names','updates_per_epoch')
    states = {key:state_digest(left[key]) == state_digest(right[key]) for key in keys}
    if not all(components.values()) or not all(states.values()):
        raise ValueError('Continuous versus resumed model/optimizer/scheduler/RNG/selection differs')
    a, b = read_csv(continuous/'val_predictions.csv'), read_csv(resumed/'val_predictions.csv')
    if [(r['sample_id'],r['target']) for r in a] != [(r['sample_id'],r['target']) for r in b]:
        raise ValueError('Recovery prediction identities differ')
    error = max(abs(float(x['p_positive'])-float(y['p_positive'])) for x,y in zip(a,b))
    if error > 1e-5:
        raise ValueError('Recovered selected-best predictions differ')
    return dict(components_equal=components, training_state_equal=states, prediction_max_error=error,
                epochs=left['epoch'], optimizer_steps=left['optimizer_steps'])


def execute(acceptance, output):
    evidence = read(acceptance/'acceptance.json')
    if evidence['status'] != 'passed':
        raise ValueError('T10 acceptance must pass before T15 independent recovery')
    config = read(acceptance/'run/config.resolved.json')
    if config['experiment']['variant'] != 'full':
        raise ValueError('GPU recovery acceptance must exercise the full four-agent method')
    output.mkdir(parents=True, exist_ok=False)
    try:
        config = copy.deepcopy(config)
        config['runtime'].pop('save_dir', None)
        config['runtime'].pop('run_id', None)
        config['runtime'].pop('stop_after_epoch', None)
        config['runtime'].update(run_test_after_train=False, save_each_epoch=True)
        config['training'].update(epochs=2, warmup_epochs=0, collab_epochs=2)
        for split in ('train','val'):
            config['data'][split+'_file'] = str((acceptance/'acceptance_inputs'/(split+'.csv')).resolve())
        config['data'].pop('test_file', None)
        config['data']['freeze_manifest'] = str((acceptance/'acceptance_inputs/freeze_manifest.json').resolve())
        cfg_path = output/'recovery_config.json'
        write_json(cfg_path, config)
        cli = str(Path(__file__).with_name('train.py'))
        commands = [
            [sys.executable, cli, '--config', str(cfg_path), '--save_dir', str(output/'continuous')],
            [sys.executable, cli, '--config', str(cfg_path), '--save_dir', str(output/'interrupted'), '--stop_after_epoch', '1'],
            [sys.executable, cli, '--resume', str(output/'interrupted/last'), '--save_dir', str(output/'resumed')],
        ]
        original = None
        for i,command in enumerate(commands):
            with (output/f'process_{i+1}.log').open('w', encoding='utf-8') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            if i == 1:
                if read(output/'interrupted/status.json')['status'] != 'interrupted':
                    raise ValueError('Expected an epoch-boundary stop, not completed training')
                original = {str(p.relative_to(output/'interrupted')):sha256(p)
                            for p in (output/'interrupted').rglob('*') if p.is_file()}
        if original != {str(p.relative_to(output/'interrupted')):sha256(p)
                        for p in (output/'interrupted').rglob('*') if p.is_file()}:
            raise ValueError('Source run changed during resume')
        result = dict(status='passed', formal_experiment=False,
            acceptance_sha256=sha256(acceptance/'acceptance.json'),
            data_freeze_sha256=evidence['data_freeze_sha256'], independent_processes=3,
            source_checkpoint_preserved=True, comparison=compare(output/'continuous', output/'resumed'))
        write_json(output/'acceptance.json', result)
        return result
    except BaseException as error:
        write_json(output/'acceptance.json', dict(status='failed', error_type=type(error).__name__))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acceptance-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='planned_not_executed', independent_processes=3,
                              pretrained_model_loaded=False, training_started=False)))
    else:
        print(json.dumps(execute(args.acceptance_dir.resolve(), args.output.resolve()), indent=2))


if __name__ == '__main__':
    main()
