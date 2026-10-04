"""Write a portable T17 -> T19/T20 matrix; never load models or launch training.

The emitted commands must be run later on an accepted training server.
"""
import argparse
import copy
import json
from pathlib import Path

from utils import freeze_pheme as f
from utils.pseudo_quality import write_json, object_hash, load_source
from training.options import resolve_variant
from utils.pseudo_policy import training_release_ready
from utils.experiment_protocol import IN_EVENT, CROSS_EVENT, split_relative

CORE = ('full', 'no_comm', 'no_dim', 'no_gate', 'no_cons', 'no_pred', 'no_div', 'no_gate_loss')
BASELINES = ('single_agent', 'tfidf_nb', 'teacher_direct')


def prepare(config, source, output, *, model_path, model_manifest, events=None, seeds=(42,),
            devices=('cuda:0', 'cuda:1'), include_baselines=False):
    protocol = json.loads((Path(source) / 'protocol.json').read_text(encoding='utf-8'))
    cross = protocol.get('evaluation_scope') == CROSS_EVENT
    if cross:
        from utils.cross_event import verify
        check = verify(source)
        rows = f.read_csv(Path(source)/'parent_canonical.csv')
    else:
        rows, check = load_source(source)
    if len(devices) != 2 or len(set(devices)) != 2 or any(not d.startswith('cuda:') for d in devices):
        raise ValueError('Provide two distinct CUDA devices for the planned acceptance profile')
    events = list(events or f.EVENTS)
    if len(set(events)) != len(events) or not set(events) <= set(f.EVENTS):
        raise ValueError('Unknown/duplicate event')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configs = output / 'configs'
    configs.mkdir()
    records = []
    for event in events:
        for seed in seeds:
            for variant in CORE + (BASELINES if include_baselines else ()):
                if variant in ('tfidf_nb', 'teacher_direct') and seed != seeds[0]:
                    continue  # Fixed text fit/API response cache is not a neural random-seed replicate.
                cfg = copy.deepcopy(config)
                cfg['model']['base_model'] = model_path
                cfg['model']['manifest'] = model_manifest
                cfg['experiment'].update(dataset=CROSS_EVENT if cross else Path(source).name, event=event, variant=variant,
                    evaluation_scope=CROSS_EVENT if cross else IN_EVENT,
                    pseudo_release_policy=protocol.get('pseudo_release_policy', 'unreleased'),
                    human_review_completed=protocol.get('human_review_completed', False))
                if cross:
                    cfg['experiment'].update(held_out_event=event,training_events=[e for e in f.EVENTS if e!=event])
                else:
                    for key in ('held_out_event','training_events'):
                        cfg['experiment'].pop(key,None)
                data_env = '${AEGIS_CROSS_DATA_ROOT}' if cross else '${AEGIS_DATA_ROOT}'
                cfg['data']['freeze_manifest'] = data_env + '/freeze_manifest.json'
                for split in f.SPLITS:
                    cfg['data'][split + '_file'] = data_env + '/' + split_relative(cfg,split)
                cfg['runtime'].update(seed=seed, runs_root='${AEGIS_RUNS_ROOT}')
                cfg['runtime'].pop('save_dir', None)
                cfg['runtime'].pop('run_id', None)
                cfg['runtime'].pop('stop_after_epoch', None)
                cfg['hardware']['device_map'] = {f'agent_{dim}': devices[i // 2] for i, dim in enumerate(f.DIMS)}
                cfg['hardware']['device_map']['communication'] = devices[1]
                if variant in BASELINES:
                    cfg['experiment']['method_id'] = {'single_agent':'single_agent_lora_v1',
                        'tfidf_nb':'tfidf_multinomial_nb_v1', 'teacher_direct':'teacher_direct_pheme_v1'}[variant]
                    cfg['data']['use_pseudo_labels'] = False
                if variant == 'single_agent':
                    cfg['hardware']['device_map'] = {key:devices[0] for key in cfg['hardware']['device_map']}
                if variant in ('tfidf_nb', 'teacher_direct'):
                    cfg['model'].pop('base_model', None)
                    cfg['model'].pop('manifest', None)
                    cfg['baseline'] = dict(tfidf=dict(alpha=1., min_df=2, max_features=50000, ngram_max=2)) if variant == 'tfidf_nb' else dict(
                        teacher=dict(model='deepseek-chat', endpoint='https://api.deepseek.com/chat/completions'))
                # Resolve from the SAME full input configuration for each ablation.
                resolve_variant(copy.deepcopy(cfg))
                name = f'{event}.{variant}.seed{seed}.json'
                write_json(configs / name, cfg)
                cli = 'run_baseline.py' if variant in ('tfidf_nb', 'teacher_direct') else 'train.py'
                records.append(dict(event=event, seed=seed, variant=variant, config='configs/' + name,
                    config_sha256=f.file_hash(configs / name),
                    command=['python', '${AEGIS_SOURCE_ROOT}/' + cli, '--config', 'configs/' + name]))
    ready = training_release_ready(rows)
    pilot = min(events, key=lambda event:(sum(r['event_name'] == event for r in rows), event))
    manifest = dict(schema='aegis_experiment_matrix_v1', dataset=CROSS_EVENT if cross else Path(source).name,
        evaluation_scope=CROSS_EVENT if cross else IN_EVENT,
        source_manifest_sha256=check['freeze_sha256'], pilot_event=pilot, pilot_seed=seeds[0],
        pilot_reason='Smallest event chosen by sample count before observing any model outcome.',
        seed_policy='Neural seeds only; text/API baselines run once per event at the first seed. Frozen membership never changes.',
        include_baselines=include_baselines,
        data_ready_for_formal_training=ready, status='planned_not_executed', runs=records,
        pseudo_release_policy=protocol.get('pseudo_release_policy', 'unreleased'),
        human_review_completed=protocol.get('human_review_completed', False),
        execution_order=['Teacher supervision explicit-policy data release', 'T10 server_acceptance.py', 'T15 resume_acceptance.py',
                         f'T17 {pilot} full seed{seeds[0]} + independent evaluate.py',
                         'T19 same-split baselines', 'T20 core ablations',
                         'T21 seeds42,43,44', 'Cross-event separate leave-one-event-out protocol'],
        limitations=['Baseline configs are included with --include-baselines.',
                     ('Held-out results are reporting only; never tune later folds from earlier test results.' if cross else
                      'No held-out-event split is generated: Cross-event needs a separate frozen protocol.'),
                     'Data seal must match this manifest; regenerate plans after publishing new labels.'])
    if cross:
        manifest['execution_order']=['T10 source-event server acceptance','T15 independent recovery',
            f'Cross-event {pilot} held-out pilot (no test-driven tuning)','Cross-event fixed first-seed baselines and ablations',
            'Cross-event remaining training seeds','Separate cross-event test summary and post-training analyses']
    manifest['matrix_sha256'] = object_hash(manifest)
    write_json(output / 'matrix.json', manifest)
    (output / 'RUN.md').write_text(
        '# Experiment commands\n\n'
        'Set AEGIS_SOURCE_ROOT, AEGIS_DATA_ROOT, AEGIS_MODEL_ROOT, AEGIS_MODEL_MANIFEST, '
        'and AEGIS_RUNS_ROOT to absolute server paths. The model manifest must match the model snapshot. '
        'Launch this matrix with run_experiments.py.\n\n'
        '1. Verify the frozen dataset matches matrix.json. '
        'Teacher-only labels must not be reported as human validated.\n'
        f'2. Run server_acceptance.py with the {pilot} full config.\n'
        f'3. Run resume_acceptance.py, then train.py with configs/{pilot}.full.seed{seeds[0]}.json.\n'
        '4. Run evaluate.py on its val_predictions.csv and test_predictions.csv in separate new output directories.\n'
        '5. Use run_experiments.py --matrix matrix.json --output <absolute-path> --stage all to inspect the plan; '
        'add --execute only on the training server. This sequences acceptance, recovery, pilot, baselines and ablations.\n\n'
        'Never tune on test results or merge repeated seeds into one prediction population.\n', encoding='utf-8')
    if cross:
        (output/'RUN.md').write_text(
            '# Cross-event held-out-event plan\n\nSet AEGIS_CROSS_DATA_ROOT to the adjacent frozen directory; '
            'set model/source/runs environment variables. Use run_experiments.py and '
            'summarize_experiments.py with this matrix in a separate pipeline directory. '
            'T10/T15 and pilot validation gates still apply. All hyperparameters must be fixed before '
            'viewing held-out results. Source validation populations overlap between folds; only '
            'held-out test populations may be pooled.\n',encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--events', nargs='+', choices=f.EVENTS)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--include-baselines', action='store_true')
    p.add_argument('--model-path', default='${AEGIS_MODEL_ROOT}')
    p.add_argument('--model-manifest', default='${AEGIS_MODEL_MANIFEST}')
    args = p.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    result = prepare(config, args.source, args.output, model_path=args.model_path,
                     model_manifest=args.model_manifest, events=args.events, seeds=args.seeds,
                     include_baselines=args.include_baselines)
    print(json.dumps({k:v for k,v in result.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
