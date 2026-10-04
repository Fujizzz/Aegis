"""Recompute all completed matrix cells; report missing cells and seed dispersion."""
import argparse
import csv
import statistics
from pathlib import Path
from run_experiments import read, load_matrix, run_key, verify_run
from utils.run_archive import write_json
from training.metrics import SCORES
from evaluate import evaluate_files
from utils.experiment_protocol import IN_EVENT, CROSS_EVENT


def summarize(matrix_path, pipeline, output):
    matrix_path, pipeline, output = Path(matrix_path).resolve(), Path(pipeline).resolve(), Path(output)
    matrix = load_matrix(matrix_path)
    cross = matrix.get('evaluation_scope',IN_EVENT) == CROSS_EVENT
    ledger = read(pipeline/'ledger.json')
    if ledger['matrix_sha256'] != matrix['matrix_sha256']:
        raise ValueError('Result ledger belongs to a different matrix')
    groups, missing, cells = {}, [], []
    for item in matrix['runs']:
        key = run_key(item)
        attempts = ledger['runs'].get(key, [])
        root = Path(attempts[-1]) if attempts else None
        if not root or not (root/'status.json').exists() or read(root/'status.json')['status'] != 'completed':
            missing.append(key)
            continue
        reports = verify_run(root,item,matrix,matrix_path)
        for split,report in reports.items():
            cells.append(dict(event=item['event'],variant=item['variant'],seed=item['seed'],split=split,
                              **{k:report['sample_pooled'][k] for k in SCORES},run=str(root)))
            if not (cross and split=='val'):
                groups.setdefault((item['variant'],item['seed'],split),[]).append(root/(split+'_predictions.csv'))
    events = {r['event'] for r in matrix['runs']}
    by_seed = []
    for (variant,seed,split),paths in sorted(groups.items()):
        report = evaluate_files(paths)
        by_seed.append({**report, 'variant':variant, 'seed':seed, 'split':split,
            'complete_event_coverage':set(report['per_event']) == events})
    dispersion = []
    for variant in sorted({r['variant'] for r in matrix['runs']}):
        expected_seeds = {r['seed'] for r in matrix['runs'] if r['variant']==variant}
        for split in ('val','test'):
            if cross and split=='val':
                # Source validation members repeat across folds. Summarize fold scores only.
                runs=[]
                for seed in sorted(expected_seeds):
                    fold_cells=[r for r in cells if r['variant']==variant and r['seed']==seed and r['split']=='val']
                    if {r['event'] for r in fold_cells}==events:
                        runs.append(dict(seed=seed,fold_equal_mean={k:statistics.mean(r[k] for r in fold_cells)
                            if all(r[k] is not None for r in fold_cells) else None for k in SCORES}))
                aggregations=('fold_equal_mean',)
            else:
                runs = [r for r in by_seed if r['variant']==variant and r['split']==split and r['complete_event_coverage']]
                aggregations=('event_equal_mean','sample_pooled')
            complete = {r['seed'] for r in runs} == expected_seeds
            for aggregation in aggregations:
                for metric in SCORES:
                    values = [r[aggregation][metric] for r in runs]
                    defined = complete and bool(values) and all(v is not None for v in values)
                    dispersion.append(dict(variant=variant,split=split,aggregation=aggregation,metric=metric,
                        planned_seeds=len(expected_seeds),complete_seeds=len(runs),complete=complete,
                        mean=statistics.mean(values) if defined else None,
                        sample_std=statistics.stdev(values) if defined and len(values)>1 else None))
    result = dict(format='aegis_matrix_summary_v1',matrix_sha256=matrix['matrix_sha256'],
        evaluation_scope=matrix.get('evaluation_scope',IN_EVENT),
        source_manifest_sha256=matrix['source_manifest_sha256'],missing=missing,cells=cells,by_seed=by_seed,
        dispersion=dispersion,notes=['Seeds are never pooled as extra independent samples.',
        'sample_std uses ddof=1; undefined values and incomplete comparisons remain null.',
        'Text/API baselines have one cached run per event, not three fabricated seed replicates.',
        'Teacher pseudo labels have no human-validation claim. Test results must not guide tuning.',
        'Cross-fold source validation is summarized by fold, never pooled as extra independent samples.'])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'summary.json',result)
    for name,rows in (('cells',cells),('seed_summary',dispersion)):
        if rows:
            with (output/(name+'.csv')).open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--pipeline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=summarize(args.matrix,args.pipeline,args.output)
    print(f"Verified {len(result['cells'])} split results; missing runs: {len(result['missing'])}")


if __name__=='__main__':
    main()
