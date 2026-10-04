"""Teacher supervision/T25/T26 offline analysis of a completed, archived run. Never loads a model."""
import argparse
import json
from pathlib import Path
from evaluate import evaluate_files
from utils.freeze_pheme import read_csv, write_csv
from utils.run_archive import sha256, write_json
from utils.experiment_protocol import scope, CROSS_EVENT
from utils.experiment_protocol import split_relative
from utils.label_schema import schema_from_config
from utils.post_analysis import expertise, cases, stability_report, describe, compare_predictions


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()] if Path(path).exists() else []


def analyze(run, output, *, split='test', reference_freeze=None, benchmarks=(), per_stratum=5, seed=42,
            comparison_predictions=None):
    run,output=Path(run).resolve(),Path(output)
    if output.exists():raise FileExistsError(output)
    if split not in ('val','test'):raise ValueError('Expected val/test')
    if read(run/'status.json')['status']!='completed':raise ValueError('Analysis requires a completed run')
    cfg=read(run/'config.resolved.json');schema=schema_from_config(cfg)
    source=run/'data'/(split+'.csv');prediction=run/(split+'_predictions.csv')
    artifact=read(prediction.with_suffix('.manifest.json'))
    archive=read(run/'data_manifest.json')
    for part,info in archive.items():
        if sha256(run/'data'/(part+'.csv'))!=info['sha256']:raise ValueError('Archived data hash mismatch')
    if artifact['source_sha256']!=sha256(source):raise ValueError('Prediction source binding mismatch')
    checkpoint=Path(artifact['checkpoint_manifest']).resolve()
    if not checkpoint.is_relative_to(run) or sha256(checkpoint)!=artifact['checkpoint_manifest_sha256']:
        raise ValueError('Checkpoint provenance mismatch')
    if cfg['experiment']['variant'] not in ('tfidf_nb','teacher_direct'):
        from utils.resume import verify_checkpoint
        verify_checkpoint(checkpoint.parent)
    if (run/'data/freeze_manifest.json').exists():
        frozen=read(run/'data/freeze_manifest.json')['files']
        for part,info in archive.items():
            if frozen.get(split_relative(cfg,part))!=info['sha256']:
                raise ValueError('Archived inputs differ from frozen fold')
    metrics=evaluate_files([prediction])
    if metrics['evaluation_scope']!=scope(cfg):raise ValueError('Prediction scope differs from run')
    rows=read_csv(prediction);original=read_csv(source);train=read_csv(run/'data/train.csv')
    if [(r['sample_id'],r['event_name'],schema.encode_source(r['label'])) for r in original]!=[
        (r['sample_id'],r['event'],int(r['target'])) for r in rows]:raise ValueError('Prediction population mismatch')
    if any((r['variant'],int(r['seed']),r['split'],r.get('fold',r['event']))!=(
        cfg['experiment']['variant'],cfg['runtime']['seed'],split,cfg['experiment']['event']) for r in rows):
        raise ValueError('Prediction run identity mismatch')
    refs={r['sample_id']:r for r in original};reference_binding=None
    if reference_freeze:
        root=Path(reference_freeze)
        if sha256(root/'freeze_manifest.json')!=sha256(run/'data/freeze_manifest.json'):
            raise ValueError('Reference freeze differs from run freeze')
        if scope(cfg)==CROSS_EVENT:
            from utils.cross_event import verify
            verify(root);refpath=root/'parent_canonical.csv'
        else:
            from utils.freeze_pheme import verify
            verify(root,recompute_near=False);refpath=root/'canonical.csv'
        references={r['sample_id']:r for r in read_csv(refpath)}
        for row in original:
            ref=references[row['sample_id']]
            if any(row[k]!=ref[k] for k in ('event_name','label','text')):raise ValueError('Teacher reference identity/text mismatch')
            refs[row['sample_id']]=ref
        reference_binding=dict(path=str(refpath.resolve()),sha256=sha256(refpath))
    spec=expertise(rows,refs,train)
    comparison=None
    if comparison_predictions:
        other=evaluate_files([comparison_predictions])
        if other['evaluation_scope']!=scope(cfg):raise ValueError('Comparison scope differs')
        rows,comparison=compare_predictions(rows,read_csv(comparison_predictions))
        comparison.update(path=str(Path(comparison_predictions).resolve()),sha256=sha256(comparison_predictions))
    selected,selection=cases(rows,refs,per_stratum=per_stratum,seed=seed)
    epochs=jsonl(run/'metrics.jsonl');micro=jsonl(run/'stability.jsonl');updates=jsonl(run/'optimizer_steps.jsonl')
    cost=dict(resource_profile=read(run/'resource_profile.json') if (run/'resource_profile.json').exists() else None,
        epoch_seconds=describe(r['elapsed_s'] for r in epochs),training_segment_seconds=sum(r['elapsed_s'] for r in epochs),
        epoch_cuda_peaks=[dict(epoch=r['epoch'],devices=r.get('cuda_peak_memory')) for r in epochs],
        resumed_segment_only=(run/'resume.json').exists(),benchmarks=[],
        note='Epoch sums exclude validation/checkpoint/model load; resumed directories contain only this segment. '
             'Missing resource data is unavailable, not zero. API billing is not inferred from neural cost.')
    for path in benchmarks:
        value=read(path)
        if value['checkpoint_manifest_sha256']!=artifact['checkpoint_manifest_sha256']:
            raise ValueError('Benchmark used a different checkpoint')
        if value['source_sha256']!=archive['val']['sha256']:
            raise ValueError('Benchmark did not use this run validation population')
        cost['benchmarks'].append(dict(path=str(Path(path).resolve()),sha256=sha256(path),measurement=value))
    inputs={str(p.relative_to(run)):sha256(p) for p in (prediction,prediction.with_suffix('.manifest.json'),
        source,run/'data/train.csv',run/'config.resolved.json',run/'status.json',run/'data_manifest.json',checkpoint)}
    for name in ('resource_profile.json','metrics.jsonl','stability.jsonl','optimizer_steps.jsonl'):
        if (run/name).exists():inputs[name]=sha256(run/name)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'expertise.json',spec)
    write_json(output/'cost.json',cost)
    write_json(output/'stability.json',stability_report(micro,updates))
    write_json(output/'case_selection.json',selection)
    if comparison:write_json(output/'comparison.json',comparison)
    write_csv(output/'cases.csv',selected,list(selected[0]))
    result=dict(schema='aegis_post_analysis_v1',status='completed',run=str(run),split=split,
        evaluation_scope=scope(cfg),input_hashes=inputs,teacher_reference=reference_binding,
        expertise_status=spec['status'],metrics=metrics,selected_cases=len(selected),
        human_validation_claim=False,formal_experiment_claim='Only the supplied run; fixtures are not PHEME research results.')
    write_json(output/'analysis.json',result)
    write_json(output/'artifacts.json',{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--reference-freeze',type=Path)
    p.add_argument('--benchmarks',type=Path,nargs='*',default=[])
    p.add_argument('--per-stratum',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--comparison-predictions',type=Path,help='Optional same-population earlier-stage or baseline canonical CSV')
    args=p.parse_args()
    result=analyze(args.run,args.output,split=args.split,reference_freeze=args.reference_freeze,
                   benchmarks=args.benchmarks,per_stratum=args.per_stratum,seed=args.seed,
                   comparison_predictions=args.comparison_predictions)
    print(json.dumps(dict(status=result['status'],selected_cases=result['selected_cases'],expertise_status=result['expertise_status'])))


if __name__=='__main__':main()
