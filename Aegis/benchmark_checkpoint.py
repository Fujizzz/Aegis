"""T25 repeatable end-to-end inference benchmark. Default: plan only, no model load."""
import argparse
import json
import statistics
import time
import platform
from pathlib import Path
from utils.run_archive import sha256, write_json, environment
from utils.freeze_pheme import read_csv, digest


def measure(predictor, rows, *, batch_size=1, warmup=2, repeats=5, clock=time.perf_counter):
    from utils.diagnostics import cuda_devices, reset_memory, memory, synchronize, parameters
    if any(type(v) is not int or v<1 for v in (batch_size,warmup,repeats)) or not rows:
        raise ValueError('Nonempty input and positive batch/warmup/repeats required')
    modules=[predictor.ae,predictor.ax,predictor.ab,predictor.ac,predictor.comm,predictor.final_head,predictor.team_layer]
    devices=cuda_devices(modules)
    batches=[rows[i:i+batch_size] for i in range(0,len(rows),batch_size)]
    # Warm up the same population/shape sequence; not included in latency samples.
    for _ in range(warmup):
        for batch in batches:predictor.predict_batch(batch,diagnostics=False)
    reset_memory(devices)
    passes=[]; batch_seconds=[]
    for _ in range(repeats):
        samples=[]
        for batch in batches:
            synchronize(devices);start=clock()
            predictor.predict_batch(batch,diagnostics=False)
            synchronize(devices);samples.append(clock()-start)
        passes.append(sum(samples));batch_seconds.append(samples)
    return dict(schema='aegis_inference_benchmark_v1',batch_size=batch_size,warmup_passes=warmup,repeats=repeats,
        rows=len(rows),pass_seconds=passes,batch_seconds=batch_seconds,
        mean_pass_seconds=statistics.mean(passes),sample_std_seconds=statistics.stdev(passes) if repeats>1 else None,
        mean_ms_per_sample=1000*statistics.mean(passes)/len(rows),
        samples_per_second=len(rows)/statistics.mean(passes),cuda_peak_memory=memory(devices),
        parameters=parameters(modules),
        parameter_scope='Inference modules only; use archived resource_profile for training parameter counts including auxiliary heads.',
        timing_scope='Tokenization + device transfer + encoders + communication + final classifier + CPU probabilities; '
                     'CUDA synchronized on every used device; model load, warmup and dimension diagnostics excluded.',
        memory_scope='PyTorch process allocator peaks after warmup; includes loaded model; CPU RSS not measured.',
        no_sparse_speedup_claim=True)


def execute(checkpoint, output, *, batch_size=1, warmup=2, repeats=5, limit=128, device=None):
    from utils.checkpoints import checkpoint_config, checkpoint_paths
    from utils.resume import verify_checkpoint
    from inference.predictor import Predictor
    checkpoint,output=Path(checkpoint).resolve(),Path(output)
    if output.exists():raise FileExistsError(output)
    if limit<1:raise ValueError('limit must be positive')
    verify_checkpoint(checkpoint)
    cfg=checkpoint_config(checkpoint)
    run=checkpoint.parent
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    source=run/'data/val.csv'
    if sha256(source)!=manifest['val']['sha256']:raise ValueError('Archived validation data changed')
    # A stable hash order chosen without outcomes supports fair matched-input comparisons.
    rows=sorted(read_csv(source),key=lambda r:(digest(r['sample_id']),r['sample_id']))[:limit]
    if device:cfg['hardware']['device_map']=dict.fromkeys(cfg['hardware']['device_map'],device)
    model=Predictor(cfg,checkpoint_paths(checkpoint))
    result=measure(model,rows,batch_size=batch_size,warmup=warmup,repeats=repeats)
    result.update(checkpoint_manifest_sha256=sha256(checkpoint/'checkpoint_manifest.json'),
        model_identity=json.loads((run/'model_manifest.json').read_text(encoding='utf-8'))['identity'],
        source_sha256=sha256(source),sample_ids=[r['sample_id'] for r in rows],
        experiment=cfg['experiment'],seed=cfg['runtime']['seed'],max_length=cfg['data'].get('max_length',512),
        quantization=cfg['model'].get('quantization'),device_map=cfg['hardware']['device_map'],
        cpu_name=platform.processor(),environment=environment())
    profile=run/'resource_profile.json'
    result['training_resource_profile']=json.loads(profile.read_text(encoding='utf-8')) if profile.exists() else None
    result['training_resource_profile_sha256']=sha256(profile) if profile.exists() else None
    log=run/'metrics.jsonl'
    epochs=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()] if log.exists() else []
    result['training_segment']=dict(epochs=len(epochs),seconds=sum(r['elapsed_s'] for r in epochs),
        source_sha256=manifest['train']['sha256'],metrics_sha256=sha256(log) if log.exists() else None,
        resumed_segment_only=(run/'resume.json').exists(),
        cuda_peaks=[dict(epoch=r['epoch'],devices=r.get('cuda_peak_memory')) for r in epochs])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'benchmark.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--limit',type=int,default=128)
    p.add_argument('--device')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='planned_not_executed',model_loaded=False,network_called=False,
                              batch_size=args.batch_size,warmup=args.warmup,repeats=args.repeats)))
        return
    result=execute(args.checkpoint_dir,args.output,batch_size=args.batch_size,warmup=args.warmup,
                   repeats=args.repeats,limit=args.limit,device=args.device)
    print(json.dumps(dict(status='completed',mean_ms_per_sample=result['mean_ms_per_sample'])))


if __name__=='__main__':main()
