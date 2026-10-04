"""Compare measured costs only when input, hardware and measurement settings match."""
import argparse
import json
from pathlib import Path
from utils.run_archive import sha256, write_json
from utils.freeze_pheme import write_csv
from utils.pseudo_quality import object_hash


def summarize(paths, output):
    if not paths:raise ValueError('Supply one or more benchmark.json files')
    rows=[];groups={};seen=set()
    for path in paths:
        path=Path(path).resolve()
        if path in seen:raise ValueError('Duplicate benchmark file')
        seen.add(path)
        result=json.loads(path.read_text(encoding='utf-8'))
        if result.get('schema')!='aegis_inference_benchmark_v1':raise ValueError('Unknown benchmark schema')
        # Device maps may differ for single versus four encoders; available hardware must match.
        env=result['environment']
        condition={k:result[k] for k in ('source_sha256','sample_ids','model_identity','max_length','quantization',
            'batch_size','warmup_passes','repeats','timing_scope')}
        condition['hardware']={k:env.get(k) for k in ('platform','nvidia_smi','cuda_visible_devices','packages')}
        condition['cpu_name']=result.get('cpu_name')
        key=object_hash(condition)
        groups[key]=condition
        peaks=result['cuda_peak_memory']
        training=result.get('training_resource_profile') or {}
        segment=result.get('training_segment') or {}
        rows.append(dict(condition_group=key,variant=result['experiment']['variant'],event=result['experiment']['event'],
            seed=result['seed'],batch_size=result['batch_size'],rows=result['rows'],
            mean_ms_per_sample=result['mean_ms_per_sample'],pass_sample_std_seconds=result['sample_std_seconds'],
            samples_per_second=result['samples_per_second'],used_cuda_devices=len(peaks),
            training_registered_parameter_elements=training.get('parameters',{}).get('registered_parameter_elements'),
            trainable_parameter_elements=training.get('parameters',{}).get('trainable_parameter_elements'),
            training_segment_seconds=segment.get('seconds'),training_completed_epochs=segment.get('epochs'),
            training_batch_size=training.get('batch_size'),grad_accum_steps=training.get('grad_accum_steps'),
            resumed_segment_only=segment.get('resumed_segment_only'),training_source_sha256=segment.get('source_sha256'),
            sum_device_peak_allocated_bytes=sum(v['peak_allocated_bytes'] for v in peaks.values()) if peaks else None,
            benchmark_sha256=sha256(path),path=str(path)))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'cost_comparison.csv',rows,list(rows[0]))
    result=dict(schema='aegis_cost_comparison_v1',conditions=groups,rows=rows,
        note='Only compare rows within one condition_group. Device-peak sums are not simultaneous process peak. '
             'Repeated measurements are not independent training seeds. Training parameters/times and stability live in analyze_run outputs. '
             'CPU hardware identity is reported separately; verify host equivalence before comparing CPU latency.')
    write_json(output/'cost_comparison.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmarks',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=summarize(args.benchmarks,args.output)
    print(json.dumps(dict(rows=len(result['rows']),condition_groups=len(result['conditions']))))


if __name__=='__main__':main()
