"""Freeze Cross-event folds and prepare a separately bound experiment matrix; no training."""
import argparse
import json
from pathlib import Path
from utils.cross_event import build, verify
from prepare_experiments import prepare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,help='Parent teacher-only frozen release')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify',action='store_true')
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--include-baselines',action='store_true')
    args=p.parse_args()
    if args.verify:
        result=verify(args.output)
    else:
        if not args.source:p.error('--source is required to build')
        import yaml
        args.output.mkdir(parents=True,exist_ok=False)
        result=build(args.source,args.output/'frozen')
        matrix=prepare(yaml.safe_load(args.config.read_text(encoding='utf-8')),
            args.output/'frozen',args.output/'experiment_plan',model_path='${AEGIS_MODEL_ROOT}',
            model_manifest='${AEGIS_MODEL_MANIFEST}',seeds=args.seeds,include_baselines=args.include_baselines)
        result.update(matrix_runs=len(matrix['runs']),status='planned_not_executed')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
