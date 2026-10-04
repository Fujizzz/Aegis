"""Review experiment entry point: default plan only; --simulate is offline; --execute permits API calls."""
import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.protocol import configuration, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--input', type=Path, help='Override input_csv (relative to config directory)')
    parser.add_argument('--output', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Default: freeze plan without loading any transport')
    mode.add_argument('--simulate', action='store_true', help='Run deterministic offline mock responses')
    mode.add_argument('--execute', action='store_true', help='Explicitly enable real API requests')
    mode.add_argument('--analyze-only', action='store_true', help='Recompute from archived journal, no API')
    parser.add_argument('--fixtures', type=Path, help='Simulation JSON keyed by sample_id')
    parser.add_argument('--max-samples', type=int, help='Complete at most this many pending samples this invocation')
    parser.add_argument('--retry-unknown', action='store_true', help='Permit retry of an interrupted, possibly billed call')
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        if args.fixtures and not args.simulate:
            raise ValueError('--fixtures requires --simulate')
        if args.analyze_only:
            from experiments.analysis import analyze
            summary = analyze(args.output)
        else:
            if args.config is None:
                raise ValueError('--config is required')
            plan = prepare(configuration(args.config, args.input), args.output)
            if not (args.execute or args.simulate):
                print(json.dumps(dict(status='dry_run', samples=len(plan['samples']),
                    planned_calls=plan['planned_calls'], plan_sha256=plan['plan_sha256']), ensure_ascii=False))
                return 0
            from experiments.runner import execute
            fixtures = json.loads(args.fixtures.read_text(encoding='utf-8-sig')) if args.fixtures else None
            summary = execute(plan, args.output, 'live' if args.execute else 'simulate', fixtures,
                              args.max_samples, args.retry_unknown)
        print(json.dumps({key: summary[key] for key in ('status', 'mode', 'population', 'completed', 'pending')}, ensure_ascii=False))
        return 1 if summary['status'] == 'complete_with_errors' else 0
    except (ValueError, RuntimeError, KeyError, OSError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
