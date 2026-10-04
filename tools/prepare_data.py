"""Create a grouped five-event PHEME release from the official source archive."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Aegis'))
from utils.freeze_pheme import build, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = build(None, args.archive.resolve(), args.output.resolve(), args.seed)
    print(json.dumps(dict(summary=result, verification=verify(args.output)), indent=2))


if __name__ == '__main__':
    main()
