"""Publish a frozen dataset with original recorded teacher scores."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Aegis'))
from utils.pseudo_quality import build_release
from utils.freeze_pheme import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    decision = dict(schema='teacher_only_release_decision_v1', release_policy='teacher_only',
        user_instruction='Publish original teacher scores as dimension supervision.',
        audit_sha256=file_hash(args.audit/'audit_manifest.json'))
    path = args.audit.parent/(args.audit.name+'.teacher-decision.json')
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != decision:
            raise ValueError('Existing release decision does not match this audit')
    else:
        path.write_text(json.dumps(decision, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(build_release(args.source, args.audit, None, args.output,
        release_policy='teacher_only', decision_record=path), indent=2))


if __name__ == '__main__':
    main()
