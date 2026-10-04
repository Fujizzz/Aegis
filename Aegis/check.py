# check.py
# 规范化：去掉硬编码路径，支持命令行检查 train/val 是否有重复样本（基于 title+text 默认签名）
# 用法：
#   python check.py --train train.csv --val val.csv
#   python check.py --train train.csv --val val.csv --cols title text subject date

import argparse
import hashlib
import pandas as pd
from typing import List
from pathlib import Path
from utils.freeze_pheme import verify, normalize


def make_signature(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV for signature: {c}")
    s = None
    for c in cols:
        part = df[c].fillna("").astype(str).str.strip().str.lower()
        s = part if s is None else (s + "||" + part)
    return s.apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-dir", type=Path, help="Verify the sealed release, all three splits and duplicate groups")
    ap.add_argument("--root", type=Path, help="Also verify historical input hashes for a frozen release")
    ap.add_argument("--train", type=str, help="Train CSV path")
    ap.add_argument("--val", type=str, help="Val CSV path")
    ap.add_argument("--test", type=str, help="Test CSV path")
    ap.add_argument("--cols", type=str, nargs="+", default=["title", "text"], help="Columns used for duplicate signature")
    ap.add_argument("--show", type=int, default=0, help="Print first N duplicate signatures (0=off)")
    args = ap.parse_args()

    if args.frozen_dir:
        import json
        print(json.dumps(verify(args.frozen_dir, args.root), indent=2))
        return
    if not args.train or not args.val:
        ap.error("Use --frozen-dir or both --train and --val (plus --test when available)")

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)

    sig_tr = set(make_signature(train, args.cols))
    sig_va = set(make_signature(val, args.cols))
    inter = list(sig_tr & sig_va)

    print(f"[check.py] Train rows = {len(train)} | Val rows = {len(val)}")
    print(f"[check.py] Duplicate count (intersection) = {len(inter)}")
    if args.show > 0 and len(inter) > 0:
        print("[check.py] First duplicates:")
        for x in inter[:args.show]:
            print("  ", x)

    # Check text independently of optional metadata and include all split pairs.
    frames = {'train': train, 'val': val}
    if args.test:
        frames['test'] = pd.read_csv(args.test, dtype=str, keep_default_na=False)
    failures = bool(inter)
    from itertools import combinations
    for left, right in combinations(frames, 2):
        a, b = frames[left], frames[right]
        columns = ['text'] + [k for k in ('sample_id', 'thread_id', 'group_id') if k in a and k in b]
        for key in columns:
            if key not in a or key not in b:
                raise ValueError('Missing column: ' + key)
            av = {normalize(str(v)) for v in a[key].fillna('') if str(v).strip()}
            bv = {normalize(str(v)) for v in b[key].fillna('') if str(v).strip()}
            overlap = av & bv
            print(f'[check.py] {left}/{right} {key}: {len(overlap)} overlaps')
            failures |= bool(overlap)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
