"""Recover PHEME source identities and freeze a globally group-disjoint release.

Standard library only. Never joins on historical rounded IDs or chooses a match
using its label. Rebuilds refuse existing output directories. See --help.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import tarfile
import unicodedata

EVENTS = ('charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege')
STEMS = {e: e.replace('-crash', '') for e in EVENTS}
DIMS = ('emotion', 'exaggeration', 'bias', 'commonsense')
SPLITS = ('train', 'val', 'test')
RATIOS = (.8, .1, .1)
SCHEMA = 'pheme_rumour_v1'
OFFICIAL_MD5 = '07f004eca867447d69c71e87be242d75'
NEAR_THRESHOLD = .85


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', html.unescape(text or '')).split())


def compact(text):
    return ''.join(normalize(text).split())


def skeleton(text):
    text = re.sub(r'https?://\S+', ' ', normalize(text).casefold())
    text = re.sub(r'(?<!\w)@\w+', ' ', text)
    text = re.sub(r'^\s*rt\b\s*:?', '', text)
    return ' '.join(re.findall(r'\w+', text, flags=re.UNICODE))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def load_official(archive):
    if hashlib.md5(archive.read_bytes()).hexdigest() != OFFICIAL_MD5:
        raise ValueError('Official archive MD5 does not match the previously verified release')
    rows = []
    with tarfile.open(archive) as tar:
        for member in tar:
            p = member.name.split('/')
            if (len(p) != 6 or p[1] not in EVENTS or p[2] not in ('rumours', 'non-rumours')
                    or p[4] != 'source-tweet' or not p[5].endswith('.json')):
                continue
            obj = json.load(tar.extractfile(member))
            sid = obj['id_str']
            if not isinstance(sid, str) or not sid.isdecimal() or p[3] != sid or p[5] != sid + '.json':
                raise ValueError('Source path / string ID mismatch: ' + member.name)
            text = obj['text']
            if not normalize(text):
                raise ValueError('Empty official source text: ' + sid)
            rows.append(dict(sample_id='pheme:' + sid, id=sid, event_name=p[1],
                             thread_id=sid, reply_to_id=obj.get('in_reply_to_status_id_str') or '',
                             label=str(int(p[2] == 'rumours')), title='', text=text, subject='', date='',
                             source_created_at=obj.get('created_at', ''), source_member=member.name,
                             source_json_sha256=digest(json.dumps(obj, sort_keys=True, ensure_ascii=False)),
                             text_sha256=digest(text), normalized_text_sha256=digest(normalize(text))))
    rows.sort(key=lambda r: (r['event_name'], r['id']))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate official source ID')
    if set(r['event_name'] for r in rows) != set(EVENTS):
        raise ValueError('Missing event in official archive')
    return rows


class Matcher:
    def __init__(self, official):
        self.exact = defaultdict(list)
        self.compact = defaultdict(list)
        for row in official:
            self.exact[(row['event_name'], normalize(row['text']))].append(row)
            self.compact[(row['event_name'], compact(row['text']))].append(row)

    def match(self, event, text):
        if not normalize(text):
            return 'empty_text', []
        candidates = self.exact.get((event, normalize(text)), [])
        if candidates:
            return 'normalized_exact', candidates
        candidates = self.compact.get((event, compact(text)), [])
        return ('whitespace_repaired', candidates) if candidates else ('unmatched', [])


def source_files(root):
    base = root / 'idea1_data/rumor'
    for event in EVENTS:
        yield event, 'raw', base / (event + '.csv')
        stem = STEMS[event]
        for suffix, kind in [('.formatted.csv', 'formatted'), ('.formatted.with_pseudo.csv', 'pseudo')]:
            yield event, kind, base / 'prepared' / (stem + suffix)
        for split in SPLITS:
            yield event, split, base / 'prepared' / ('split_' + stem) / (split + '.csv')


def recover_history(root, official):
    matcher = Matcher(official)
    mappings, pseudo, inputs = [], defaultdict(list), []
    for event, kind, path in source_files(root):
        relative = path.relative_to(root).as_posix()
        inputs.append(dict(path=relative, sha256=file_hash(path), bytes=path.stat().st_size))
        for ordinal, row in enumerate(read_csv(path), 1):
            text = row.get('content' if kind == 'raw' else 'text') or ''
            method, candidates = matcher.match(event, text)
            label = row.get('label', '')
            canonical_label = {'0': '0', '0.0': '0', '1': '1', '1.0': '1'}.get(label)
            status = ('recovered' if len(candidates) == 1 else 'ambiguous' if candidates else method)
            if canonical_label is None:
                status = 'invalid_legacy_label'
            elif candidates and any(c['label'] != canonical_label for c in candidates):
                status = 'label_conflict' if len(candidates) == 1 else 'ambiguous_label_conflict'
            recovered = candidates[0] if status == 'recovered' else None
            legacy_id = 'legacy:' + digest(relative + ':' + str(ordinal))
            mapping = dict(legacy_record_id=legacy_id, source_file=relative, source_record=str(ordinal),
                           source_kind=kind, event_name=event, damaged_id=row.get('id') or '',
                           legacy_label=label, legacy_text_sha256=digest(text),
                           fallback_content_id='text:' + digest(normalize(text)), method=method, status=status,
                           sample_id=recovered['sample_id'] if recovered else '',
                           candidate_ids=' '.join(c['id'] for c in candidates), text=text)
            mappings.append(mapping)
            if kind == 'pseudo' and recovered and method == 'normalized_exact':
                pseudo[recovered['sample_id']].append((mapping, row))
    return mappings, pseudo, inputs


def migrate_pseudo(official, pseudo):
    extras = sorted({k for records in pseudo.values() for _, row in records for k in row
                     if k and any(k.startswith(d + '_') for d in DIMS)})
    for row in official:
        row.update({k: '' for k in extras})
        candidates = pseudo.get(row['sample_id'], [])
        row['pseudo_source_records'] = ' '.join(m['legacy_record_id'] for m, _ in candidates)
        row['pseudo_status'] = 'regeneration_required'
        if not candidates:
            continue
        vectors = []
        try:
            for _, old in candidates:
                vec = tuple(float(old[d + '_label']) for d in DIMS)
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in vec):
                    raise ValueError('Invalid pseudo score')
                if not all(old.get(d + '_parse_ok') == '1' for d in DIMS):
                    raise ValueError('Pseudo parsing failed')
                vectors.append(vec)
        except (KeyError, TypeError, ValueError):
            row['pseudo_status'] = 'invalid_legacy_pseudo'
            continue
        if len(set(vectors)) != 1:
            row['pseudo_status'] = 'conflicting_legacy_pseudo'
            continue
        # Deterministic selection only after all equivalent records agree on scores.
        old = candidates[0][1]
        row.update({k: old.get(k, '') for k in extras})
        row['pseudo_status'] = 'reused_identical_normalized_text'
    return extras


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def near_pairs(rows, threshold=NEAR_THRESHOLD):
    """Exact Jaccard search over all character-5-shingle sets, no LSH sampling.

    Inverted lists enumerate every pair sharing a shingle. Length and overlap
    bounds prune only pairs mathematically unable to meet the threshold.
    """
    index = defaultdict(list)
    sizes = []
    for i, row in enumerate(rows):
        value = skeleton(row['text'])
        grams = {value[k:k+5] for k in range(len(value)-4)} if len(value) >= 20 else set()
        n = len(grams)
        counts = Counter()
        for gram in sorted(grams):
            for j in index[gram]:
                if threshold * n <= sizes[j] <= n / threshold:
                    counts[j] += 1
        for j, common in sorted(counts.items()):
            score = common / (n + sizes[j] - common)
            if score >= threshold:
                yield j, i, score
        sizes.append(n)
        for gram in sorted(grams):
            index[gram].append(i)


def group_rows(rows):
    uf, edges = UnionFind(len(rows)), []
    def link(a, b, reason, score=''):
        uf.union(a, b)
        edges.append(dict(left=rows[a]['sample_id'], right=rows[b]['sample_id'], reason=reason, similarity=score))
    for key in ('id', 'thread_id', 'normalized_text_sha256'):
        seen = {}
        for i, row in enumerate(rows):
            value = row[key]
            if value in seen:
                link(seen[value], i, key)
            else:
                seen[value] = i
    by_id = {r['id']: i for i, r in enumerate(rows)}
    for i, row in enumerate(rows):
        if row['reply_to_id'] in by_id:
            link(i, by_id[row['reply_to_id']], 'source_reply_link')
    for a, b, score in near_pairs(rows):
        link(a, b, 'near_char5_jaccard', f'{score:.8f}')
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    for members in groups.values():
        gid = 'group:' + digest('\n'.join(sorted(rows[i]['sample_id'] for i in members)))
        for i in members:
            rows[i]['group_id'] = gid
    return edges


def assign_splits(rows, seed):
    """Greedy group allocation minimizing per-event/per-class squared count error.

    Largest groups first; SHA-256(seed, group_id) breaks ties independently of
    input order or Python RNG version. Ratios are targets, never split a group.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    totals = Counter((r['event_name'], r['label']) for r in rows)
    allocated = {s: Counter() for s in SPLITS}
    order = sorted(groups, key=lambda g: (-len(groups[g]), digest(f'{seed}:{g}')))
    for gid in order:
        delta = Counter((r['event_name'], r['label']) for r in groups[gid])
        def cost(split):
            ratio = RATIOS[SPLITS.index(split)]
            return sum(((allocated[split][key] + count - totals[key]*ratio)**2
                        - (allocated[split][key] - totals[key]*ratio)**2) / totals[key]
                       for key, count in delta.items())
        split = min(SPLITS, key=lambda s: (cost(s), SPLITS.index(s)))
        allocated[split].update(delta)
        for row in groups[gid]:
            row['split'] = split


def check_disjoint(rows, edges):
    by_id = {}
    for row in rows:
        if row['sample_id'] in by_id:
            raise ValueError('Duplicate sample identity: ' + row['sample_id'])
        if row['split'] not in SPLITS or row['label'] not in ('0', '1'):
            raise ValueError('Invalid split or label')
        if row['normalized_text_sha256'] != digest(normalize(row['text'])):
            raise ValueError('Text hash mismatch')
        by_id[row['sample_id']] = row
    checks = {}
    for key in ('id', 'thread_id', 'text_sha256', 'normalized_text_sha256', 'group_id'):
        locations = defaultdict(set)
        for row in rows:
            locations[row[key]].add(row['split'])
        conflicts = [v for v, splits in locations.items() if len(splits) > 1]
        checks[key] = len(conflicts)
        if conflicts:
            raise ValueError('Cross-split overlap: ' + key)
    for edge in edges:
        a, b = by_id[edge['left']], by_id[edge['right']]
        if a['split'] != b['split'] or a['group_id'] != b['group_id']:
            raise ValueError('Split or group separates a linked pair')
    return checks


def stats(rows):
    return {e: {s: {'rows': sum(r['event_name'] == e and r['split'] == s for r in rows),
                   'labels': dict(sorted(Counter(r['label'] for r in rows if r['event_name'] == e and r['split'] == s).items())),
                   'pseudo_ready': sum(r['event_name'] == e and r['split'] == s and r['pseudo_status'].startswith('reused_') for r in rows)}
                for s in SPLITS} for e in EVENTS}


def build(root, archive, output, seed=42):
    if output.exists():
        raise FileExistsError('Frozen releases are never overwritten; choose a new --output')
    official = load_official(archive)
    mappings, pseudo, inputs = recover_history(root, official) if root is not None else ([], {}, [])
    extras = migrate_pseudo(official, pseudo)
    edges = group_rows(official)
    assign_splits(official, seed)
    overlaps = check_disjoint(official, edges)
    by_id = {r['sample_id']: r for r in official}
    raw_matches = Counter(m['sample_id'] for m in mappings if m['source_kind'] == 'raw' and m['sample_id'])
    for m in mappings:
        r = by_id.get(m['sample_id'])
        m['new_split'] = r['split'] if r else ''
        m['new_group_id'] = r['group_id'] if r else ''
    for r in official:
        r['legacy_raw_match_count'] = str(raw_matches[r['sample_id']])
    output.mkdir(parents=True)
    fields = list(official[0])
    write_csv(output / 'canonical.csv', official, fields)
    manifest_fields = ['sample_id', 'id', 'thread_id', 'event_name', 'label', 'split', 'group_id',
                       'text_sha256', 'normalized_text_sha256', 'source_member', 'source_json_sha256',
                       'legacy_raw_match_count', 'pseudo_status', 'pseudo_source_records']
    write_csv(output / 'split_manifest.csv', [{k: r[k] for k in manifest_fields} for r in official], manifest_fields)
    for event in EVENTS:
        for split in SPLITS:
            write_csv(output / ('split_' + STEMS[event]) / (split + '.csv'),
                      [r for r in official if r['event_name'] == event and r['split'] == split], fields)
    mapping_fields = list(mappings[0]) if mappings else ['legacy_record_id', 'source_file', 'source_record',
        'source_kind', 'event_name', 'damaged_id', 'legacy_label', 'legacy_text_sha256',
        'fallback_content_id', 'method', 'status', 'sample_id', 'candidate_ids', 'text',
        'new_split', 'new_group_id']
    write_csv(output / 'legacy_mapping.csv', mappings, mapping_fields)
    write_csv(output / 'quarantine.csv', [m for m in mappings if not m['sample_id']], mapping_fields)
    write_csv(output / 'group_edges.csv', edges, ['left', 'right', 'reason', 'similarity'])
    mixed_groups = {g for g, labels in _group_labels(official).items() if len(labels) > 1}
    review_fields = ['sample_id', 'event_name', 'label', 'split', 'group_id', 'text', 'source_member']
    write_csv(output / 'mixed_label_group_review.csv',
              [{k: r[k] for k in review_fields} for r in official if r['group_id'] in mixed_groups], review_fields)
    write_csv(output / 'pseudo_regeneration_queue.csv',
              [{k: r[k] for k in ['sample_id', 'event_name', 'split', 'text', 'pseudo_status']} for r in official
               if not r['pseudo_status'].startswith('reused_')], ['sample_id', 'event_name', 'split', 'text', 'pseudo_status'])
    summary = dict(total_rows=len(official), groups=len({r['group_id'] for r in official}),
                   largest_group=max(Counter(r['group_id'] for r in official).values()),
                   split_counts=dict(Counter(r['split'] for r in official)), events=stats(official),
                   legacy_statuses={kind: dict(Counter(m['status'] for m in mappings if m['source_kind'] == kind))
                                    for kind in ('raw', 'formatted', 'pseudo', *SPLITS)},
                   raw_record_count=sum(m['source_kind'] == 'raw' for m in mappings),
                   legacy_record_count=len(mappings),
                   raw_matching_methods=dict(Counter(m['method'] for m in mappings if m['source_kind'] == 'raw' and m['sample_id'])),
                   official_without_unique_raw_match=sum(not raw_matches[r['sample_id']] for r in official),
                   pseudo_statuses=dict(Counter(r['pseudo_status'] for r in official)),
                   group_edge_counts=dict(Counter(e['reason'] for e in edges)), cross_split_overlaps=overlaps,
                   mixed_label_groups=sum(len(v) > 1 for v in _group_labels(official).values()),
                   training_ready_with_full_dimension_supervision=all(r['pseudo_status'].startswith('reused_')
                                                                      for r in official if r['split'] == 'train'))
    write_json(output / 'audit_summary.json', summary)
    write_json(output / 'label_schema.json', dict(schema_id=SCHEMA, source_labels={'0': 'non_rumour', '1': 'rumour'},
                                                positive_label='rumour', probability_field='p_rumour'))
    (output / 'freeze_pheme.py').write_bytes(Path(__file__).read_bytes())
    archive_relative = archive.relative_to(root).as_posix() if root is not None else str(archive.resolve())
    inputs.append(dict(path=archive_relative, sha256=file_hash(archive), bytes=archive.stat().st_size))
    protocol = dict(version='pheme_identity_grouped_v1', seed=seed, ratios=dict(zip(SPLITS, RATIOS)),
                    scope='Five-event in-event evaluation; global groups shared across event files; not leave-one-event-out',
                    source='https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619',
                    official_archive_md5=OFFICIAL_MD5, canonical_population='All official source tweets in the five events',
                    normalization='NFC + HTML unescape + whitespace collapse; case retained',
                    identity_recovery='Unique event-local exact normalized or whitespace-removed text, never rounded ID or label selection',
                    unresolved='Legacy content hash + source file/record index; excluded from canonical population; no guessed tweet ID',
                    near_duplicate=dict(method='Exhaustive character 5-shingle set Jaccard', threshold=NEAR_THRESHOLD,
                                        minimum_skeleton_characters=20, skeleton='casefold; remove URLs, mentions, leading RT; retain Unicode word tokens'),
                    grouping='Transitive union of IDs, thread roots, normalized text, source reply links and near-duplicate edges; global across events',
                    thread_scope='One official source tweet per thread; reactions excluded; only explicit available reply links used',
                    split_algorithm='Largest groups first; seeded SHA-256 tie order; minimum squared per-event/class target count error',
                    pseudo_policy='Reuse only uniquely identified normalized-exact text and valid agreeing four-score vectors; otherwise blank and queued',
                    limitations=['Near-duplicate check is lexical, not a semantic paraphrase guarantee.',
                                 'Underlying real-world claims may span different thread roots.',
                                 'Legacy pseudo annotations are not newly human-validated; missing labels require regeneration before full-supervision training.',
                                 'Historical metrics do not apply to this population or these splits.'],
                    inputs=inputs)
    write_json(output / 'protocol.json', protocol)
    (output / 'README.md').write_text(
        '# PHEME identity/grouped v1\n\n'
        'Frozen five-event source-tweet population. Labels: 0=non_rumour, 1=rumour. '
        'Historical experiment metrics do not describe this release.\n\n'
        f'{len(official)} unique official string IDs; seed {seed}; split counts: {summary["split_counts"]}.\n\n'
        '- `canonical.csv`: official text and source lineage; no rounded IDs.\n'
        '- `split_*/train.csv`, `val.csv`, `test.csv`: grouped event-internal evaluation.\n'
        '- `split_manifest.csv`: stable membership; `protocol.json`: rules and input hashes.\n'
        '- `legacy_mapping.csv`: all original raw/formatted/pseudo/split records; '
        '`source_record` is a 1-based CSV data-record ordinal, not a physical line number.\n'
        '- `quarantine.csv`: unresolved/ambiguous/invalid old records, never guessed into the new data.\n'
        '- `group_edges.csv`: exact and lexical near-duplicate links; transitive groups stay together.\n'
        '- `mixed_label_group_review.csv`: retain official labels in mixed-label groups, flag for review.\n'
        '- `pseudo_regeneration_queue.csv`: missing/stale four-dimension supervision. '
        'Do not enable full dimension supervision until these training rows are relabeled in a new version.\n\n'
        'Verify: `python freeze_pheme.py verify --output .`.\n\n'
        'The SHA-256 seal detects edits; it is not an OS write lock or digital signature. '
        'Build refuses an existing output directory. Create a new version for changes. '
        'Keep the seal in the execution report. Lexical checks cannot certify absence of all semantic paraphrases.\n',
        encoding='utf-8')
    # The manifest commits every artifact; its seal is kept separately for easy copying.
    hashes = {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'freeze_manifest.json', dict(version=protocol['version'], files=hashes))
    (output / 'FREEZE.sha256').write_text(file_hash(output / 'freeze_manifest.json') + '  freeze_manifest.json\n', encoding='ascii')
    return summary


def _group_labels(rows):
    labels = defaultdict(set)
    for row in rows:
        labels[row['group_id']].add(row['label'])
    return labels


def verify(output, root=None, recompute_near=True):
    manifest_path = output / 'freeze_manifest.json'
    if (output / 'FREEZE.sha256').read_text(encoding='ascii').split()[0] != file_hash(manifest_path):
        raise ValueError('Freeze manifest seal mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    observed = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if observed != set(manifest['files']) | {'freeze_manifest.json', 'FREEZE.sha256'}:
        raise ValueError('Unexpected or missing frozen artifacts')
    for name, expected in manifest['files'].items():
        if file_hash(output / name) != expected:
            raise ValueError('Frozen artifact hash mismatch: ' + name)
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    if root is not None:
        for src in protocol['inputs']:
            if file_hash(root / src['path']) != src['sha256']:
                raise ValueError('Historical input changed: ' + src['path'])
    rows = read_csv(output / 'canonical.csv')
    split_rows = []
    for event in EVENTS:
        for split in SPLITS:
            part = read_csv(output / ('split_' + STEMS[event]) / (split + '.csv'))
            if not part or any(r['split'] != split or r['event_name'] != event for r in part):
                raise ValueError('Empty or misfiled split')
            split_rows.extend(part)
    if sorted(split_rows, key=lambda r: r['id']) != sorted(rows, key=lambda r: r['id']):
        raise ValueError('Split population differs from canonical population')
    split_manifest = read_csv(output / 'split_manifest.csv')
    if len(split_manifest) != len(rows) or any(any(r.get(k) != v for k, v in m.items()) for r, m in zip(rows, split_manifest)):
        raise ValueError('Split manifest differs from canonical population')
    edges = read_csv(output / 'group_edges.csv')
    overlaps = check_disjoint(rows, edges)
    if recompute_near:
        rebuilt = [dict(r) for r in rows]
        regenerated_edges = group_rows(rebuilt)
        if regenerated_edges != edges or any(a['group_id'] != b['group_id'] for a, b in zip(rows, rebuilt)):
            raise ValueError('Recomputed duplicate groups differ')
        assign_splits(rebuilt, protocol['seed'])
        if any(a['split'] != b['split'] for a, b in zip(rows, rebuilt)):
            raise ValueError('Recomputed seeded splits differ')
    return dict(status='passed', rows=len(rows), artifact_count=len(manifest['files']),
                input_hashes_checked=root is not None, near_groups_recomputed=recompute_near,
                cross_split_overlaps=overlaps, freeze_sha256=file_hash(manifest_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    build_parser = sub.add_parser('build')
    build_parser.add_argument('--root', type=Path, help='Optional workspace of legacy CSVs to migrate')
    build_parser.add_argument('--archive', type=Path, required=True)
    build_parser.add_argument('--output', type=Path, required=True, help='New, nonexistent release directory')
    build_parser.add_argument('--seed', type=int, default=42)
    verify_parser = sub.add_parser('verify')
    verify_parser.add_argument('--output', type=Path, required=True)
    verify_parser.add_argument('--root', type=Path, help='Also verify unchanged original inputs')
    args = ap.parse_args()
    if args.command == 'build':
        result = build(args.root.resolve() if args.root else None, args.archive.resolve(), args.output.resolve(), args.seed)
    else:
        result = verify(args.output.resolve(), args.root.resolve() if args.root else None)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
