"""Deterministic leave-one-event-out membership, with global group exclusion.

Source train members train; source val+test members select the model; every
held-out-event member tests. No teacher request or model execution occurs here.
"""
import json
import shutil
from collections import Counter
from pathlib import Path

from utils import freeze_pheme as f
from utils.pseudo_quality import load_source
from utils.pseudo_policy import training_release_ready
from utils.experiment_protocol import CROSS_EVENT


def fold_rows(rows, event):
    if event not in f.EVENTS or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid event or duplicate canonical identity')
    held = [r for r in rows if r['event_name'] == event]
    blocked = {r['group_id'] for r in held}
    parts = {s: [] for s in f.SPLITS}
    excluded = []
    for row in rows:
        if row['event_name'] != event and row['group_id'] in blocked:
            excluded.append(dict(sample_id=row['sample_id'], event_name=row['event_name'],
                                 group_id=row['group_id'], reason='group_touches_held_out_event'))
            continue
        split = 'test' if row['event_name'] == event else ('train' if row['split'] == 'train' else 'val')
        value = dict(row, parent_split=row['split'], split=split)
        # Keep reference scores in parent_canonical.csv only; no evaluation supervision.
        if split != 'train':
            value = {k:v for k,v in value.items() if not k.startswith(tuple(d+'_' for d in f.DIMS) + ('pseudo_',))}
        parts[split].append(value)
    for split, part in parts.items():
        if not part:
            raise ValueError('Empty cross-event split: ' + split)
        expected = {event} if split == 'test' else set(f.EVENTS)-{event}
        if {r['event_name'] for r in part} != expected:
            raise ValueError('Missing or leaked event: ' + split)
    for key in ('sample_id', 'group_id', 'normalized_text_sha256'):
        for a,b in (('train','val'),('train','test'),('val','test')):
            if {r[key] for r in parts[a]} & {r[key] for r in parts[b]}:
                raise ValueError('Cross-event leakage: ' + key)
    return parts, excluded


def build(source, output):
    source, output = Path(source), Path(output)
    rows, check = load_source(source)
    parent = json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    folds = {event: fold_rows(rows,event) for event in f.EVENTS}
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source/'canonical.csv',output/'parent_canonical.csv')
    shutil.copyfile(source/'freeze_manifest.json',output/'parent_freeze_manifest.json')
    summaries = {}
    for event,(parts,excluded) in folds.items():
        for split,part in parts.items():
            f.write_csv(output/'folds'/event/(split+'.csv'),part,list(part[0]))
        f.write_csv(output/'folds'/event/'excluded.csv',excluded,
                    ['sample_id','event_name','group_id','reason'])
        summaries[event] = dict(counts={s:len(p) for s,p in parts.items()}, excluded=len(excluded),
            labels={s:dict(Counter(r['label'] for r in p)) for s,p in parts.items()},
            training_ready=training_release_ready(parts['train']))
    protocol = dict(version=CROSS_EVENT, evaluation_scope=CROSS_EVENT,
        parent_manifest_sha256=check['freeze_sha256'], source_split_seed=parent['seed'],
        train='Other events: inherited parent train membership, minus groups touching held-out event.',
        val='Other events: inherited parent val and test combined, minus held-out groups.',
        test='All canonical members of the held-out event; never used for model or hyperparameter selection.',
        grouping='Inherit global identity/thread/exact/near-duplicate transitive groups; remove source-side held-out groups.',
        teacher_boundary='Only source-train scores enter optimization. Val/test files omit all scores and rationales. '
                         'parent_canonical.csv is offline post-training reference only, with missing scores retained.',
        pseudo_release_policy=parent.get('pseudo_release_policy','unreleased'),
        human_review_completed=parent.get('human_review_completed',False),
        selection='Fixed threshold 0.5; select checkpoints on source validation only. '
                  'Freeze hyperparameters across folds before viewing any held-out results.',
        limitations=['Teacher consistency is not independent factual quality.',
                     'Teachers may have pretraining knowledge of held-out events; this is no guarantee of temporal isolation.',
                     'This uses inherited source membership, not a new random 90/10 split.',
                     'Validation populations overlap across folds; never pool them as independent observations.',
                     'Lexical grouping cannot exclude all semantic paraphrases.'])
    f.write_json(output/'protocol.json',protocol)
    f.write_json(output/'fold_summary.json',summaries)
    f.write_json(output/'freeze_manifest.json',dict(version=CROSS_EVENT,files={
        p.relative_to(output).as_posix():f.file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    (output/'FREEZE.sha256').write_text(f.file_hash(output/'freeze_manifest.json')+'  freeze_manifest.json\n',encoding='ascii')
    return verify(output)


def verify(output):
    output=Path(output)
    manifest=json.loads((output/'freeze_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != CROSS_EVENT:
        raise ValueError('Not a cross-event freeze')
    if (output/'FREEZE.sha256').read_text().split()[0] != f.file_hash(output/'freeze_manifest.json'):
        raise ValueError('Cross-event seal mismatch')
    observed={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if observed != set(manifest['files'])|{'freeze_manifest.json','FREEZE.sha256'}:
        raise ValueError('Missing or unexpected cross-event artifacts')
    for name,digest in manifest['files'].items():
        path=(output/name).resolve()
        if not path.is_relative_to(output.resolve()) or f.file_hash(path)!=digest:
            raise ValueError('Cross-event artifact hash mismatch')
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    parent=json.loads((output/'parent_freeze_manifest.json').read_text(encoding='utf-8'))
    if (f.file_hash(output/'parent_freeze_manifest.json')!=protocol['parent_manifest_sha256'] or
            f.file_hash(output/'parent_canonical.csv')!=parent['files']['canonical.csv']):
        raise ValueError('Parent canonical provenance mismatch')
    rows=f.read_csv(output/'parent_canonical.csv')
    for event in f.EVENTS:
        parts,excluded=fold_rows(rows,event)
        for split,part in parts.items():
            if f.read_csv(output/'folds'/event/(split+'.csv'))!=part:
                raise ValueError('Cross-event membership differs from deterministic protocol')
        if f.read_csv(output/'folds'/event/'excluded.csv')!=excluded:
            raise ValueError('Cross-event exclusions differ')
    return dict(status='passed',folds=len(f.EVENTS),rows=len(rows),artifact_count=len(manifest['files']),
                freeze_sha256=f.file_hash(output/'freeze_manifest.json'),
                parent_manifest_sha256=protocol['parent_manifest_sha256'])
