"""Lossless canonical prediction CSVs and explicit sample identities."""
import csv
from pathlib import Path
from utils.run_archive import sha256, write_json
from utils.experiment_protocol import allowed_events, scope


def sample_identity(row, index, input_sha256):
    sample_id = str(row.get('sample_id', '')).strip()
    if sample_id:
        return sample_id, 'source_sample_id'
    # Legacy numeric IDs may already be corrupted. Do not pretend to recover them.
    return f'csv:{input_sha256}:row:{index + 1}', 'csv_sha256_row_v1'


def prediction_rows(identities, targets, probabilities, schema, config, split, checkpoint_id):
    if not (len(identities) == len(targets) == len(probabilities)) or not identities:
        raise ValueError('Prediction metadata must align with nonempty targets/probabilities')
    experiment, runtime = config['experiment'], config['runtime']
    event = str(experiment.get('event', '')).strip()
    if not event:
        raise ValueError('experiment.event is required for prediction export')
    rows, seen = [], set()
    for identity, target, p in zip(identities, targets, probabilities):
        sample_id = identity['sample_id']
        if not sample_id or sample_id in seen:
            raise ValueError('Empty or duplicate sample_id in evaluation')
        seen.add(sample_id)
        actual_event = identity.get('event') or event
        if actual_event not in allowed_events(config,split):
            raise ValueError('Sample event differs from experiment.event')
        if target not in (0, 1):
            raise ValueError('Target must be canonical 0/1')
        pred = schema.prediction(float(p))
        rows.append(dict(sample_id=sample_id, event=actual_event, target=int(target),
                         **pred, variant=experiment['variant'], seed=runtime['seed'], split=split,
                         checkpoint_id=checkpoint_id, run_id=runtime['run_id'],
                         identity_source=identity['identity_source'], evaluation_scope=scope(config), fold=event))
    return rows


def write_predictions(path, rows, checkpoint_manifest, source_data):
    path = Path(path)
    if path.exists() or path.with_suffix('.manifest.json').exists():
        raise FileExistsError('Prediction output already exists')
    if not rows:
        raise ValueError('Cannot write empty predictions')
    checkpoint_sha = sha256(checkpoint_manifest)
    source_sha = sha256(source_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)
    manifest = dict(format='aegis_predictions_v1', rows=len(rows), sha256=sha256(path),
                    checkpoint_manifest=str(Path(checkpoint_manifest).resolve()),
                    checkpoint_manifest_sha256=checkpoint_sha,
                    source_data=str(Path(source_data).resolve()), source_sha256=source_sha)
    write_json(path.with_suffix('.manifest.json'), manifest)
    return manifest
