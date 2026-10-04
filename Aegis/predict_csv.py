"""Export an existing checkpoint on a labeled CSV; no training or network needed."""
import argparse
import csv
import json
from pathlib import Path

from inference.predictor import Predictor
from utils.checkpoints import checkpoint_config, checkpoint_paths
from utils.predictions import sample_identity, prediction_rows, write_predictions
from utils.run_archive import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True, help='New CSV path')
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--event', help='Required for legacy configs lacking experiment.event')
    parser.add_argument('--device')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    cfg = checkpoint_config(args.checkpoint_dir)
    if args.event:
        if cfg['experiment'].get('event') not in (None, args.event):
            raise ValueError('Event override conflicts with checkpoint experiment')
        cfg['experiment']['event'] = args.event
    if args.device:
        cfg['hardware']['device_map'] = dict.fromkeys(cfg['hardware']['device_map'], args.device)
    model = Predictor(cfg, checkpoint_paths(args.checkpoint_dir))
    digest = sha256(args.input)
    identities, targets, probabilities, details = [], [], [], []
    with Path(args.input).open(encoding='utf-8-sig', newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            sample_id, source = sample_identity(row, index, digest)
            identities.append(dict(sample_id=sample_id, identity_source=source,
                                   event=row.get('event_name') or row.get('event') or ''))
            targets.append(model.label_schema.encode_source(row['label']))
            result=model.predict(row)
            probabilities.append(result['p_positive'])
            details.append({k:v for k,v in result.items() if k.startswith(('pre_','post_')) or k in ('team_p_positive','gate_mean')})
    manifest = Path(args.checkpoint_dir)/'checkpoint_manifest.json'
    checkpoint_id = 'sha256:' + sha256(manifest)
    rows = prediction_rows(identities, targets, probabilities, model.label_schema, cfg, args.split, checkpoint_id)
    for row,detail in zip(rows,details):row.update(detail)
    result = write_predictions(args.output, rows, manifest, args.input)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
