"""Recompute a single variant/seed/split from canonical prediction CSV files."""
import argparse
import csv
import json
from pathlib import Path

from training.metrics import PROTOCOL, SCORES, event_metrics
from utils.label_schema import get_label_schema
from utils.run_archive import sha256, write_json
from utils.experiment_protocol import IN_EVENT, CROSS_EVENT

REQUIRED = {"sample_id", "event", "target", "p_positive", "predicted_target", "label_schema",
            "variant", "seed", "split", "checkpoint_id"}


def evaluate_files(paths):
    rows, seen, inputs = [], set(), []
    for path in paths:
        manifest_path = Path(path).with_suffix('.manifest.json')
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if sha256(path) != manifest['sha256']:
                raise ValueError('Prediction file hash mismatch')
        else:
            manifest = None
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not REQUIRED.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing prediction columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
            for row in reader:
                if any(not row.get(key, "").strip() for key in REQUIRED):
                    raise ValueError("Missing prediction metadata/value")
                if row["sample_id"] in seen:
                    raise ValueError("Duplicate sample_id; never pool repeated seeds/runs")
                seen.add(row["sample_id"])
                if manifest and row['checkpoint_id'] != 'sha256:' + manifest['checkpoint_manifest_sha256']:
                    raise ValueError('Prediction checkpoint differs from export manifest')
                schema = get_label_schema(row["label_schema"])
                if row["target"] not in ("0", "1") or row["predicted_target"] not in ("0", "1"):
                    raise ValueError("target/predicted_target must be canonical 0/1, not source labels")
                p = float(row["p_positive"])
                expected = schema.prediction(p)
                if expected["predicted_target"] != int(row["predicted_target"]):
                    raise ValueError("Prediction does not use the fixed protocol threshold 0.5")
                if row.get(schema.probability_field) and float(row[schema.probability_field]) != p:
                    raise ValueError("Named positive probability disagrees with p_positive")
                rows.append(row)
        inputs.append({"path": str(Path(path).resolve()), "sha256": sha256(path)})
    if not rows:
        raise ValueError("Cannot evaluate empty prediction files")
    for row in rows:
        row.setdefault('evaluation_scope',IN_EVENT)
    for key in ("label_schema", "variant", "seed", "split", "evaluation_scope"):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f"Cannot mix {key} values")
    if rows[0]['evaluation_scope'] == CROSS_EVENT and rows[0]['split'] == 'val' and len({r.get('fold') for r in rows})>1:
        raise ValueError('Never pool cross-fold validation populations')
    for event in {r["event"] for r in rows}:
        if len({r["checkpoint_id"] for r in rows if r["event"] == event}) != 1:
            raise ValueError("Cannot mix checkpoints within one event")
    result = event_metrics([r["event"] for r in rows], [int(r["target"]) for r in rows],
                           [float(r["p_positive"]) for r in rows])
    result.update(protocol=PROTOCOL, label_semantics=get_label_schema(rows[0]["label_schema"]).metadata(),
                  variant=rows[0]["variant"], seed=rows[0]["seed"], split=rows[0]["split"], inputs=inputs,
                  evaluation_scope=rows[0]['evaluation_scope'])
    return result


def write_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", result)
    fields = ["aggregation", "event", "n", *SCORES]
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for event, metrics in result["per_event"].items():
            writer.writerow(dict(metrics, aggregation="per_event", event=event))
        for name in ("event_equal_mean", "sample_pooled"):
            writer.writerow(dict(result[name], aggregation=name, event="ALL"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--output", required=True, help="New report directory; existing directories are refused")
    args = parser.parse_args()
    write_report(evaluate_files(args.predictions), args.output)


if __name__ == "__main__":
    main()
