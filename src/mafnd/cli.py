"""Command-line interface for the two-stage multi-agent detector."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from .detector import MultiAgentDetector
from .models import NewsItem
from .providers import DeepSeekProvider, HeuristicProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Detect one article supplied on the command line")
    source.add_argument("--input", type=Path, help="Detect every row in a CSV file")
    parser.add_argument("--title", default="", help="Title for --text mode")
    parser.add_argument("--provider", choices=("heuristic", "deepseek"), default="heuristic")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    parser.add_argument("--self-consistency", type=int, default=1)
    return parser


def _provider(args: argparse.Namespace):
    if args.provider == "heuristic":
        return HeuristicProvider()
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is required when --provider deepseek is selected")
    return DeepSeekProvider(key, base_url=args.base_url, model=args.model)


def _item(row: dict[str, str], index: int) -> NewsItem:
    content = row.get("content") or row.get("text") or row.get("Text") or ""
    if not content.strip():
        raise ValueError(f"Row {index} has no content/text value")
    return NewsItem(
        identifier=row.get("id") or row.get("ID") or str(index),
        title=row.get("title") or row.get("Title") or "",
        content=content,
        subject=row.get("subject") or row.get("Subject") or "",
        date=row.get("date") or row.get("Date") or "",
    )


def _normalize_label(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "real"}:
        return "real"
    if normalized in {"0", "false", "fake"}:
        return "fake"
    return normalized


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    detector = MultiAgentDetector(_provider(args), self_consistency=args.self_consistency)

    if args.text is not None:
        result = detector.detect(NewsItem(title=args.title, content=args.text))
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.status == 'success' else 1

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    if not input_rows:
        raise ValueError(f"No rows found in {args.input}")

    output_rows = []
    correct = evaluated = 0
    for index, row in enumerate(input_rows, start=1):
        item = _item(row, index)
        result = detector.detect(item)
        output = dict(row)
        output.update(
            {
                "prediction": result.label,
                "fake_probability": result.fake_probability,
                "p_fake": result.p_fake,
                "class_confidence": result.class_confidence,
                "status": result.status,
                "error_type": result.error_type,
                "stage": result.stage,
                "aspect_scores": json.dumps(result.aspect_scores, sort_keys=True),
                "explanation": result.explanation,
            }
        )
        output_rows.append(output)
        actual = row.get("label") or row.get("Label")
        if actual not in (None, ""):
            evaluated += 1
            correct += int(_normalize_label(actual) == result.label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary: dict[str, object] = {"output": str(args.output), "samples": len(output_rows)}
    if evaluated:
        summary["accuracy"] = correct / evaluated
        summary["evaluated"] = evaluated
    summary['failed'] = sum(row['status'] == 'error' for row in output_rows)
    print(json.dumps(summary, indent=2))
    return 1 if summary['failed'] else 0


if __name__ == "__main__":
    raise SystemExit(main())
