"""Explicit dataset and checkpoint label semantics (standard library only).

Model target 1 always means the named positive class. Rumour detection and
fake-news classification are different tasks, never interchangeable aliases.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class LabelSchema:
    name: str
    task: str
    source_zero: str
    source_one: str
    negative: str
    positive: str
    probability_field: str

    def source_code(self, value) -> int:
        """Read an original dataset label, without changing its coding."""
        token = str(value).strip().lower()
        aliases = {"non-rumours": "non_rumour", "non-rumour": "non_rumour",
                   "nonrumor": "non_rumour", "non-rumor": "non_rumour",
                   "non_rumor": "non_rumour", "rumours": "rumour", "rumor": "rumour"}
        token = aliases.get(token, token)
        if token in {"0", "0.0", self.source_zero}:
            return 0
        if token in {"1", "1.0", self.source_one}:
            return 1
        raise ValueError(f"Invalid label {value!r} for {self.name}; expected 0/1 or "
                         f"{self.source_zero}/{self.source_one}. No cross-task aliases are allowed.")

    def encode_source(self, value) -> int:
        """Map the source label to the model target used by BCE and metrics."""
        source_class = (self.source_zero, self.source_one)[self.source_code(value)]
        return int(source_class == self.positive)

    def decode_target(self, value: int) -> str:
        if value not in (0, 1):
            raise ValueError(f"Invalid model target: {value!r}")
        return (self.negative, self.positive)[int(value)]

    def prediction(self, p_positive: float, threshold: float = 0.5) -> Dict:
        p, threshold = float(p_positive), float(threshold)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p_positive must be a finite value in [0, 1]")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite value in [0, 1]")
        target = int(p >= threshold)
        return {"task": self.task, "label_schema": self.name,
                "positive_label": self.positive, "negative_label": self.negative,
                "final_label": self.decode_target(target), "predicted_target": target,
                "p_positive": p, self.probability_field: p,
                "final_confidence": p if target else 1 - p}

    def metadata(self) -> Dict:
        return {"schema_version": 1, "label_schema": self.name, "task": self.task,
                "source_label_map": {"0": self.source_zero, "1": self.source_one},
                "model_label_map": {"0": self.negative, "1": self.positive},
                "source_to_target": {"0": self.encode_source(0), "1": self.encode_source(1)},
                "positive_label": self.positive, "probability_field": self.probability_field,
                "threshold": 0.5}


SCHEMAS = {
    "pheme_rumour_v1": LabelSchema("pheme_rumour_v1", "rumour_detection", "non_rumour", "rumour",
                                  "non_rumour", "rumour", "p_rumour"),
    "archive_fake_v1": LabelSchema("archive_fake_v1", "fake_news_classification", "fake", "real",
                                  "real", "fake", "p_fake"),
}


def get_label_schema(name: str) -> LabelSchema:
    if name not in SCHEMAS:
        raise ValueError(f"Explicit data.label_schema required; choose one of {sorted(SCHEMAS)}, got {name!r}")
    return SCHEMAS[name]


def schema_from_config(config: Dict) -> LabelSchema:
    return get_label_schema((config.get("data") or {}).get("label_schema"))


def require_schema_metadata(path, schema: LabelSchema) -> None:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing checkpoint label metadata: {path}. Legacy weights must not be "
                         "silently interpreted using a new label map; retrain or explicitly migrate them.")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != schema.metadata():
        raise ValueError(f"Label semantics mismatch in {path}; expected {schema.name}. "
                         "Use the matching data/checkpoint or explicitly migrate legacy weights.")


def write_schema_metadata(path, schema: LabelSchema) -> None:
    path = Path(path)
    if path.exists():
        require_schema_metadata(path, schema)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema.metadata(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
