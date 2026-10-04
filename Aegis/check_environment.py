"""Minimal dependency and computation check; never load pretrained weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re


def check(profile="audit"):
    root = Path(__file__).resolve().parent
    errors = []
    if platform.python_version() != "3.12.14":
        errors.append("Expected CPython 3.12.14")
    if profile == "training" and (platform.system() != "Linux" or platform.machine() != "x86_64"):
        errors.append("Training lock targets Linux x86_64")
    name = "requirements-audit.txt" if profile == "audit" else "requirements-train-linux.lock"
    packages = {}
    for line in (root / "environment" / name).read_text(encoding="utf-8").splitlines():
        match = re.match(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
        if not match:
            continue
        package, expected = match.groups()
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[package] = actual
        if actual != expected:
            errors.append(f"{package}: expected {expected}, found {actual}")
    if errors:
        raise RuntimeError("Environment preflight failed: " + "; ".join(errors))
    from training.metrics import binary_metrics
    assert binary_metrics([0, 1], [.1, .9])["roc_auc"] == 1
    if profile == "training":
        import torch
        import transformers
        import peft
        import bitsandbytes
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for the pinned training profile")
        x = torch.tensor([2.0], device="cuda", requires_grad=True)
        (x * x).sum().backward()
        assert x.grad.item() == 4
    return {"profile": profile, "python": platform.python_version(), "packages": packages,
            "status": "passed", "pretrained_model_loaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["audit", "training"], default="audit")
    args = parser.parse_args()
    print(json.dumps(check(args.profile), indent=2))
