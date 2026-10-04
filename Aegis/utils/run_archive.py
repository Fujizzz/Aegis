"""Immutable run inputs and content-addressed local model identity. No torch needed."""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import uuid

from utils.label_schema import schema_from_config, write_schema_metadata
from training.metrics import PROTOCOL
from training.options import grad_accum_steps, resolve_variant
from utils.experiment_protocol import allowed_events, CROSS_EVENT, scope, split_relative

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SECRET_KEY = re.compile(r"(^|_)(api_key|token|secret|password|authorization|credential)($|_)", re.I)
SECRET_VALUE = re.compile(r"sk-[A-Za-z0-9_-]{12,}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def reject_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if SECRET_KEY.search(str(key)) and item not in (None, ""):
                raise ValueError("Credentials must be supplied through environment variables, not configuration")
            reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            reject_secrets(item)
    elif isinstance(value, str) and SECRET_VALUE.search(value):
        raise ValueError("Credential-like literal rejected (value withheld)")


def model_manifest(root):
    root = Path(root).resolve(strict=True)
    if not (root / "config.json").is_file():
        raise ValueError("Use a local Hugging Face model snapshot with config.json")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and
                   ".cache" not in p.relative_to(root).parts and ".git" not in p.relative_to(root).parts)
    if not paths:
        raise ValueError("Empty model snapshot")
    files = {p.relative_to(root).as_posix(): {"size": p.stat().st_size, "sha256": sha256(p)} for p in paths}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"identity_kind": "local_content_sha256", "identity": identity, "files": files,
            "upstream_revision": None}


def environment():
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()})
    gpu = None
    try:
        command = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10)
        if command.returncode == 0:
            gpu = command.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "executable": sys.executable,
            "packages": packages, "nvidia_smi": gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED")}


def component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) or value.endswith("."):
        raise ValueError("Run path components must be simple alphanumeric names")
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("Reserved path component")
    return value


def resolve_config(config, config_path):
    cfg = copy.deepcopy(config)
    reject_secrets(cfg)
    base = Path(config_path).resolve().parent
    for section, names in {"data": ("train_file", "val_file", "test_file", "freeze_manifest"),
                           "model": ("base_model", "manifest"), "runtime": ("runs_root", "save_dir")}.items():
        for name in names:
            raw = cfg.get(section, {}).get(name)
            if raw:
                path = Path(os.path.expandvars(raw)).expanduser()
                if "$" in str(path) or re.search(r"%[A-Za-z_]+%", str(path)):
                    raise ValueError(f"Unresolved environment variable in {section}.{name}")
                cfg[section][name] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
    schema_from_config(cfg)
    rt = cfg.setdefault("runtime", {})
    rt.setdefault("seed", 42)
    rt.setdefault("runs_root", str(base / "runs"))
    if isinstance(rt["seed"], bool) or not isinstance(rt["seed"], int) or not 0 <= rt["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if rt.get("num_workers", 0) != 0:
        raise ValueError("Current reproducibility profile requires num_workers=0")
    select = {"f1": "binary_f1", "auc": "roc_auc"}.get(rt.get("select_metric"), rt.get("select_metric", "binary_f1"))
    if select not in ("binary_f1", "class_macro_f1", "roc_auc", "accuracy"):
        raise ValueError("Unsupported validation selection metric")
    rt["select_metric"] = select
    evaluation = cfg.setdefault("evaluation", {})
    if evaluation.get("threshold", 0.5) != 0.5 or evaluation.get("threshold_policy", "fixed_before_test") != "fixed_before_test":
        raise ValueError("aegis_binary_v1 fixes threshold=0.5 before test; tuning requires a new protocol")
    evaluation.update(threshold=0.5, threshold_policy="fixed_before_test")
    experiment = cfg.get("experiment", {})
    for key in ("dataset", "event", "variant", "method_id"):
        component(experiment[key])
    training = cfg.setdefault("training", {})
    training["grad_accum_steps"] = grad_accum_steps(training)
    return resolve_variant(cfg)


def prepare_run(config, config_path, source_root=SOURCE_ROOT, recovery_contract=None, *, require_model=True):
    cfg = resolve_config(config, config_path)
    rt, exp = cfg["runtime"], cfg["experiment"]
    if require_model:
        model = model_manifest(cfg["model"]["base_model"])
        expected = json.loads(Path(cfg["model"]["manifest"]).read_text(encoding="utf-8"))
        if model != expected:
            raise ValueError("Base model differs from the pinned content manifest")
    else:
        if exp['variant'] not in ('tfidf_nb', 'teacher_direct'):
            raise ValueError('Only non-neural baselines may omit the local model manifest')
        model = dict(identity_kind='no_local_pretrained_model', identity=exp['variant'], files={})
    inputs = {}
    for split in ("train", "val", "test"):
        path = cfg["data"].get(f"{split}_file")
        if not path:
            if split != "test":
                raise ValueError(f"Missing {split} data")
            continue
        inputs[split] = {"source": path, "sha256": sha256(path), "size": Path(path).stat().st_size}
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "event_name" in (reader.fieldnames or []):
                events = {row["event_name"] for row in reader}
                if events != allowed_events(cfg, split):
                    raise ValueError("Configured event does not match input CSV event_name")
    freeze_path = cfg["data"].get("freeze_manifest")
    if freeze_path:
        freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
        if scope(cfg) == CROSS_EVENT:
            if freeze.get('version') != CROSS_EVENT:
                raise ValueError('Cross-event config requires its own frozen protocol')
            for split,item in inputs.items():
                if freeze['files'].get(split_relative(cfg,split)) != item['sha256']:
                    raise ValueError('Cross-event fold data binding mismatch')
        # Recovery archives rename files to train/val/test.csv. Accept that layout
        # only when both the original freeze file and per-split hashes are bound
        # by a previously verified recovery checkpoint.
        archived_recovery = (recovery_contract is not None and
            sha256(freeze_path) == recovery_contract['freeze_sha256'] and
            {key: value['sha256'] for key, value in inputs.items()} == recovery_contract['data'])
        if not archived_recovery:
            for item in inputs.values():
                relative = Path(item["source"]).relative_to(Path(freeze_path).parent).as_posix()
                if freeze["files"].get(relative) != item["sha256"]:
                    raise ValueError("Data differs from the frozen split manifest")
    source_root = Path(source_root)
    candidates = list(source_root.iterdir())
    for name in ("agents", "communication", "inference", "training", "utils", "tests", "environment", "baselines"):
        candidates.extend((source_root / name).rglob("*"))
    sources = sorted(p for p in candidates if p.is_file() and
                     (p.suffix in (".py", ".md", ".yaml", ".json", ".txt", ".in", ".lock") or p.name == ".python-version") and
                     not any(part in ("runs", "__pycache__", ".git", ".venv") for part in p.relative_to(source_root).parts))
    for p in sources:
        if SECRET_VALUE.search(p.read_text(encoding="utf-8-sig")):
            raise ValueError(f"Credential-like literal in source file {p.name}; archive refused")
    run_id = component(rt.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10])
    parent = Path(rt["runs_root"]) / exp["dataset"] / exp["event"] / exp["variant"] / f"seed_{rt['seed']}"
    run = Path(rt.get("save_dir") or parent / run_id)
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "status.json", {"status": "preparing", "run_id": run_id})
    try:
        for folder in ("data", "source", "environment"):
            (run / folder).mkdir()
        code_files = {}
        for p in sources:
            rel = p.relative_to(source_root)
            out = run / "source" / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, out)
            code_files[rel.as_posix()] = sha256(out)
        for split, item in inputs.items():
            dest = run / "data" / f"{split}.csv"
            shutil.copyfile(item["source"], dest)
            if sha256(dest) != item["sha256"]:
                raise ValueError("Data changed while creating snapshot")
            cfg["data"][f"{split}_file"] = str(dest.resolve())
        if freeze_path:
            shutil.copyfile(freeze_path, run / "data" / "freeze_manifest.json")
        env = environment()
        write_json(run / "environment" / "environment.json", env)
        (run / "environment" / "installed.txt").write_text("\n".join(env["packages"]) + "\n", encoding="utf-8")
        git = {"revision": None, "status": "not_available; use source hashes"}
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True, timeout=10)
            if revision.returncode == 0:
                git = {"revision": revision.stdout.strip(), "status": subprocess.run(
                    ["git", "status", "--porcelain"], cwd=source_root, capture_output=True, text=True, timeout=10).stdout}
        except (OSError, subprocess.TimeoutExpired):
            pass
        cfg["runtime"].update(save_dir=str(run.resolve()), run_id=run_id)
        write_json(run / "config.requested.json", config)
        write_json(run / "config.resolved.json", cfg)
        write_json(run / "model_manifest.json", model)
        write_json(run / "data_manifest.json", inputs)
        write_json(run / "code_manifest.json", {"git": git, "files": code_files})
        protocol = dict(PROTOCOL, selection_metric=rt["select_metric"], label_semantics=schema_from_config(cfg).metadata())
        write_json(run / "evaluation_protocol.json", protocol)
        write_schema_metadata(run / "label_schema.json", schema_from_config(cfg))
        write_json(run / "run.json", {"schema_version": 1, "run_id": run_id, "experiment": exp,
                                      "seed": rt["seed"], "model_identity": model["identity"],
                                      "created_utc": datetime.now(timezone.utc).isoformat()})
        write_json(run / "status.json", {"status": "prepared", "run_id": run_id})
    except BaseException as error:
        write_json(run / "status.json", {"status": "preparation_failed", "error_type": type(error).__name__})
        raise
    return cfg, run
