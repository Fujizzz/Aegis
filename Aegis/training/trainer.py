# training/trainer.py
from typing import Dict, Tuple
import os
import time
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from utils.run_archive import sha256, write_json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from agents.agent_base import BaseAgent
from communication.comm_module import CommunicationModule
from training.losses import bce_logits_weighted, diversity_loss, linear_ramp
from training.metrics import binary_metrics
from training.options import grad_accum_steps, epoch_counts, resolve_variant, NON_NEURAL_VARIANTS
from utils.data_loader import build_dataset, collate_fn
from utils.label_schema import schema_from_config, write_schema_metadata, require_schema_metadata
from utils.text_input import INPUT_FORMAT
from utils.checkpoints import checkpoint_paths, validate_paths
from utils.run_status import guarded, update_status
from utils.predictions import prediction_rows, write_predictions
from utils.resume import save_checkpoint, restore_training, verify_resume_inputs
from training.single_agent import InactiveAgent, SingleFusion
from utils.diagnostics import dimension_outputs, parameters, cuda_devices, reset_memory, memory, synchronize, gate_statistics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




from utils.devices import _normalize_agent_device_map, _normalize_move_to_device


class Trainer:
    @guarded
    def __init__(self, config: Dict):
        self.cfg = config
        if config.get('experiment', {}).get('variant') in NON_NEURAL_VARIANTS:
            raise ValueError('Use run_baseline.py for non-neural variants')
        self.grad_accum_steps = grad_accum_steps(config.get("training", {}))
        self.warmup_epochs, self.collab_epochs = epoch_counts(config.get("training", {}))
        run_dir = Path(config.get("runtime", {}).get("save_dir", ""))
        if not (run_dir / "run.json").is_file():
            raise ValueError("Use train.py to prepare an isolated, archived run first")
        if json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8")) != config:
            raise ValueError("Trainer config differs from archived config")
        if resolve_variant(config) != config:
            raise ValueError('Use train.py to resolve ablation settings before training')
        resuming = (run_dir / 'resume.json').is_file()
        if resuming:
            verify_resume_inputs(config)
        existing = [p for p in run_dir.glob('**/*.pth')
                    if not (resuming and p.relative_to(run_dir).parts[0] == 'recovery')]
        if (run_dir / "metrics.jsonl").exists() or existing:
            raise ValueError("Run already contains training outputs; choose a new run directory")
        self.label_schema = schema_from_config(config)
        self.method_id = (config.get("experiment") or {}).get("method_id", "aegis_neural_v1")
        print("[CFG] label_schema =", self.label_schema.name, "positive =", self.label_schema.positive)
        print("[CFG] train_file =", self.cfg['data']['train_file'])
        print("[CFG] val_file   =", self.cfg['data']['val_file'])

        rt = self.cfg.get("runtime", {}) or {}
        self.save_dir = rt.get("save_dir", "saved_models")
        os.makedirs(self.save_dir, exist_ok=True)
        self.run_test_after_train = bool(rt.get('run_test_after_train', True))
        self.run_state = dict(status='running', phase='initializing', run_id=rt['run_id'],
                              training={'status': 'pending'}, val={'status': 'pending'}, best=None,
                              test={'status': 'pending' if self.run_test_after_train else 'not_requested'})
        update_status(self)
        self.best_record = None
        if self.cfg['evaluation']['threshold'] != .5:
            raise ValueError('Prediction protocol requires fixed threshold 0.5')
        if self.run_test_after_train and not config['data'].get('test_file'):
            update_status(self, phase='test', test={'status': 'failed', 'reason': 'missing_test_file'})
            raise ValueError('Test requested but data.test_file is missing; explicitly disable it to train without test')
        schema_path = os.path.join(self.save_dir, "label_schema.json")
        if not os.path.exists(schema_path):
            root = Path(self.save_dir)
            if any(root.glob("**/*.pth")) or (root / "metrics.jsonl").exists() or (root / "test_metrics.json").exists():
                raise ValueError("Existing run lacks label metadata. Choose a fresh --save_dir; "
                                 "do not append new label semantics to legacy results.")
        write_schema_metadata(schema_path, self.label_schema)
        self.seed = int(rt.get("seed", 42))
        set_seed(self.seed)
        deterministic = bool(rt.get("deterministic", True))
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        write_json(Path(self.save_dir) / "environment" / "torch.json", {
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "deterministic_algorithms": deterministic,
            "device_map": self.cfg["hardware"]["device_map"]})
        self.save_each_epoch = bool(rt.get("save_each_epoch", False))
        self.select_metric = str(rt.get("select_metric", "binary_f1")).lower()
        if self.select_metric not in ('binary_f1', 'class_macro_f1', 'accuracy', 'roc_auc'):
            raise ValueError('Unsupported best selection metric')
        self.metrics_path = os.path.join(self.save_dir, "metrics.jsonl")
        print("[CFG] save_dir   =", self.save_dir)
        print("[CFG] seed       =", self.seed)

        # ✅ 通信层设备：你在 config 里手动指定
        self.device_comm = torch.device(self.cfg['hardware']['device_map']['communication'])
        print("[CFG] comm_device=", str(self.device_comm))

        # 1) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model']['base_model'], use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None)
        self.tokenizer.padding_side = "right"

        # 2) Data & Loader
        max_len = int(self.cfg['data'].get('max_length', 512))
        use_pseudo = bool(self.cfg['data'].get('use_pseudo_labels', True))
        num_workers = int(self.cfg.get("runtime", {}).get("num_workers", 0) or 0)

        train_set = build_dataset(self.cfg['data']['train_file'], self.tokenizer, max_len=max_len, use_pseudo=use_pseudo, mode="train", label_schema=self.label_schema)
        val_set = build_dataset(self.cfg['data']['val_file'], self.tokenizer, max_len=max_len, use_pseudo=False, mode="val", label_schema=self.label_schema)
        if not len(train_set) or not len(val_set):
            update_status(self, phase='validation' if not len(val_set) else 'training')
            raise ValueError('Train and validation datasets must be nonempty')

        bs = int(self.cfg['training'].get('batch_size', 4))
        self.train_loader = DataLoader(
            train_set, batch_size=bs, shuffle=True, num_workers=num_workers,
            collate_fn=lambda b: collate_fn(b, self.tokenizer, max_len)
        )
        self.val_loader = DataLoader(
            val_set, batch_size=bs, shuffle=False, num_workers=num_workers,
            collate_fn=lambda b: collate_fn(b, self.tokenizer, max_len)
        )

        # Optional: Test Loader (for final evaluation only)
        self.test_loader = None
        self.test_file = (self.cfg.get('data', {}) or {}).get('test_file', None)
        self.run_test_after_train = bool((self.cfg.get('runtime', {}) or {}).get('run_test_after_train', True))
        if self.test_file and self.run_test_after_train:
            print("[CFG] test_file  =", self.test_file)
            try:
                test_set = build_dataset(self.test_file, self.tokenizer, max_len=max_len, use_pseudo=False, mode="test", label_schema=self.label_schema)
                if not len(test_set):
                    raise ValueError('Empty test dataset')
                self.test_loader = DataLoader(
                    test_set, batch_size=bs, shuffle=False, num_workers=num_workers,
                    collate_fn=lambda b: collate_fn(b, self.tokenizer, max_len)
                )
            except Exception as e:
                update_status(self, phase='test')
                raise RuntimeError("Requested test data failed to load") from e

        # 3) Agents
        base = self.cfg['model']['base_model']
        lora_conf = dict(
            r=self.cfg['model'].get('lora_r', 8),
            alpha=self.cfg['model'].get('lora_alpha', 16),
            dropout=self.cfg['model'].get('lora_dropout', 0.1),
            target_modules=self.cfg['model'].get('target_modules', ["q_proj", "v_proj"]),
        )

        quant = self.cfg['model'].get('quantization', '4bit')
        grad_ckpt = self.cfg['model'].get('grad_ckpt', self.cfg['model'].get('gradient_checkpointing', True))

        def mk_agent(device_key: str) -> BaseAgent:
            dev_cfg = self.cfg['hardware']['device_map'].get(device_key, None)
            device_map = _normalize_agent_device_map(dev_cfg, quantization=quant)
            move_to = _normalize_move_to_device(dev_cfg, quantization=quant)

            return BaseAgent(
                model_name=base,
                use_lora=bool(self.cfg['model'].get('use_lora', False)),
                lora_config=lora_conf,
                explicit_hidden_size=None,
                use_fp16=bool(self.cfg['model'].get('use_fp16', True)),
                quantization=quant,
                gradient_checkpointing=bool(grad_ckpt),
                device_map=device_map,
                move_to_device=move_to,
            )

        self.agent_e = mk_agent('agent_emotion')
        inferred_hidden = int(self.agent_e.classifier.in_features)
        print(f"[INFO] Inferred hidden_size: {inferred_hidden}")
        single = config.get('experiment', {}).get('variant') == 'single_agent'
        if single:
            self.agent_e.classifier.requires_grad_(False)  # Only the final task head is supervised.
        self.agent_x = InactiveAgent(self.device_comm) if single else mk_agent('agent_exaggeration')
        self.agent_b = InactiveAgent(self.device_comm) if single else mk_agent('agent_bias')
        self.agent_c = InactiveAgent(self.device_comm) if single else mk_agent('agent_commonsense')

        print("[CFG] agent_emotion      =", str(self._agent_device(self.agent_e)))
        print("[CFG] agent_exaggeration =", str(self._agent_device(self.agent_x)))
        print("[CFG] agent_bias         =", str(self._agent_device(self.agent_b)))
        print("[CFG] agent_commonsense  =", str(self._agent_device(self.agent_c)))

        # 4) Comm + Heads
        self.comm = SingleFusion(self.device_comm) if single else CommunicationModule(
            dim=inferred_hidden,
            rounds=int(self.cfg['model'].get('comm_rounds', 2)),
            num_agents=int(self.cfg['model'].get('num_agents', 4)),
            attn_heads=int(self.cfg['model'].get('comm_heads', 4)),
            attn_dropout=float(self.cfg['model'].get('attn_dropout', 0.0)),
            ff_dropout=float(self.cfg['model'].get('ff_dropout', 0.0)),
            fuse_mode=str(self.cfg['model'].get('fuse_mode', 'mean')),
            use_self_mask=bool(self.cfg['model'].get('use_self_mask', False)),
            use_gate=self.cfg['model'].get('use_gate', True),
        ).to(self.device_comm)

        self.final_head = nn.Linear(inferred_hidden, 1).to(self.device_comm)
        self.team_layer = nn.Identity() if single else nn.Linear(4, 1).to(self.device_comm)

        self.pred_heads = nn.ModuleDict()
        for i in range(0 if single else 4):
            for j in range(4):
                if i == j:
                    continue
                self.pred_heads[f"{i}->{j}"] = nn.Linear(inferred_hidden, 1).to(self.device_comm)

        # 5) Optimizer & Scheduler
        lr = float(self.cfg['training'].get('learning_rate', 5e-5))
        wd = float(self.cfg['training'].get('weight_decay', 0.01))
        self.optim = AdamW(list(self._trainable_params()), lr=lr, weight_decay=wd)

        self.updates_per_epoch = (len(self.train_loader) + self.grad_accum_steps - 1) // self.grad_accum_steps
        total_steps = self.updates_per_epoch * (self.warmup_epochs + self.collab_epochs)
        warmup_steps = int(total_steps * 0.1)
        self.sched = get_linear_schedule_with_warmup(self.optim, warmup_steps, total_steps)
        self.optimizer_steps = 0
        self.start_epoch = 1
        self.best_score, self.best_epoch = -1e9, -1
        self.optimizer_steps_path = os.path.join(self.save_dir, "optimizer_steps.jsonl")
        print(f"[CFG] grad_accum_steps={self.grad_accum_steps} nominal_effective_batch={bs * self.grad_accum_steps} "
              f"updates_per_epoch={self.updates_per_epoch} total_updates={total_steps}")
        if resuming:
            update_status(self, phase='restoring')
            restore_training(self)

    @staticmethod
    def _agent_device(agent: BaseAgent) -> torch.device:
        # 以 classifier 的 device 作为 agent device
        return next(agent.classifier.parameters()).device

    def _forward_agents(self, batch):
        if self.cfg.get('experiment', {}).get('variant') == 'single_agent':
            device = self._agent_device(self.agent_e)
            logit, embedding = self.agent_e(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            embedding = embedding.to(self.device_comm, dtype=self.final_head.weight.dtype)
            logit = logit.to(self.device_comm, dtype=embedding.dtype)
            zero = torch.zeros_like(logit)
            return (logit, zero, zero, zero), (embedding, embedding, embedding, embedding)
        dbg = self.cfg.get('runtime', {}) or {}
        seq = bool(dbg.get('sequential_agents', True))
        dbg_sync = bool(dbg.get('debug_sync', True))
        use_streams = bool(dbg.get('use_cuda_streams', False))

        input_ids = batch['input_ids']
        attn = batch['attention_mask']

        dev_e = self._agent_device(self.agent_e)
        dev_x = self._agent_device(self.agent_x)
        dev_b = self._agent_device(self.agent_b)
        dev_c = self._agent_device(self.agent_c)

        if seq or (not use_streams):
            ids0 = input_ids.to(dev_e, non_blocking=False); attn0 = attn.to(dev_e, non_blocking=False)
            le, he = self.agent_e(ids0, attn0)
            if dbg_sync and dev_e.type == 'cuda': torch.cuda.synchronize(dev_e)

            ids0 = input_ids.to(dev_x, non_blocking=False); attn0 = attn.to(dev_x, non_blocking=False)
            lx, hx = self.agent_x(ids0, attn0)
            if dbg_sync and dev_x.type == 'cuda': torch.cuda.synchronize(dev_x)

            ids1 = input_ids.to(dev_b, non_blocking=False); attn1 = attn.to(dev_b, non_blocking=False)
            lb, hb = self.agent_b(ids1, attn1)
            if dbg_sync and dev_b.type == 'cuda': torch.cuda.synchronize(dev_b)

            ids1 = input_ids.to(dev_c, non_blocking=False); attn1 = attn.to(dev_c, non_blocking=False)
            lc, hc = self.agent_c(ids1, attn1)
            if dbg_sync and dev_c.type == 'cuda': torch.cuda.synchronize(dev_c)
        else:
            s0 = torch.cuda.Stream(device=dev_e) if dev_e.type == 'cuda' else None
            s1 = torch.cuda.Stream(device=dev_x) if dev_x.type == 'cuda' else None
            s2 = torch.cuda.Stream(device=dev_b) if dev_b.type == 'cuda' else None
            s3 = torch.cuda.Stream(device=dev_c) if dev_c.type == 'cuda' else None
            for stream, device in zip((s0, s1, s2, s3), (dev_e, dev_x, dev_b, dev_c)):
                if stream is not None:
                    stream.wait_stream(torch.cuda.current_stream(device))

            with torch.cuda.stream(s0) if s0 else nullcontext():
                ids = input_ids.to(dev_e, non_blocking=True); am = attn.to(dev_e, non_blocking=True)
                le, he = self.agent_e(ids, am)
            with torch.cuda.stream(s1) if s1 else nullcontext():
                ids = input_ids.to(dev_x, non_blocking=True); am = attn.to(dev_x, non_blocking=True)
                lx, hx = self.agent_x(ids, am)
            with torch.cuda.stream(s2) if s2 else nullcontext():
                ids = input_ids.to(dev_b, non_blocking=True); am = attn.to(dev_b, non_blocking=True)
                lb, hb = self.agent_b(ids, am)
            with torch.cuda.stream(s3) if s3 else nullcontext():
                ids = input_ids.to(dev_c, non_blocking=True); am = attn.to(dev_c, non_blocking=True)
                lc, hc = self.agent_c(ids, am)

            # Forward outputs and saved backward tensors must outlive worker streams.
            # Synchronize each producer even when debug_sync is disabled.
            for stream in (s0, s1, s2, s3):
                if stream is not None:
                    stream.synchronize()

        comm_param = next(self.comm.parameters())
        comm_dtype = comm_param.dtype

        # ✅ 表征统一搬到 comm device 做通信融合（非常关键，支持 3+1 手动分配）
        he = he.to(self.device_comm, dtype=comm_dtype)
        hx = hx.to(self.device_comm, dtype=comm_dtype)
        hb = hb.to(self.device_comm, dtype=comm_dtype)
        hc = hc.to(self.device_comm, dtype=comm_dtype)

        le = le.to(self.device_comm, dtype=comm_dtype)
        lx = lx.to(self.device_comm, dtype=comm_dtype)
        lb = lb.to(self.device_comm, dtype=comm_dtype)
        lc = lc.to(self.device_comm, dtype=comm_dtype)

        return (le, lx, lb, lc), (he, hx, hb, hc)

    @guarded
    def evaluate_loader(self, loader: DataLoader, split: str = "val", *, export=False) -> Dict[str, float]:
        """Evaluate on a given dataloader (val/test)."""
        if split not in ('val', 'test'):
            raise ValueError('Expected val or test split')
        update_status(self, phase='validation' if split == 'val' else 'test',
                      **{split: {'status': 'running', 'epoch': getattr(self, 'current_epoch', None)}})
        self.agent_e.eval()
        self.agent_x.eval()
        self.agent_b.eval()
        self.agent_c.eval()
        self.comm.eval()
        self.final_head.eval()
        self.team_layer.eval()
        self.pred_heads.eval()

        y_true_all, y_prob_all, identities, diagnostics = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                logits, (he, hx, hb, hc) = self._forward_agents(batch)
                states, fused = self.comm([he, hx, hb, hc])
                final_logit = self.final_head(fused)
                probs = torch.sigmoid(final_logit).squeeze(1)
                labels = batch['label'].to(self.device_comm).long()

                y_true_all.append(labels.detach().cpu())
                y_prob_all.append(probs.detach().cpu())
                if export:
                    if len(batch.get('metadata', [])) != len(labels):
                        raise ValueError('Missing or misaligned sample identity metadata')
                    identities.extend(batch['metadata'])
                    if self.cfg['experiment']['variant'] != 'single_agent':
                        diagnostics.extend(dimension_outputs([self.agent_e,self.agent_x,self.agent_b,self.agent_c],
                            logits,states,self.comm,self.team_layer,
                            team_enabled=float(self.cfg['training']['loss_weights'].get('cons',0))>0))

        if not y_true_all:
            raise ValueError(f"Empty {split} loader")
        y_true = torch.cat(y_true_all, dim=0).numpy()
        y_prob = torch.cat(y_prob_all, dim=0).numpy()
        metrics = binary_metrics(y_true, y_prob, thr=self.cfg['evaluation']['threshold'])
        if export:
            manifest_path = Path(self.save_dir)/'best/checkpoint_manifest.json'
            checkpoint_id = 'sha256:' + sha256(manifest_path)
            if (getattr(self, 'loaded_checkpoint_id', None) != checkpoint_id or not self.best_record or
                    checkpoint_id != 'sha256:' + self.best_record['checkpoint_manifest_sha256']):
                raise ValueError('Prediction export requires the selected, loaded best checkpoint')
            rows = prediction_rows(identities, y_true, y_prob, self.label_schema, self.cfg, split, checkpoint_id)
            if diagnostics:
                if len(diagnostics)!=len(rows):raise ValueError('Misaligned dimension diagnostics')
                for row,detail in zip(rows,diagnostics):row.update(detail)
            artifact = write_predictions(Path(self.save_dir)/f'{split}_predictions.csv', rows, manifest_path,
                                         self.cfg['data'][f'{split}_file'])
            update_status(self, **{f'{split}_predictions': artifact})
        update_status(self, **{split: dict(status='success', n=metrics['n'],
                      epoch=self.best_record['epoch'] if export else getattr(self, 'current_epoch', None), auc_defined=metrics['auc_defined'],
                      auc_reason=None if metrics['auc_defined'] else 'single_class')})
        auc_str = "undefined" if metrics['roc_auc'] is None else f"{metrics['roc_auc']:.4f}"
        tag = split.upper()
        print(
            f"[{tag} positive={self.label_schema.positive}] acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
            f"binary_f1={metrics['binary_f1']:.4f} class_macro_f1={metrics['class_macro_f1']:.4f} roc_auc={auc_str} | tp={metrics['tp']:.0f} fp={metrics['fp']:.0f} "
            f"tn={metrics['tn']:.0f} fn={metrics['fn']:.0f}"
        )
        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Backward-compatible: evaluate on validation set."""
        return self.evaluate_loader(self.val_loader, split="val")

    @guarded
    def train_loop(self):
        import torch.nn.functional as F

        W_base = self.cfg['training']['loss_weights']
        warmup_epochs, collab_epochs = self.warmup_epochs, self.collab_epochs
        total_epochs = warmup_epochs + collab_epochs
        grad_clip = float(self.cfg['training'].get('grad_clip', 1.0))
        log_steps = int(self.cfg.get('runtime', {}).get('log_steps', 50) or 50)

        W0 = {"cons": 0.0, "pred": 0.0, "div": 0.0, "gate": 0.0}
        best_score, best_epoch = self.best_score, self.best_epoch
        modules=[self.agent_e,self.agent_x,self.agent_b,self.agent_c,self.comm,self.final_head,self.team_layer,self.pred_heads]
        devices=cuda_devices(modules)
        write_json(Path(self.save_dir)/'resource_profile.json',dict(parameters=parameters(modules),
            batch_size=self.train_loader.batch_size,grad_accum_steps=self.grad_accum_steps,
            max_length=self.cfg['data'].get('max_length',512),device_map=self.cfg['hardware']['device_map'],
            quantization=self.cfg['model'].get('quantization'),
            timing_scope='Training epoch including data loading, diagnostics and optimizer; excludes validation/checkpoint I/O.',
            resumed_segment_only=(Path(self.save_dir)/'resume.json').exists()))

        for ep in range(self.start_epoch, total_epochs + 1):
            self.current_epoch = ep
            self.loaded_checkpoint_id = None
            update_status(self, phase='training', training={'status': 'running', 'epoch': ep})
            self.agent_e.train()
            self.agent_x.train()
            self.agent_b.train()
            self.agent_c.train()
            self.comm.train()
            self.final_head.train()
            self.team_layer.train()
            self.pred_heads.train()

            reset_memory(devices)
            t0 = time.perf_counter()
            running = {"final": 0.0, "dim": 0.0, "cons": 0.0, "pred": 0.0, "div": 0.0, "gate": 0.0, "total": 0.0}
            samples_seen = 0
            window_samples = 0
            window_batches = 0
            gate_sum = 0.0
            # Backward uses a fixed nominal denominator. At each update correct it
            # to the actual sample count, including short batches/windows.
            nominal_samples = self.train_loader.batch_size * self.grad_accum_steps
            self.optim.zero_grad(set_to_none=True)

            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {ep}"), start=1):
                (le, lx, lb, lc), (he, hx, hb, hc) = self._forward_agents(batch)
                per_states, fused = self.comm([he, hx, hb, hc])
                final_logit = self.final_head(fused)

                # ====== ✅ 关键：head_on_embed 在各自 agent device 上算 ======
                # per_states 在 comm device；为了支持 3+1 分配，需要把 slice 搬到对应 agent device
                dev_e = self._agent_device(self.agent_e)
                dev_x = self._agent_device(self.agent_x)
                dev_b = self._agent_device(self.agent_b)
                dev_c = self._agent_device(self.agent_c)

                ze2 = self.agent_e.head_on_embed(per_states[:, 0, :].to(dev_e))
                zx2 = self.agent_x.head_on_embed(per_states[:, 1, :].to(dev_x))
                zb2 = self.agent_b.head_on_embed(per_states[:, 2, :].to(dev_b))
                zc2 = self.agent_c.head_on_embed(per_states[:, 3, :].to(dev_c))

                # ====== ✅ p_vec2 必须在同一张卡上拼接：统一搬回 comm device ======
                ze2_c = ze2.to(device=self.device_comm,dtype=per_states.dtype)
                zx2_c = zx2.to(device=self.device_comm,dtype=per_states.dtype)
                zb2_c = zb2.to(device=self.device_comm,dtype=per_states.dtype)
                zc2_c = zc2.to(device=self.device_comm,dtype=per_states.dtype)
                p_vec2 = torch.cat(
                    [torch.sigmoid(ze2_c), torch.sigmoid(zx2_c), torch.sigmoid(zb2_c), torch.sigmoid(zc2_c)], dim=1
                )

                y_final = batch['label'].to(self.device_comm).float().unsqueeze(1)
                p_final = torch.sigmoid(final_logit)

                loss_final = bce_logits_weighted(final_logit, y_final, None) * float(W_base.get("final", 1.0))

                # ====== ✅ 维度 loss：在 logit.device 上算，再搬回 comm 汇总 ======
                def dim_loss_one(logit, name: str):
                    if (not self.cfg['data'].get('use_pseudo_labels', True) or
                            float(W_base.get(name, 0.0)) == 0):
                        return final_logit.reshape(-1)[:0].sum()
                    y = batch.get(f"{name}_label", None)
                    m = batch.get(f"{name}_mask", None)
                    if y is None or m is None:
                        return logit.reshape(-1)[:0].sum().to(self.device_comm)

                    dev = logit.device  # ✅ 与 logit 同卡
                    y = y.to(dev).float().unsqueeze(1)
                    m = m.to(dev).float().unsqueeze(1)
                    w = batch.get(f"{name}_weight", None)
                    if w is not None:
                        w = w.to(dev).float().unsqueeze(1)
                        w = w * m
                    else:
                        w = m

                    loss_raw = bce_logits_weighted(logit, y, w)
                    # 兼容 bce_logits_weighted 返回向量/标量两种实现
                    if hasattr(loss_raw, "ndim") and loss_raw.ndim > 0:
                        loss_raw = loss_raw.mean()

                    return loss_raw.to(self.device_comm)

                loss_dim = torch.tensor(0.0, device=self.device_comm)
                loss_dim = loss_dim + dim_loss_one(ze2, 'emotion')      * float(W_base.get('emotion', 0.0))
                loss_dim = loss_dim + dim_loss_one(zx2, 'exaggeration') * float(W_base.get('exaggeration', 0.0))
                loss_dim = loss_dim + dim_loss_one(zb2, 'bias')         * float(W_base.get('bias', 0.0))
                loss_dim = loss_dim + dim_loss_one(zc2, 'commonsense')  * float(W_base.get('commonsense', 0.0))

                if ep <= warmup_epochs:
                    Wc = W0
                else:
                    cur = ep - warmup_epochs
                    tot = max(1, collab_epochs)
                    Wc = {k: float(linear_ramp(0.0, float(W_base.get(k, 0.0)), cur, tot)) for k in ["cons", "pred", "div", "gate"]}

                zero = final_logit.reshape(-1)[:0].sum()
                loss_cons = zero
                if Wc['cons'] > 0:
                    team_logit = self.team_layer(p_vec2)
                    p_team = torch.sigmoid(team_logit)
                    loss_cons = (F.mse_loss(p_team, p_final) + bce_logits_weighted(team_logit, y_final, None)) * Wc['cons']

                loss_pred = zero
                if Wc['pred'] > 0:
                    for i in range(4):
                        si = per_states[:, i, :]
                        for j in range(4):
                            if i == j:
                                continue
                            pj = p_vec2[:, j].unsqueeze(1).detach()
                            pred = self.pred_heads[f"{i}->{j}"](si)
                            loss_pred = loss_pred + F.mse_loss(torch.sigmoid(pred), pj)
                    loss_pred = loss_pred * Wc['pred']

                loss_div = diversity_loss(p_vec2) * Wc['div'] if Wc['div'] > 0 else zero

                gate_l1 = self.comm.gate_l1()
                loss_gate = gate_l1 * Wc['gate'] if Wc['gate'] > 0 else zero

                loss_total = loss_final + loss_dim + loss_cons + loss_pred + loss_div + loss_gate
                # Observe existing losses only; disabled ablations must not execute them.
                detail=dict(epoch=ep,micro_step=step,batch_size=len(y_final),
                    diversity_raw=float((loss_div.detach()/Wc['div']).cpu()) if Wc['div']>0 else None,
                    diversity_weight=Wc['div'],gate=gate_statistics(self.comm),
                    dimension_std=p_vec2.detach().float().std(dim=0,unbiased=False).cpu().tolist())
                with (Path(self.save_dir)/'stability.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(detail,allow_nan=False)+'\n')

                batch_samples = int(y_final.shape[0])
                window_samples += batch_samples
                window_batches += 1
                (loss_total * (batch_samples / nominal_samples)).backward()
                if window_batches == self.grad_accum_steps or step == len(self.train_loader):
                    params = list(self._trainable_params())
                    with torch.no_grad():
                        for p in params:
                            if p.grad is not None:
                                p.grad.mul_(nominal_samples / window_samples)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    lr_used = [group['lr'] for group in self.optim.param_groups]
                    self.optim.step()
                    self.sched.step()
                    self.optimizer_steps += 1
                    update = {
                        "epoch": ep, "micro_step": step, "optimizer_step": self.optimizer_steps,
                        "micro_batches": window_batches, "effective_batch_size": window_samples,
                        "lr_used": lr_used, "lr_next": self.sched.get_last_lr(),
                        "grad_norm_before_clip": float(grad_norm.detach().cpu()),
                    }
                    with open(self.optimizer_steps_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(update, allow_nan=False) + "\n")
                    self.optim.zero_grad(set_to_none=True)
                    window_samples = window_batches = 0

                samples_seen += batch_samples
                gate_sum += float(gate_l1.detach().cpu()) * batch_samples
                for name, loss in (("final", loss_final), ("dim", loss_dim), ("cons", loss_cons),
                                   ("pred", loss_pred), ("div", loss_div), ("gate", loss_gate), ("total", loss_total)):
                    running[name] += float(loss.detach().cpu()) * batch_samples

                if step % log_steps == 0:
                    avg = {k: running[k] / max(1, samples_seen) for k in running}
                    print(
                        f"[EP{ep} step{step}] total={avg['total']:.4f} final={avg['final']:.4f} dim={avg['dim']:.4f} "
                        f"cons={avg['cons']:.4f} pred={avg['pred']:.4f} div={avg['div']:.4f} gate={avg['gate']:.4f}"
                    )

            synchronize(devices)
            elapsed = time.perf_counter() - t0
            peak_memory=memory(devices)
            avg = {k: running[k] / max(1, samples_seen) for k in running}
            print(
                f"[EP{ep} done] total={avg['total']:.4f} final={avg['final']:.4f} dim={avg['dim']:.4f} cons={avg['cons']:.4f} "
                f"pred={avg['pred']:.4f} div={avg['div']:.4f} gate={avg['gate']:.4f} | elapsed={elapsed:.1f}s"
            )

            metrics = {}
            update_status(self, phase='validation')
            try:
                metrics = self.evaluate()
                rec = {"epoch": ep, "elapsed_s": float(elapsed)}
                rec['cuda_peak_memory']=peak_memory
                rec['samples_seen']=samples_seen
                rec["train_losses"] = avg
                rec["gate_activation_mean"] = (gate_sum / max(1, samples_seen)
                                               if self.comm.rounds > 0 and self.comm.use_gate else None)
                rec["variant"] = self.cfg.get('experiment', {}).get('variant', 'custom')
                rec["comm_rounds"] = self.comm.rounds
                rec["use_gate"] = self.comm.use_gate
                rec["use_pseudo_labels"] = self.cfg['data'].get('use_pseudo_labels', True)
                rec["optimizer_steps"] = self.optimizer_steps
                rec["grad_accum_steps"] = self.grad_accum_steps
                rec.update(metrics)
                rec["method_id"] = self.method_id
                rec["label_semantics"] = self.label_schema.metadata()
                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
            except Exception as e:
                raise RuntimeError("Validation failed; run is incomplete") from e

            score = metrics.get(self.select_metric)
            if score is None or not math.isfinite(score):
                raise ValueError('Selection metric is undefined or nonfinite; no valid best selection')
            if score > best_score:
                best_score, best_epoch = float(score), ep
                self.best_score, self.best_epoch = best_score, best_epoch
                update_status(self, phase='saving_best')
                self.best_record = dict(epoch=ep, metric=self.select_metric, score=best_score,
                                        run_id=self.cfg['runtime']['run_id'], split='val',
                                        validation_metrics=metrics)
                self._save_all(subdir="best")
                self.best_record['checkpoint_manifest_sha256'] = sha256(Path(self.save_dir)/'best/checkpoint_manifest.json')
                write_json(Path(self.save_dir)/'best_selection.json', self.best_record)
                update_status(self, best=self.best_record)
                print(f"[SAVE] best updated: epoch={best_epoch} {self.select_metric}={best_score:.4f}")

            if self.save_each_epoch:
                self._save_all(subdir=f"epoch_{ep}")
            update_status(self, phase='saving_last')
            self._save_all(subdir='last')

            stop = self.cfg.get('runtime', {}).get('stop_after_epoch')
            if stop is not None and ep >= stop and ep < total_epochs:
                update_status(self, status='interrupted', phase='epoch_boundary',
                              training={'status': 'interrupted', 'epochs': ep},
                              recovery_checkpoint=str(Path(self.save_dir) / 'last'))
                return  # No final test evaluation before the remaining epochs.

        if self.start_epoch > total_epochs:
            # A completed checkpoint can retry failed final evaluation with zero updates.
            update_status(self, phase='saving_last')
            self._save_all(subdir='last')
        update_status(self, training={'status': 'success', 'epochs': total_epochs})
        if best_epoch > 0:
            print(f"[DONE] best_epoch={best_epoch} best_{self.select_metric}={best_score:.4f}")

        if best_epoch < 1:
            raise RuntimeError("No valid best checkpoint was selected")

        # Reload the selected best and export its validation predictions as well.
        update_status(self, phase='validation')
        self._load_all(subdir='best')
        selected_metrics = self.evaluate_loader(self.val_loader, split='val', export=True)
        if selected_metrics != self.best_record['validation_metrics']:
            raise RuntimeError('Reloaded best validation metrics differ from selection record')
        write_json(Path(self.save_dir)/'val_metrics.json', dict(selected_metrics,
                   best_epoch=best_epoch, checkpoint_id=self.loaded_checkpoint_id))

        # Final evaluation on test set (best checkpoint)
        if self.test_loader is not None and self.run_test_after_train:
            update_status(self, phase='test')
            try:
                test_metrics = self.evaluate_loader(self.test_loader, split="test", export=True)
                out_path = os.path.join(self.save_dir, "test_metrics.json")
                rec = {"best_epoch": int(best_epoch), f"best_{self.select_metric}": float(best_score)}
                rec.update(test_metrics)
                rec["method_id"] = self.method_id
                rec["label_semantics"] = self.label_schema.metadata()
                rec['checkpoint_id'] = self.loaded_checkpoint_id
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                print(f"[TEST] metrics saved -> {out_path}")
            except Exception as e:
                raise RuntimeError("Requested test evaluation failed") from e
        update_status(self, status='completed', phase='completed',
                      training={'status': 'success', 'epochs': total_epochs})

    def _trainable_params(self):
        for m in [self.agent_e, self.agent_x, self.agent_b, self.agent_c,
                  self.comm, self.final_head, self.team_layer, self.pred_heads]:
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    def _load_all(self, subdir: str = ""):
        """Load checkpoints saved by _save_all (e.g., subdir='best')."""
        in_dir = self.save_dir if not subdir else os.path.join(self.save_dir, subdir)
        manifest_path = Path(in_dir)/'checkpoint_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['run_id'] != self.cfg['runtime']['run_id']:
            raise ValueError('Checkpoint belongs to a different run')
        if subdir == 'best':
            if (not self.best_record or manifest['epoch'] != self.best_record['epoch'] or
                    sha256(manifest_path) != self.best_record['checkpoint_manifest_sha256']):
                raise ValueError('Best checkpoint is not bound to this run validation selection')
        require_schema_metadata(os.path.join(in_dir, "label_schema.json"), self.label_schema)
        validate_paths(checkpoint_paths(in_dir), self.cfg)

        def _load(m: nn.Module, path: str):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            try:
                dev = next(m.parameters()).device
            except StopIteration:
                dev = self.device_comm
            manifest = json.loads((Path(in_dir) / "checkpoint_manifest.json").read_text(encoding="utf-8"))
            if sha256(path) != manifest["sha256"][Path(path).name]:
                raise ValueError("Checkpoint content hash mismatch")
            sd = torch.load(path, map_location='cpu', weights_only=True)
            m.load_state_dict(sd, strict=True)

        _load(self.agent_e, os.path.join(in_dir, "agent_emotion.pth"))
        _load(self.agent_x, os.path.join(in_dir, "agent_exaggeration.pth"))
        _load(self.agent_b, os.path.join(in_dir, "agent_bias.pth"))
        _load(self.agent_c, os.path.join(in_dir, "agent_commonsense.pth"))
        _load(self.comm, os.path.join(in_dir, "communication_module.pth"))
        _load(self.final_head, os.path.join(in_dir, "final_classifier.pth"))
        _load(self.team_layer, os.path.join(in_dir, "team_layer.pth"))
        _load(self.pred_heads, os.path.join(in_dir, "pred_heads.pth"))
        self.loaded_checkpoint_id = 'sha256:' + sha256(manifest_path)
        print(f"[LOAD] checkpoints <- {in_dir}")

    def _save_all(self, subdir: str = ""):
        if not subdir or Path(subdir).name != subdir:
            raise ValueError('Checkpoint requires a simple subdirectory name')
        save_checkpoint(self, subdir)
        print(f"[SAVE] checkpoints -> {Path(self.save_dir) / subdir}")
