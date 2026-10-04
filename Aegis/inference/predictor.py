# inference/predictor.py
from typing import Dict, Optional, Tuple
from pathlib import Path
import torch
from transformers import AutoTokenizer

from agents.agent_base import BaseAgent
from communication.comm_module import CommunicationModule
from utils.label_schema import schema_from_config, require_schema_metadata
from training.options import resolve_variant, NON_NEURAL_VARIANTS
from utils.devices import _normalize_agent_device_map, _normalize_move_to_device
from utils.text_input import build_news_text, tokenize_texts
from utils.checkpoints import validate_paths
from training.single_agent import InactiveAgent, SingleFusion


class Predictor:
    def __init__(self, config: Dict, model_paths: Dict[str, str]):
        """
        model_paths 需要包含（至少）：
          - 'agent_emotion', 'agent_exaggeration', 'agent_bias', 'agent_commonsense'
          - 'communication_module'（或旧 key: 'communication'）
          - 'final_classifier'
        可选：
          - 'team_layer' （若训练时存过，可加载用于解释性投票）
        """
        self.cfg = resolve_variant(config)
        if self.cfg.get('experiment', {}).get('variant') in NON_NEURAL_VARIANTS:
            raise ValueError('Neural Predictor cannot load a text/API baseline')
        self.label_schema = schema_from_config(config)
        # Check every loaded component before constructing expensive models.
        for key in ("agent_emotion", "agent_exaggeration", "agent_bias", "agent_commonsense",
                    "final_classifier", "communication_module", "communication", "team_layer"):
            if key in model_paths:
                require_schema_metadata(Path(model_paths[key]).parent / "label_schema.json", self.label_schema)
        validate_paths(model_paths, self.cfg)
        self.device_comm = torch.device(self.cfg['hardware']['device_map']['communication'])

        # --- tokenizer ---
        self.tok = AutoTokenizer.from_pretrained(self.cfg['model']['base_model'], use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = getattr(self.tok, "eos_token", None)
        self.tok.padding_side = "right"

        base = self.cfg['model']['base_model']
        hidden_cfg: Optional[int] = self.cfg['model'].get('hidden_size', None)

        # LoRA 配置（与训练对齐）
        lora_conf = dict(
            r=self.cfg['model'].get('lora_r', 8),
            alpha=self.cfg['model'].get('lora_alpha', 16),
            dropout=self.cfg['model'].get('lora_dropout', 0.1),
            target_modules=self.cfg['model'].get('target_modules', ["q_proj", "v_proj"]),
        )

        quant = self.cfg['model'].get('quantization', '4bit')
        grad_ckpt = self.cfg['model'].get('grad_ckpt', self.cfg['model'].get('gradient_checkpointing', True))
        use_fp16 = self.cfg['model'].get('use_fp16', True)

        def mk_agent(key):
            placement = self.cfg['hardware']['device_map'].get(key)
            device_map = _normalize_agent_device_map(placement, quant)
            move_to_device = _normalize_move_to_device(placement, quant)
            return BaseAgent(
                model_name=base,
                use_lora=self.cfg['model'].get('use_lora', False),
                lora_config=lora_conf,
                freeze_non_lora=True,
                use_fp16=use_fp16,
                explicit_hidden_size=hidden_cfg,
                quantization=quant,
                gradient_checkpointing=grad_ckpt,
                device_map=device_map,
                move_to_device=move_to_device,
            ).eval()

        # --- 四个 agent ---
        self.ae = mk_agent('agent_emotion')
        single = self.cfg.get('experiment', {}).get('variant') == 'single_agent'
        self.ax = InactiveAgent(self.device_comm) if single else mk_agent('agent_exaggeration')
        self.ab = InactiveAgent(self.device_comm) if single else mk_agent('agent_bias')
        self.ac = InactiveAgent(self.device_comm) if single else mk_agent('agent_commonsense')

        self.ae.load_state_dict(torch.load(model_paths['agent_emotion'], map_location="cpu", weights_only=True))
        self.ax.load_state_dict(torch.load(model_paths['agent_exaggeration'], map_location="cpu", weights_only=True))
        self.ab.load_state_dict(torch.load(model_paths['agent_bias'], map_location="cpu", weights_only=True))
        self.ac.load_state_dict(torch.load(model_paths['agent_commonsense'], map_location="cpu", weights_only=True))

        # 推断 hidden_size（若 config 未显式给出）
        inferred_hidden = hidden_cfg
        if inferred_hidden is None:
            inferred_hidden = int(self.ae.classifier.in_features)

        # --- 通信模块 & 最终分类器 ---
        self.comm = SingleFusion(self.device_comm) if single else CommunicationModule(
            dim=inferred_hidden,
            rounds=self.cfg['model'].get('comm_rounds', 2),
            num_agents=self.cfg['model'].get('num_agents', 4),
            attn_heads=self.cfg['model'].get('comm_heads', 4),
            attn_dropout=self.cfg['model'].get('attn_dropout', 0.0),
            ff_dropout=self.cfg['model'].get('ff_dropout', 0.0),
            fuse_mode=self.cfg['model'].get('fuse_mode', 'mean'),
            use_self_mask=self.cfg['model'].get('use_self_mask', False),
            use_gate=self.cfg['model'].get('use_gate', True),
        ).to(self.device_comm).eval()

        # 兼容旧 key：'communication'
        comm_path = model_paths.get('communication_module', None) or model_paths.get('communication', None)
        if comm_path is None:
            raise KeyError("model_paths must contain 'communication_module' (or legacy key 'communication').")
        self.comm.load_state_dict(torch.load(comm_path, map_location="cpu", weights_only=True))

        self.final_head = torch.nn.Linear(inferred_hidden, 1).to(self.device_comm).eval()
        self.final_head.load_state_dict(torch.load(model_paths['final_classifier'], map_location="cpu", weights_only=True))

        # 可选：team_layer（若存在用于解释的“会后投票”）
        self.team_layer = None
        if 'team_layer' in model_paths:
            tl = torch.nn.Identity() if single else torch.nn.Linear(4, 1).to(self.device_comm).eval()
            tl.load_state_dict(torch.load(model_paths['team_layer'], map_location='cpu', weights_only=True))
            self.team_layer = tl

        self.ae.eval(); self.ax.eval(); self.ab.eval(); self.ac.eval()
        self.comm.eval(); self.final_head.eval()
        if self.team_layer is not None:
            self.team_layer.eval()

    @staticmethod
    def _agent_device(agent: BaseAgent) -> torch.device:
        # 用分类头所在 device 作为 agent 的“入口 device”
        return next(agent.classifier.parameters()).device

    @torch.no_grad()
    def _forward_agent(
        self,
        agent: BaseAgent,
        input_ids_cpu: torch.Tensor,
        attn_cpu: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self._agent_device(agent)
        input_ids = input_ids_cpu.to(dev)
        attn = attn_cpu.to(dev) if attn_cpu is not None else None
        logit, emb = agent(input_ids, attn)
        return logit, emb

    @torch.no_grad()
    def predict(self, text) -> Dict:
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts, *, diagnostics=True):
        if not texts:
            raise ValueError('Cannot predict an empty batch')
        enc = tokenize_texts(self.tok, [build_news_text(text) for text in texts],
                             self.cfg['data'].get('max_length', 512))
        single = self.cfg.get('experiment', {}).get('variant') == 'single_agent'
        agents = [self.ae] if single else [self.ae, self.ax, self.ab, self.ac]
        outputs = [self._forward_agent(a, enc['input_ids'], enc.get('attention_mask')) for a in agents]
        if single:
            fused = outputs[0][1].to(self.device_comm, dtype=self.final_head.weight.dtype)
            states = None
        else:
            dtype = next(self.comm.parameters()).dtype
            states, fused = self.comm([embedding.to(self.device_comm, dtype=dtype) for _,embedding in outputs])
        probabilities = torch.sigmoid(self.final_head(fused)).flatten().cpu().tolist()
        rows = [self.label_schema.prediction(p) for p in probabilities]
        if diagnostics and not single:
            from utils.diagnostics import dimension_outputs
            details = dimension_outputs(agents, [z for z,_ in outputs], states, self.comm, self.team_layer,
                team_enabled=float(self.cfg['training']['loss_weights'].get('cons',0)) > 0)
            for row,detail in zip(rows,details):
                row.update(detail)
                # Retain the historical public inference aliases.
                for stage in ('pre','post'):
                    row[stage+'_commonsense_conflict'] = row[stage+'_commonsense']
        return rows
