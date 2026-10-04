# agents/agent_base.py
from typing import Optional, Dict, Tuple, Iterable
import inspect
import torch
import torch.nn as nn
from transformers import AutoModel

# --- PEFT (LoRA) ---
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
except Exception:
    LoraConfig = None
    PeftModel = None
    prepare_model_for_kbit_training = None  # 安全兜底

# --- bitsandbytes 量化（可选） ---
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


class BaseAgent(nn.Module):
    """
    通用 Agent 基类（支持 4/8bit 量化、LoRA、梯度检查点）：
    - forward() 返回 (logit, pooled)
    - encode() 仅返回 pooled（给通信前向）
    - head_on_embed() 接受通信后的 embedding，输出本维度 logit
    - 保证分类头与底层核心模型 device/dtype 对齐
    """
    def __init__(
        self,
        model_name: str,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        freeze_non_lora: bool = True,
        use_fp16: bool = True,
        explicit_hidden_size: Optional[int] = None,
        quantization: str = "4bit",             # "none"|"8bit"|"4bit"
        gradient_checkpointing: bool = True,
        device_map: Optional[str] = None,       # "auto" 或 None（不要传 "cuda:0"）
        move_to_device: Optional[str] = None,   # 非量化时可用 .to("cuda:0")
    ):
        super().__init__()

        if quantization not in ('none', '4bit', '8bit'):
            raise ValueError('quantization must be none, 4bit or 8bit')
        if use_lora and (LoraConfig is None or lora_config is None):
            raise RuntimeError('LoRA requested but PEFT or lora_config is unavailable')
        if quantization != 'none' and not _HAS_BNB:
            raise RuntimeError('Quantization requested but BitsAndBytesConfig is unavailable')
        torch_dtype = torch.float16 if use_fp16 else torch.float32

        # 量化配置（与 load_in_*bit 互斥，统一用 quantization_config）
        quantization_config = None
        if _HAS_BNB and quantization in ("8bit", "4bit"):
            if quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:  # "4bit"
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16 if use_fp16 else torch.float32,
                )

        # 仅当量化或者 device_map == "auto" 时传入 device_map
        load_device_map = device_map if (quantization_config is not None or device_map == "auto") else None

        base = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=load_device_map,
            quantization_config=quantization_config,
            return_dict=True,
        )

        # LoRA
        if use_lora and (LoraConfig is not None) and (lora_config is not None):
            if prepare_model_for_kbit_training is not None and quantization_config is not None:
                # 仅当量化训练时需要做预处理
                base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)

            lora_conf = LoraConfig(
                r=lora_config.get('r', 8),
                lora_alpha=lora_config.get('alpha', 16),
                target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
                lora_dropout=lora_config.get('dropout', 0.1),
                bias="none",
                task_type="SEQ_CLS",   # 避免用 CAUSAL_LM
            )
            base = get_peft_model(base, lora_conf)

            if freeze_non_lora:
                for n, p in base.named_parameters():
                    # 仅 LoRA 权重参与训练
                    if "lora_" not in n:
                        p.requires_grad = False
        elif freeze_non_lora:
            for p in base.parameters():
                p.requires_grad = False

        # 非量化且需要把整模搬到指定设备时
        if quantization_config is None and move_to_device is not None:
            base = base.to(move_to_device)

        self.base_model = base  # 可能是 PeftModel，也可能是原始 AutoModel

        # 梯度检查点
        if gradient_checkpointing:
            # Non-reentrant checkpointing preserves LoRA gradients with frozen embeddings.
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})
            self.base_model.config.use_cache = False

        # hidden size
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "hidden_size"):
            hidden_size = int(self.base_model.config.hidden_size)
        elif explicit_hidden_size is not None:
            hidden_size = int(explicit_hidden_size)
        else:
            raise ValueError("Cannot infer hidden_size; pass explicit_hidden_size.")

        # 获取被 LoRA 打补丁后的“核心模型”，用于对齐 device/dtype
        self._core_model = self._get_core_model(self.base_model)

        # 分类头
        self.classifier = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # 分类头迁到核心模型 device/dtype
        core_param = self._core_model.get_input_embeddings().weight
        core_device = core_param.device
        core_dtype  = core_param.dtype
        self.classifier = self.classifier.to(device=core_device, dtype=core_dtype)

        # 记录底层 forward 的可接受参数名，只传递安全 kwargs
        self._core_forward_allowed = set(inspect.signature(self._core_model.forward).parameters.keys())

    @staticmethod
    def _get_core_model(model: nn.Module) -> nn.Module:
        """若是 PeftModel，返回其内部的底层模型；否则返回自身。"""
        if PeftModel is not None and isinstance(model, PeftModel):
            if hasattr(model, "get_base_model") and callable(model.get_base_model):
                try:
                    return model.get_base_model()
                except Exception:
                    pass
            for attr in ("base_model", "model"):
                if hasattr(model, attr):
                    core = getattr(model, attr)
                    if isinstance(core, nn.Module):
                        return core
        return model

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return (last_hidden * mask).sum(dim=1) / denom

    def _core_forward_filtered(self, **kwargs):
        """仅把底层 forward 支持的 kwargs 传下去（剔除 labels 等不被支持的键）"""
        safe_kwargs = {k: v for k, v in kwargs.items() if k in self._core_forward_allowed}
        return self._core_model(**safe_kwargs)

    # ---------- 公共接口：一次性拿 (logit, pooled) ----------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self._core_forward_filtered(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
            pooled = outputs.pooler_output
        else:
            pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)

        # 确保 pooled 与分类头 device/dtype 一致
        if pooled.device != self.classifier.weight.device or pooled.dtype != self.classifier.weight.dtype:
            pooled = pooled.to(device=self.classifier.weight.device, dtype=self.classifier.weight.dtype)

        logit = self.classifier(pooled)  # [B,1]
        return logit, pooled

    # ---------- 通信友好：仅编码拿 embedding ----------
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        只进行编码，返回池化 embedding（不计算分类头）。
        适合“通信前：先拿四个 agent 的向量”。
        """
        outputs = self._core_forward_filtered(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
            pooled = outputs.pooler_output
        else:
            pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)

        if pooled.device != self.classifier.weight.device or pooled.dtype != self.classifier.weight.dtype:
            pooled = pooled.to(device=self.classifier.weight.device, dtype=self.classifier.weight.dtype)
        return pooled  # [B,D]

    # ---------- 通信后：给 embedding 直接过分类头 ----------
    def head_on_embed(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        接受通信后的 embedding（如 H'[:, i, :]），直接输出本维度 logit。
        """
        if pooled.device != self.classifier.weight.device or pooled.dtype != self.classifier.weight.dtype:
            pooled = pooled.to(device=self.classifier.weight.device, dtype=self.classifier.weight.dtype)
        return self.classifier(pooled)  # [B,1]

    # ---------- 便捷：仅返回需要训练的参数 ----------
    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """
        返回建议传给优化器的参数（LoRA 可训练权重 + 分类头）。
        若未使用 LoRA，则返回分类头参数。
        """
        return [p for p in self.parameters() if p.requires_grad]
