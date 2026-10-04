# training/losses.py
from typing import Optional
import torch
import torch.nn.functional as F


def bce_logits_weighted(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    带权重/掩码的 BCE；targets 和 weights 按 logits 的形状广播。
    权重须有限且非负；权重为零的位置不参与 BCE 或梯度。
    全屏蔽返回保持计算图的零标量；有效目标须有限且在 [0,1]。
    缺失标签由数据层转换为零权重；直接调用时也可用零权重屏蔽 NaN。
    """
    targets = torch.broadcast_to(targets.to(logits), logits.shape)
    weights = (torch.ones_like(logits) if weights is None else
               torch.broadcast_to(weights.to(logits), logits.shape))
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("BCE weights must be finite and nonnegative")
    active = weights > 0
    # Index before BCE: multiplying an already computed NaN by zero is unsafe.
    selected_logits, selected_targets = logits[active], targets[active]
    if not torch.isfinite(selected_targets).all() or ((selected_targets < 0) | (selected_targets > 1)).any():
        raise ValueError("Active BCE targets must be finite and in [0, 1]")
    if not active.any():
        return selected_logits.sum()
    selected_weights = weights[active]
    # Scaling avoids overflow in the denominator for large finite weights.
    selected_weights = selected_weights / selected_weights.max()
    loss = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="none")
    return (loss * selected_weights).sum() / selected_weights.sum()


def diversity_loss(p_vec: torch.Tensor) -> torch.Tensor:
    """
    多样性惩罚：对 [B,N] 的维度概率做标准化，
    计算相关矩阵的非对角元素平方均值，鼓励各维输出解耦。
    """
    if p_vec.ndim != 2:
        raise ValueError(f"diversity_loss expects 2D tensor [B,N], got shape {tuple(p_vec.shape)}")
    B, N = p_vec.shape
    if B <= 1 or N <= 1:
        return torch.zeros((), device=p_vec.device, dtype=p_vec.dtype)

    mean = p_vec.mean(dim=0, keepdim=True)
    std = p_vec.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.full_like(std, 1e-6), std)
    x = (p_vec - mean) / std  # [B,N]

    corr = (x.t() @ x) / (B - 1 + 1e-6)  # [N,N]
    off_diag = corr - torch.diag(torch.diag(corr))
    return (off_diag ** 2).mean()


def linear_ramp(start: float, end: float, cur: int, total: int) -> float:
    """
    线性爬坡：cur ∈ [0, total]，当 total<=1 时直接返回 end。
    """
    if total <= 1:
        return float(end)
    t = max(0.0, min(1.0, float(cur) / float(total)))
    return float(start + (end - start) * t)
