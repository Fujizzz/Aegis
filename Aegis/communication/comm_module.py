# communication/comm_module.py
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CommunicationModule(nn.Module):
    """
    多轮可微通信模块：
    - 输入：长度为 N 的 Agent 表示列表，每个 [B, D]
    - K 轮消息传递（多头自注意力 + 门控前馈更新）
    - 输出：
        per_agent_states: [B, N, D]  每个 agent 通信后的表示
        fused: [B, D]                 全局融合表示（均值池化或 Query 注意力）
    """
    def __init__(
        self,
        dim: int,
        rounds: int = 2,
        num_agents: int = 4,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        fuse_mode: str = "mean",          # "mean" | "attn"
        use_self_mask: bool = False,      # True 时抑制自注意力（不看自己）
        use_gate: bool = True,
    ):
        super().__init__()
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
            raise ValueError('rounds must be a nonnegative integer')
        if not isinstance(use_gate, bool):
            raise ValueError('use_gate must be a boolean')
        if fuse_mode not in ('mean', 'attn'):
            raise ValueError('fuse_mode must be mean or attn')
        assert dim % attn_heads == 0, f"dim ({dim}) must be divisible by attn_heads ({attn_heads})"
        self.dim = dim
        self.rounds = rounds
        self.num_agents = num_agents
        self.attn_heads = attn_heads
        self.dh = dim // attn_heads
        self.fuse_mode = fuse_mode
        self.use_self_mask = use_self_mask
        self.use_gate = use_gate

        # 多头投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(ff_dropout),
        )

        # 门控，控制引入的外部信息量（concat([m_i, agg_i]) -> [B,N,1]）
        self.gate = nn.Linear(dim * 2, 1)

        # 规范化（Pre-Norm 风格更稳定）
        self.ln_pre = nn.LayerNorm(dim)
        self.ln_post = nn.LayerNorm(dim)

        # 融合为 Query-Attention 时的可学习查询
        if self.fuse_mode == "attn":
            self.fuse_query = nn.Parameter(torch.randn(1, 1, dim))  # [1,1,D]
            nn.init.xavier_uniform_(self.fuse_query)

        # 记录门控以便正则
        self._latest_gates: Optional[torch.Tensor] = None
        self._gate_zero: Optional[torch.Tensor] = None

    def _reshape_heads(self, X: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, heads, N, dh]
        B, N, D = X.shape
        return X.view(B, N, self.attn_heads, self.dh).permute(0, 2, 1, 3)

    def _merge_heads(self, X: torch.Tensor) -> torch.Tensor:
        # [B, heads, N, dh] -> [B, N, D]
        B, H, N, dh = X.shape
        return X.permute(0, 2, 1, 3).contiguous().view(B, N, H * dh)

    def forward(self, embeds: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        embeds: list of [B, D] (长度 N) 或者一个 [B,N,D] 的等价列表输入
        """
        # 支持 list 输入（与原实现一致）
        if isinstance(embeds, list):
            M = torch.stack(embeds, dim=1)  # [B, N, D]
        else:
            # 兼容直接传 [B,N,D]
            M = embeds
        B, N, D = M.shape
        assert N == self.num_agents, f"Expected num_agents={self.num_agents}, got {N}"
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"

        self._latest_gates = None
        self._gate_zero = M.reshape(-1)[:0].sum()
        # Pure no-communication ablation: unchanged states and arithmetic mean.
        # No attention pooling, LayerNorm, FFN or learned gate participates.
        if self.rounds == 0:
            return M, M.mean(dim=1)

        all_gates = []

        # 多轮通信
        for _ in range(self.rounds):
            # Pre-Norm
            M_norm = self.ln_pre(M)  # [B,N,D]

            # 多头 Q/K/V
            Q = self._reshape_heads(self.q_proj(M_norm))  # [B,H,N,dh]
            K = self._reshape_heads(self.k_proj(M_norm))  # [B,H,N,dh]
            V = self._reshape_heads(self.v_proj(M_norm))  # [B,H,N,dh]

            # 注意力分数
            # attn_scores[b,h,i,j] = <Q[b,h,i], K[b,h,j]> / sqrt(dh)
            attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.dh ** 0.5)  # [B,H,N,N]

            # 可选：抑制自注意力（不看自己）
            if self.use_self_mask:
                mask = torch.eye(N, device=attn_scores.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)  # [1,1,N,N]
                attn_scores = attn_scores.masked_fill(mask, float("-inf"))

            attn = torch.softmax(attn_scores, dim=-1)          # [B,H,N,N]
            attn = self.attn_drop(attn)

            # 聚合
            agg = torch.matmul(attn, V)                         # [B,H,N,dh]
            agg = self._merge_heads(agg)                        # [B,N,D]

            # 门控融合：m_i <- LN_post( m_i + σ(gate([m_i, agg_i])) * FF(agg_i) )
            g = (torch.sigmoid(self.gate(torch.cat([M_norm, agg], dim=-1)))
                 if self.use_gate else torch.ones_like(M[..., :1]))
            upd = self.ff(agg)                                               # [B,N,D]
            M = self.ln_post(M + g * upd)                                    # 残差 + 后归一

            if self.use_gate:
                all_gates.append(g)

        # 记录门控（用于 L1 正则）
        self._latest_gates = torch.cat(all_gates, dim=1) if all_gates else None

        # 全局融合：均值或 Query-Attention
        if self.fuse_mode == "mean":
            fused = M.mean(dim=1)  # [B,D]
        else:
            # Query-Attention：用可学习查询对 agent 维度做注意力池化
            # Q_fuse: [B,1,D];  K,V: [B,N,D]
            Qf = self.fuse_query.expand(B, 1, self.dim)       # [B,1,D]
            Qh = self._reshape_heads(self.q_proj(Qf))         # [B,H,1,dh]
            Kh = self._reshape_heads(self.k_proj(M))          # [B,H,N,dh]
            Vh = self._reshape_heads(self.v_proj(M))          # [B,H,N,dh]
            scores = torch.matmul(Qh, Kh.transpose(-1, -2)) / (self.dh ** 0.5)  # [B,H,1,N]
            weights = torch.softmax(scores, dim=-1)                                              # [B,H,1,N]
            pooled = torch.matmul(weights, Vh)                                                   # [B,H,1,dh]
            pooled = self._merge_heads(pooled)                                                  # [B,1,D]
            fused = pooled.squeeze(1)                                                           # [B,D]

        return M, fused

    def gate_l1(self) -> torch.Tensor:
        """
        返回最近一次 forward 中所有样本、agent、轮次的 sigmoid 门控激活平均 L1。
        保留计算图；不是通信参数的 L1，梯度可沿门控输入回传至上游表示。
        返回标量张量，
        若未前向过则返回 0。
        """
        if self._latest_gates is None:
            if self._gate_zero is not None:
                return self._gate_zero
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._latest_gates.abs().mean()
