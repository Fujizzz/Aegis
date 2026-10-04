"""One real encoder + one task head, using the existing recovery file layout.

Inactive slots contain no encoder and no trainable parameters. They preserve the
eight-component archive contract, not four-agent computation or supervision.
"""
import torch
from torch import nn


class InactiveAgent(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.classifier = nn.Linear(1, 1, bias=False).to(device)
        self.requires_grad_(False)

    def forward(self, *args):
        raise RuntimeError('Inactive single-agent checkpoint slot must never encode input')

    def head_on_embed(self, embedding):
        return embedding.new_zeros((len(embedding), 1))


class SingleFusion(nn.Module):
    rounds = 0
    use_gate = False

    def __init__(self, device):
        super().__init__()
        self.dtype_anchor = nn.Parameter(torch.zeros((), device=device), requires_grad=False)

    def forward(self, embeddings):
        first = embeddings[0]
        # The compatibility states are never used for supervision or communication.
        return torch.stack([first] * 4, dim=1), first

    def gate_l1(self):
        return self.dtype_anchor * 0
