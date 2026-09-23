"""Metric-learning losses on L2-normalised embeddings.

Both losses operate on cosine similarity ``s = a·b`` (cosine distance
``d = 1 - s``) and use *all* pairs in a batch, so they pair naturally with
P×K sampling (P groups × K listings per group).
"""

from __future__ import annotations

import torch
from torch import nn


def _pair_masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    return same & ~eye, ~same


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss (Hermans et al., 2017) with cosine similarity.

    For every anchor: hardest positive = least similar same-group item,
    hardest negative = most similar other-group item;
    ``loss = relu(s_neg - s_pos + margin)``.
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = embeddings @ embeddings.T
        pos_mask, neg_mask = _pair_masks(labels)
        valid = pos_mask.any(1) & neg_mask.any(1)
        if not valid.any():
            return embeddings.sum() * 0.0
        hardest_pos = sim.masked_fill(~pos_mask, float("inf")).min(1).values
        hardest_neg = sim.masked_fill(~neg_mask, float("-inf")).max(1).values
        loss = torch.relu(hardest_neg - hardest_pos + self.margin)
        return loss[valid].mean()


class ContrastiveLoss(nn.Module):
    """Pairwise contrastive loss (Hadsell et al., 2006) on cosine distance.

    Positive pairs are pulled together (``d²``); negative pairs are pushed
    until their distance exceeds ``margin`` (``relu(margin - d)²``). The two
    terms are averaged separately so class imbalance does not swamp positives.
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = 1.0 - embeddings @ embeddings.T
        pos_mask, neg_mask = _pair_masks(labels)
        zero = embeddings.sum() * 0.0
        pos = dist[pos_mask].pow(2).mean() if pos_mask.any() else zero
        neg = torch.relu(self.margin - dist[neg_mask]).pow(2).mean() if neg_mask.any() else zero
        return pos + neg


def build_loss(name: str, margin: float) -> nn.Module:
    """Factory for the configured loss."""
    if name == "triplet":
        return BatchHardTripletLoss(margin)
    if name == "contrastive":
        return ContrastiveLoss(margin)
    raise ValueError(f"unknown loss: {name}")
