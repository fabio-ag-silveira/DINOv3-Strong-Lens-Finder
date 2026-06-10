"""Loss functions for the heavily imbalanced lens/non-lens problem."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * ((1 - p_t) ** self.gamma)
        if self.alpha is not None:
            a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = a_t * loss
        return loss.mean()
