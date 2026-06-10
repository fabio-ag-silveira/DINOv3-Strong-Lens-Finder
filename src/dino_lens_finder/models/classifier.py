"""Lens classifier = backbone + lightweight head(s)."""
from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

from ..config import Config
from .backbone import Backbone, apply_lora, set_trainable


class SegHead(nn.Module):
    """Optional decoder over dense patch tokens to localize arcs/rings."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, patches: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        b, n, d = patches.shape
        h, w = grid
        x = patches.transpose(1, 2).reshape(b, d, h, w)
        return self.conv(x)                                   # (B, 1, h, w) logits


class LensClassifier(nn.Module):
    def __init__(self, backbone: Backbone, embed_dim: int,
                 hidden_dim: int = 512, dropout: float = 0.2, seg_head: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.seg = SegHead(embed_dim) if seg_head else None

    def forward(self, x: torch.Tensor, return_seg: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        f = self.backbone(x)
        logit = self.head(f["cls"]).squeeze(-1)               # (B,)
        if return_seg and self.seg is not None and f["patches"] is not None:
            return logit, self.seg(f["patches"], f["grid"])
        return logit


def build_model(cfg: Config) -> LensClassifier:
    backbone = Backbone(cfg.backbone)
    if cfg.finetune.mode == "lora":
        backbone = apply_lora(backbone, cfg.finetune)
    else:
        backbone = set_trainable(backbone, cfg.finetune.mode)
    return LensClassifier(
        backbone, embed_dim=backbone.embed_dim,
        hidden_dim=cfg.head.hidden_dim, dropout=cfg.head.dropout,
        seg_head=cfg.head.seg_head,
    )
