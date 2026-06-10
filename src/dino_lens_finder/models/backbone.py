"""Feature-extraction backbone with a uniform interface across DINOv3/DINOv2
(HF transformers) and timm models.

    forward(x) -> {"cls": (B, D), "patches": (B, N, D) | None, "grid": (h, w) | None}

The timm path is a lightweight fallback (no gated weights / downloads) used by the
smoke test to validate the pipeline on CPU.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import BackboneCfg, Config, FinetuneCfg


class Backbone(nn.Module):
    def __init__(self, cfg: BackboneCfg):
        super().__init__()
        self.cfg = cfg
        self.source = cfg.source
        self.prefix_tokens = cfg.prefix_tokens
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size

        if cfg.source == "hf":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(cfg.name)
            self.embed_dim = self.model.config.hidden_size
            self.patch_size = getattr(self.model.config, "patch_size", cfg.patch_size)
        elif cfg.source == "timm":
            import timm
            try:
                self.model = timm.create_model(
                    cfg.name, pretrained=cfg.pretrained, num_classes=0,
                    img_size=cfg.img_size)
            except TypeError:                       # ConvNeXt etc. lack img_size kwarg
                self.model = timm.create_model(
                    cfg.name, pretrained=cfg.pretrained, num_classes=0)
            self.embed_dim = self.model.num_features
        else:
            raise ValueError(f"Unknown backbone source: {cfg.source!r}")

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        if self.source == "hf":
            h = self.model(pixel_values=x).last_hidden_state   # (B, 1+R+N, D)
            grid = self.img_size // self.patch_size
            return {"cls": h[:, 0], "patches": h[:, self.prefix_tokens:],
                    "grid": (grid, grid)}
        feat = self.model(x)                                   # (B, D) global-pooled
        return {"cls": feat, "patches": None, "grid": None}


def apply_lora(backbone: Backbone, cfg: FinetuneCfg) -> Backbone:
    """Wrap backbone weights with LoRA adapters (only adapters stay trainable)."""
    from peft import LoraConfig, get_peft_model

    targets = cfg.target_modules
    if targets is None:
        targets = (["query", "key", "value", "dense"]
                   if backbone.source == "hf" else ["qkv", "proj"])
    lconf = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                       lora_dropout=cfg.lora_dropout, target_modules=targets,
                       bias="none")
    backbone.model = get_peft_model(backbone.model, lconf)
    return backbone


def set_trainable(backbone: Backbone, mode: str) -> Backbone:
    if mode == "frozen":
        for p in backbone.parameters():
            p.requires_grad = False
    elif mode == "full":
        for p in backbone.parameters():
            p.requires_grad = True
    return backbone


def check_backbone(cfg: Config) -> None:
    """Load the configured backbone and assert the token layout is consistent.
    Run after `hf auth login` + accepting the model licence."""
    bb = Backbone(cfg.backbone).eval()
    x = torch.randn(2, 3, cfg.backbone.img_size, cfg.backbone.img_size)
    with torch.no_grad():
        out = bb(x)
    print(f"backbone  : {cfg.backbone.source} :: {cfg.backbone.name}")
    print(f"embed_dim : {bb.embed_dim}")
    print(f"cls       : {tuple(out['cls'].shape)}")
    if out["patches"] is not None:
        exp = out["grid"][0] * out["grid"][1]
        print(f"patches   : {tuple(out['patches'].shape)}  grid {out['grid']}  (expect {exp})")
        assert out["patches"].shape[1] == exp, (
            "patch count != grid*grid -> fix backbone.prefix_tokens "
            "(DINOv3=5 with registers, DINOv2=1).")
    print("OK: backbone loads and shapes are consistent.")
