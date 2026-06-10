"""Typed configuration objects (dataclasses) with YAML (de)serialization.

The YAML in configs/*.yaml maps 1:1 onto these dataclasses, giving us editor
autocomplete, defaults in one place, and validation-friendly access (cfg.train.lr
instead of cfg["train"]["lr"]).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import yaml


@dataclass
class BackboneCfg:
    source: str = "hf"                 # "hf" | "timm"
    name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    img_size: int = 224
    patch_size: int = 16
    prefix_tokens: int = 5             # CLS + register tokens (DINOv3=5, DINOv2=1)
    pretrained: bool = True


@dataclass
class FinetuneCfg:
    mode: str = "lora"                 # "frozen" | "lora" | "full"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None


@dataclass
class HeadCfg:
    hidden_dim: int = 512
    dropout: float = 0.2
    seg_head: bool = False


@dataclass
class DataCfg:
    index: str = "data/lenstronomy/index.csv"
    img_size: int = 224
    num_workers: int = 4
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class TrainCfg:
    epochs: int = 20
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    amp: bool = True
    precision: str = "bf16"            # "bf16" | "fp16"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    balance_sampler: bool = True
    seed: int = 42
    out_dir: str = "runs/exp1"


@dataclass
class EvalCfg:
    fpr_target: float = 0.1
    top_n: int = 100


@dataclass
class Config:
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    finetune: FinetuneCfg = field(default_factory=FinetuneCfg)
    head: HeadCfg = field(default_factory=HeadCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Config":
        d = d or {}
        return cls(
            backbone=BackboneCfg(**d.get("backbone", {})),
            finetune=FinetuneCfg(**d.get("finetune", {})),
            head=HeadCfg(**d.get("head", {})),
            data=DataCfg(**d.get("data", {})),
            train=TrainCfg(**d.get("train", {})),
            eval=EvalCfg(**d.get("eval", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return asdict(self)
