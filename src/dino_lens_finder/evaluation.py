"""Evaluate a checkpoint and dump a ranked candidate list (the deliverable a human
inspector actually uses)."""
from __future__ import annotations

import csv
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import LensDataset, build_transforms
from .models import build_model
from .training.metrics import compute_metrics
from .utils import get_device, get_logger


def evaluate_checkpoint(cfg: Config, ckpt: str, split: str = "val",
                        out: str = "runs/exp1/ranked_candidates.csv",
                        index: Optional[str] = None) -> dict:
    log = get_logger()
    if index:
        cfg.data.index = index
    device = get_device()
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()

    d = cfg.data
    ds = LensDataset(d.index, split, build_transforms(d.img_size, d.mean, d.std, False))
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=d.num_workers)

    ys, ss, paths = [], [], []
    with torch.no_grad():
        for x, y, p in loader:
            ss.extend(torch.sigmoid(model(x.to(device))).float().cpu().tolist())
            ys.extend(y.tolist())
            paths.extend(p)

    m = compute_metrics(ys, ss, cfg.eval.fpr_target, cfg.eval.top_n)
    log.info(f"Metrics: {m}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    order = sorted(range(len(ss)), key=lambda i: -ss[i])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "score", "label", "path"])
        for r, i in enumerate(order):
            w.writerow([r, f"{ss[i]:.4f}", ys[i], paths[i]])
    log.info(f"Ranked candidates -> {out}")
    return m
