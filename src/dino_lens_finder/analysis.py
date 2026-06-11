"""Diagnose whether the model relies on a colour shortcut (a 'blue' cue).

Two checks on a chosen dataset + checkpoint:
  1. Correlation between the model's lens-score and each cutout's **blue excess**
     (mean blue − mean red). A strong positive correlation means the model scores
     bluer cutouts as more lens-like.
  2. Grayscale ablation: re-score the same cutouts with colour removed (luminance
     replicated to 3 channels) and compare ROC-AUC and the score↔blueness correlation.

If colour is a shortcut, the correlation is clearly positive in colour and collapses
toward zero in grayscale. Run it on the SIMULATION-trained checkpoint vs real data:

    dino-lens analyze-color --ckpt runs/exp1/best.pt --index data/lenscat/index.csv
"""
from __future__ import annotations

import csv
import json
import os
from typing import List, Sequence

import numpy as np


def _blue_excess(rgb: np.ndarray) -> float:
    a = np.asarray(rgb, dtype=float) / 255.0
    return float(a[..., 2].mean() - a[..., 0].mean())


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def color_bias_report(cfg, ckpt: str, split: str = "val",
                      out_dir: str = "runs/analysis", index: str = None) -> dict:
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .models import build_model
    from .utils import get_device, get_logger

    log = get_logger()
    if index:
        cfg.data.index = index
    device = get_device()
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()

    size = cfg.data.img_size
    norm = T.Normalize(cfg.data.mean, cfg.data.std)
    to_t = T.ToTensor()
    rows = [r for r in csv.DictReader(open(cfg.data.index)) if r["split"] == split]

    @torch.no_grad()
    def score(pil_img) -> float:
        t = norm(to_t(pil_img)).unsqueeze(0).to(device)
        return torch.sigmoid(model(t)).item()

    blue: List[float] = []
    labels: List[int] = []
    sc_color: List[float] = []
    sc_gray: List[float] = []
    for r in rows:
        img = Image.open(r["path"]).convert("RGB").resize((size, size))
        blue.append(_blue_excess(np.asarray(img)))
        labels.append(int(r["label"]))
        sc_color.append(score(img))
        sc_gray.append(score(img.convert("L").convert("RGB")))  # luminance, 3ch

    out = {
        "n": len(rows),
        "auc_color": float(roc_auc_score(labels, sc_color)),
        "auc_gray": float(roc_auc_score(labels, sc_gray)),
        "pearson_color_vs_blue": _pearson(sc_color, blue),
        "pearson_gray_vs_blue": _pearson(sc_gray, blue),
    }
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    lab = np.asarray(labels)
    for cls, c, name in [(1, "#1D9E75", "lens"), (0, "#D85A30", "non-lens")]:
        m = lab == cls
        ax[0].scatter(np.asarray(blue)[m], np.asarray(sc_color)[m], s=14, alpha=0.6,
                      c=c, label=name)
    ax[0].set_xlabel("blue excess  (mean B − mean R)")
    ax[0].set_ylabel("model lens-score")
    ax[0].set_title(f"colour: score vs blueness  (r = {out['pearson_color_vs_blue']:.2f})")
    ax[0].legend()
    ax[1].bar(["colour", "grayscale"], [out["auc_color"], out["auc_gray"]],
              color=["#185FA5", "#888780"])
    ax[1].axhline(0.5, ls="--", c="gray", lw=1)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("ROC-AUC")
    ax[1].set_title("colour-ablation AUC")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "color_bias.png")
    fig.savefig(fig_path, dpi=130)
    json.dump(out, open(os.path.join(out_dir, "color_bias.json"), "w"), indent=2)
    log.info(f"color-bias report: {out}")
    log.info(f"-> {fig_path}")
    return out
