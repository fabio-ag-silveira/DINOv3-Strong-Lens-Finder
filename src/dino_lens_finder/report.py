"""Visualization helpers for the results notebook and portfolio figures.

Pure matplotlib + PIL (no torch), so they can be unit-tested and reused. Install
with the "notebook" extra:  pip install -e ".[notebook]".
"""
from __future__ import annotations

import csv
from typing import List, Sequence, Tuple

import numpy as np


def read_ranked(csv_path: str) -> List[Tuple[int, float, int, str]]:
    """Load a ranked_candidates.csv (rank,score,label,path) written by `eval`."""
    rows: List[Tuple[int, float, int, str]] = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["rank"]), float(r["score"]), int(r["label"]), r["path"]))
    return rows


def roc_figure(y_true: Sequence[int], scores: Sequence[float], fpr_target: float = 0.1):
    """ROC curve annotated with AUC and the operating point TPR@FPR."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = float(roc_auc_score(y_true, scores))
    tpr_at = float(np.interp(fpr_target, fpr, tpr))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.axvline(fpr_target, color="crimson", ls=":", lw=1)
    ax.scatter([fpr_target], [tpr_at], color="crimson", zorder=5,
               label=f"TPR@FPR={fpr_target:g} = {tpr_at:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Strong-lens finding ROC"); ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def ranked_grid(rows: Sequence[Tuple[int, float, int, str]], n: int = 16,
                cols: int = 4, title: str = "Top-ranked candidates"):
    """Grid of the top-N ranked cutouts. Green border = true lens, red = false
    positive — i.e. a visual read of precision among the candidates a human inspects."""
    import matplotlib.pyplot as plt
    from PIL import Image

    if not rows:
        raise ValueError("ranked_grid: no rows to plot")
    rows = sorted(rows, key=lambda t: t[0])[:n]
    nrows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2.2, nrows * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (rank, score, label, path) in zip(axes, rows):
        ax.imshow(Image.open(path))
        color = "limegreen" if label == 1 else "red"
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   fill=False, edgecolor=color, linewidth=4))
        ax.set_title(f"#{rank}  p={score:.2f}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
