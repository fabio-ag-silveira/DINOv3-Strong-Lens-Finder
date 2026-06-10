"""Ranking-oriented metrics. Lens finding is a needle-in-a-haystack search, so
accuracy is meaningless; we report ROC-AUC, TPR at a fixed FPR, and precision among
the top-N ranked candidates (what a human actually inspects)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(y_true: Sequence[int], scores: Sequence[float], fpr_target: float = 0.1) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(fpr_target, fpr, tpr))


def precision_at_n(y_true: Sequence[int], scores: Sequence[float], n: int = 100) -> float:
    order = np.argsort(-np.asarray(scores))
    n = min(n, len(order))
    return float(np.asarray(y_true)[order[:n]].mean()) if n else float("nan")


def compute_metrics(y_true: List[int], scores: List[float],
                    fpr_target: float = 0.1, top_n: int = 100) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    out: Dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["tpr_at_fpr"] = tpr_at_fpr(y_true, scores, fpr_target)
    else:
        out["roc_auc"] = float("nan")
        out["tpr_at_fpr"] = float("nan")
    out[f"precision@{top_n}"] = precision_at_n(y_true, scores, top_n)
    return out
