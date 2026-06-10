"""Lightweight tests. The metric/simulation tests run anywhere; the end-to-end
training test is skipped automatically if torch is not installed."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_metrics_perfect_separation():
    from dino_lens_finder.training.metrics import compute_metrics
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.1, 2)
    assert m["roc_auc"] == 1.0
    assert m["precision@2"] == 1.0


def test_toy_generator_shape():
    from dino_lens_finder.simulation.toy import generate_cutout
    img = generate_cutout(size=48, lens=True, seed=0)
    assert img.shape == (48, 48, 3) and img.dtype == np.uint8


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_end_to_end_cpu(tmp_path):
    import torch
    from torch.utils.data import DataLoader
    from dino_lens_finder.config import Config
    from dino_lens_finder.data.dataset import LensDataset, build_transforms, make_weighted_sampler
    from dino_lens_finder.models import build_model
    from dino_lens_finder.training.losses import BinaryFocalLoss
    from dino_lens_finder.simulation.toy import build_dataset

    idx = build_dataset(str(tmp_path / "syn"), n_train=24, n_val=8, size=64, pos_frac=0.4)
    cfg = Config.from_dict({
        "backbone": {"source": "timm", "name": "vit_tiny_patch16_224",
                     "img_size": 64, "prefix_tokens": 1, "pretrained": False},
        "finetune": {"mode": "frozen"},
        "head": {"hidden_dim": 64},
        "data": {"index": idx, "img_size": 64, "num_workers": 0,
                 "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "train": {"batch_size": 8},
    })
    model = build_model(cfg)
    tf = build_transforms(64, [0.5] * 3, [0.5] * 3, True)
    ds = LensDataset(idx, "train", tf)
    loader = DataLoader(ds, batch_size=8, sampler=make_weighted_sampler(ds.labels()))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    crit = BinaryFocalLoss()
    model.train()
    for x, y, _ in loader:
        loss = crit(model(x), y.float())
        opt.zero_grad(); loss.backward(); opt.step()
    assert torch.isfinite(loss)
