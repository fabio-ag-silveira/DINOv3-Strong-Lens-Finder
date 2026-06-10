"""Cross-cutting helpers: reproducibility, device, logging, dataset writing."""
from __future__ import annotations

import csv
import logging
import os
import random

import numpy as np
from typing import Iterable, List, Tuple



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logger(name: str = "dino_lens") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def write_png_dataset(out_dir: str,
                      items: Iterable[Tuple[str, int, object, str]]) -> Tuple[str, List[dict]]:
    """Write (split, label, uint8_image, name) tuples to disk as
    ``out_dir/<split>/<class>/<name>.png`` plus an ``index.csv`` the pipeline reads.
    """
    from PIL import Image

    rows: List[dict] = []
    for split, label, img, name in items:
        cls = "lens" if int(label) == 1 else "nonlens"
        d = os.path.join(out_dir, split, cls)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{name}.png")
        Image.fromarray(img).save(path)
        rows.append({"path": path, "label": int(label), "split": split})

    os.makedirs(out_dir, exist_ok=True)
    index = os.path.join(out_dir, "index.csv")
    with open(index, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        w.writeheader()
        w.writerows(rows)
    return index, rows
