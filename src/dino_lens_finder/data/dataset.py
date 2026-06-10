"""Dataset, augmentation, and an imbalance-aware sampler.

Cutouts are 3-channel composites (survey bands -> RGB) so they plug straight into
DINOv3. For >3 bands, adapt the patch-embedding conv (see docs/usage.md)."""
from __future__ import annotations

import csv
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T


def build_transforms(img_size: int, mean: List[float], std: List[float],
                     train: bool = True) -> T.Compose:
    aug = [T.Resize((img_size, img_size))]
    if train:
        # Lenses are invariant to flips & 90-deg rotations -> label-preserving aug.
        aug += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.RandomApply([T.RandomRotation((90, 90))], p=0.5)]
    aug += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(aug)


class LensDataset(Dataset):
    """Reads an index CSV with columns: path,label,split."""

    def __init__(self, index_csv: str, split: str, transform: T.Compose):
        self.items: List[Tuple[str, int]] = []
        with open(index_csv) as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.items.append((row["path"], int(row["label"])))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def labels(self) -> List[int]:
        return [lab for _, lab in self.items]

    def __getitem__(self, i: int):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, path


def make_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    arr = np.asarray(labels)
    counts = np.bincount(arr, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    sample_w = (1.0 / counts)[arr]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(labels), replacement=True,
    )
