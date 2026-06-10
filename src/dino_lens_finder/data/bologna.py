"""Loader/converter for the Bologna Strong Gravitational Lens Finding Challenge
(Metcalf et al. 2019) - the standard public benchmark for lens finding.

The challenge ships FITS cutouts plus a catalogue of labels. Because the data is
gated (free registration), this module does NOT download it: point it at your local
copy and it converts the FITS into the same PNG + index.csv layout the rest of the
pipeline consumes. That is what lets you "train on simulation, evaluate on the
benchmark" without touching the training code.

Expected input (override via arguments):
  image_dir/   FITS files, one per object, e.g. imageEUC_VIS-<ID>.fits
               (single band, or several 2-D image HDUs = bands)
  catalog.csv  a row per object with an ID column and a binary label column

Band -> RGB: if a FITS exposes >=3 two-dimensional image HDUs, the first three map
to R,G,B; a single band is replicated to three channels. An asinh stretch with
percentile clipping mimics how lenses are inspected visually.
"""
from __future__ import annotations

import csv
import glob
import os
from typing import List, Optional

import numpy as np


def _read_fits_bands(path: str) -> List[np.ndarray]:
    from astropy.io import fits

    bands: List[np.ndarray] = []
    with fits.open(path) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is not None and np.ndim(data) == 2:
                bands.append(np.asarray(data, dtype=float))
    return bands


def _stretch(x: np.ndarray, q=(1.0, 99.5), a: float = 10.0) -> np.ndarray:
    lo, hi = np.percentile(x, q)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return np.arcsinh(a * x) / np.arcsinh(a)


def fits_to_rgb(path: str, size: Optional[int] = None) -> np.ndarray:
    """Convert a (possibly multi-band) FITS cutout to an (H, W, 3) uint8 image."""
    bands = _read_fits_bands(path)
    if not bands:
        raise ValueError(f"No 2-D image HDU found in {path}")
    chans = bands[:3] if len(bands) >= 3 else [bands[0]] * 3
    rgb = np.stack([_stretch(c) for c in chans], axis=-1)
    img = (rgb * 255).astype(np.uint8)
    if size is not None:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((size, size)))
    return img


def build_bologna_index(image_dir: str, catalog_csv: str, out_dir: str,
                        id_col: str = "ID", label_col: str = "is_lens",
                        pattern: str = "*{id}*.fits", val_frac: float = 0.2,
                        size: Optional[int] = None, seed: int = 42) -> str:
    """Convert a local Bologna challenge copy into PNG + index.csv. Returns the
    index path. Objects whose FITS cannot be found are skipped with a warning."""
    from ..utils import write_png_dataset

    rng = np.random.default_rng(seed)
    labels = {}
    with open(catalog_csv) as f:
        for row in csv.DictReader(f):
            labels[str(row[id_col])] = int(float(row[label_col]))

    items, missing = [], 0
    for oid, label in labels.items():
        matches = glob.glob(os.path.join(image_dir, pattern.replace("{id}", oid)))
        if not matches:
            missing += 1
            continue
        img = fits_to_rgb(matches[0], size=size)
        split = "val" if rng.random() < val_frac else "train"
        items.append((split, label, img, str(oid)))

    index, rows = write_png_dataset(out_dir, items)
    n_lens = sum(r["label"] for r in rows)
    print(f"Bologna: wrote {len(rows)} images ({n_lens} lenses); "
          f"{missing} catalogue entries had no FITS -> {index}")
    return index
