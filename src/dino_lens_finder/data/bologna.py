"""Loader/converter for the Bologna Strong Gravitational Lens Finding Challenge
(Metcalf et al. 2019, A&A 625, A119) - the standard public lens-finding benchmark.

The challenge has two tracks, both with 101x101 px cutouts and a truth "key":
  * space-based  : 1 band (Euclid VIS-like, ~0.1"/px)  -> one FITS per object
  * ground-based : 4 bands (KiDS-like u/g/r/i, ~0.2"/px) -> one FITS per band

This module converts a local copy into the same PNG + index.csv layout the rest of
the pipeline consumes, so "train on simulation, evaluate on the benchmark" needs no
code changes - just repoint data.index. The data is gated (free registration), so
nothing is downloaded here. See docs/bologna.md.

Label: pass a binary column directly (label_col), or derive it from a numeric
quality column with `label_threshold` (e.g. number of lensed-source pixels), which
reproduces the challenge's "detectable lens" definition.
"""
from __future__ import annotations

import csv
import glob
import os
from typing import List, Optional, Sequence, Union

import numpy as np


def _first_2d(path: str) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path) as hdul:
        for hdu in hdul:
            d = getattr(hdu, "data", None)
            if d is not None and np.ndim(d) == 2:
                return np.asarray(d, dtype=float)
    raise ValueError(f"No 2-D image HDU in {path}")


def _read_bands(src: Union[str, Sequence[str]]) -> List[np.ndarray]:
    """Return a list of 2-D band arrays. `src` is one FITS (read all 2-D HDUs) or
    a list of per-band FITS paths (read the first 2-D HDU of each)."""
    if isinstance(src, (list, tuple)):
        return [_first_2d(p) for p in src]
    from astropy.io import fits
    bands: List[np.ndarray] = []
    with fits.open(src) as hdul:
        for hdu in hdul:
            d = getattr(hdu, "data", None)
            if d is not None and np.ndim(d) == 2:
                bands.append(np.asarray(d, dtype=float))
    if not bands:
        raise ValueError(f"No 2-D image HDU in {src}")
    return bands


def _stretch(x: np.ndarray, q=(1.0, 99.5), a: float = 10.0) -> np.ndarray:
    lo, hi = np.percentile(x, q)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return np.arcsinh(a * x) / np.arcsinh(a)


def fits_to_rgb(src: Union[str, Sequence[str]], size: Optional[int] = None) -> np.ndarray:
    """Convert FITS (single multi-band file or per-band files) to (H, W, 3) uint8."""
    bands = _read_bands(src)
    chans = bands[:3] if len(bands) >= 3 else [bands[0]] * 3
    rgb = np.stack([_stretch(c) for c in chans], axis=-1)
    img = (rgb * 255).astype(np.uint8)
    if size is not None:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((size, size)))
    return img


def build_bologna_index(image_dir: str, catalog_csv: str, out_dir: str,
                        id_col: str = "ID", label_col: str = "is_lens",
                        pattern: str = "*{id}*.fits",
                        band_patterns: Optional[Sequence[str]] = None,
                        label_threshold: Optional[float] = None,
                        val_frac: float = 0.2, size: Optional[int] = None,
                        seed: int = 42) -> str:
    """Convert a local Bologna copy into PNG + index.csv; returns the index path.

    band_patterns : per-band filename patterns (each containing '{id}') for the
                    ground-based multi-file layout, composed into RGB. If None,
                    the single `pattern` is used (space-based, or multi-HDU files).
    label_threshold : if set, label = int(value > threshold); else int(value).
    """
    from ..utils import write_png_dataset

    rng = np.random.default_rng(seed)
    catalog = {}
    with open(catalog_csv) as f:
        for row in csv.DictReader(f):
            catalog[str(row[id_col])] = row[label_col]

    def resolve(oid: str):
        if band_patterns:
            paths = []
            for pat in band_patterns:
                m = glob.glob(os.path.join(image_dir, pat.replace("{id}", oid)))
                if not m:
                    return None
                paths.append(sorted(m)[0])
            return paths
        m = glob.glob(os.path.join(image_dir, pattern.replace("{id}", oid)))
        return sorted(m)[0] if m else None

    def to_label(raw: str) -> int:
        return int(float(raw) > label_threshold) if label_threshold is not None else int(float(raw))

    items, missing = [], 0
    for oid, raw in catalog.items():
        src = resolve(oid)
        if src is None:
            missing += 1
            continue
        items.append((("val" if rng.random() < val_frac else "train"),
                      to_label(raw), fits_to_rgb(src, size=size), str(oid)))

    index, rows = write_png_dataset(out_dir, items)
    n_lens = sum(r["label"] for r in rows)
    print(f"Bologna: wrote {len(rows)} images ({n_lens} lenses); "
          f"{missing} catalogue entries had no FITS -> {index}")
    return index
