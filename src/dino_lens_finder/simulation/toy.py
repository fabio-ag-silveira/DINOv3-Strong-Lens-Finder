"""Lightweight, dependency-free lens-cutout generator.

This is a TOY (analytic arcs, not a lensing forward model). Its only job is to let
the full pipeline run end-to-end on CPU without heavy dependencies (used by the
smoke test). For real training data use simulation.lenstronomy_sim.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ..utils import write_png_dataset


def _sersic(yy, xx, x0, y0, amp, reff, q, theta, n=2.0):
    ct, st = np.cos(theta), np.sin(theta)
    xr = (xx - x0) * ct + (yy - y0) * st
    yr = -(xx - x0) * st + (yy - y0) * ct
    r = np.sqrt(xr ** 2 + (yr / q) ** 2)
    bn = 2 * n - 1 / 3
    return amp * np.exp(-bn * ((r / reff) ** (1.0 / n) - 1))


def _arc(yy, xx, cx, cy, radius, width, span, phi0, amp):
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    dang = np.angle(np.exp(1j * (np.arctan2(dy, dx) - phi0)))
    return amp * np.exp(-0.5 * ((r - radius) / width) ** 2) * np.exp(-0.5 * (dang / span) ** 2)


def generate_cutout(size: int = 128, lens: bool = True, rng=None, seed=None) -> np.ndarray:
    """Return an (size, size, 3) uint8 cutout."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cx, cy = size / 2 + rng.uniform(-3, 3), size / 2 + rng.uniform(-3, 3)
    img = np.zeros((size, size, 3), dtype=float)

    gal = _sersic(yy, xx, cx, cy, amp=rng.uniform(0.8, 1.2),
                  reff=rng.uniform(size * 0.06, size * 0.12),
                  q=rng.uniform(0.6, 0.95), theta=rng.uniform(0, np.pi), n=rng.uniform(2, 4))
    img[..., 0] += 1.0 * gal; img[..., 1] += 0.8 * gal; img[..., 2] += 0.6 * gal

    for _ in range(int(rng.integers(0, 4))):
        blob = _sersic(yy, xx, rng.uniform(0, size), rng.uniform(0, size),
                       amp=rng.uniform(0.2, 0.6), reff=rng.uniform(2, 6),
                       q=rng.uniform(0.5, 1), theta=rng.uniform(0, np.pi), n=1)
        img += blob[..., None] * rng.uniform(0.4, 1.0, 3)

    if lens:
        rE = rng.uniform(size * 0.18, size * 0.30)
        for _ in range(int(rng.integers(1, 3))):
            arc = _arc(yy, xx, cx, cy, radius=rE, width=rng.uniform(1.5, 3.0),
                       span=rng.uniform(0.3, 0.8), phi0=rng.uniform(0, 2 * np.pi),
                       amp=rng.uniform(0.5, 1.0))
            img[..., 2] += 1.0 * arc; img[..., 1] += 0.7 * arc; img[..., 0] += 0.4 * arc

    for c in range(3):
        img[..., c] = gaussian_filter(img[..., c], rng.uniform(0.8, 1.6))
    img += rng.normal(0, 0.02, img.shape)
    img = np.clip(img, 0, None)
    img = np.arcsinh(img / (img.std() + 1e-6))
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    return (img * 255).astype(np.uint8)


def build_dataset(out_dir: str, n_train: int = 400, n_val: int = 100,
                  pos_frac: float = 0.3, size: int = 128, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)

    def items():
        for split, n in [("train", n_train), ("val", n_val)]:
            for i in range(n):
                is_lens = bool(rng.random() < pos_frac)
                yield split, int(is_lens), generate_cutout(size, is_lens, rng), f"{i:05d}"

    index, rows = write_png_dataset(out_dir, items())
    print(f"toy: wrote {len(rows)} images ({sum(r['label'] for r in rows)} lenses) -> {index}")
    return index
