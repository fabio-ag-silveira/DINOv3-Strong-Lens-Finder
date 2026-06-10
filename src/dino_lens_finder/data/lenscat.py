"""Build a REAL strong-lens benchmark without the (unreachable) Bologna server.

Positives  : confirmed/probable lenses from the **lenscat** catalogue
             (HF: juliensimon/gravitational-lenses, 32,838 real systems).
Images     : grz colour cutouts from the **Legacy Survey** viewer (US-hosted,
             globally reachable) at each lens RA/Dec.
Negatives  : cutouts at random sky positions in the same footprint - strong lenses
             are rare, so random fields are overwhelmingly non-lenses.

Both hosts are reliable and reachable from anywhere (unlike the Bologna portal), so
this gives a genuine sim->real test. Honest caveats (see docs/lenscat.md): lenscat is
heterogeneous (many surveys/resolutions) and the negatives are random fields, so it
is a noisier benchmark than Bologna.

Needs the 'benchmark' extra:  pip install -e ".[benchmark]"   (pandas, pyarrow, requests)
"""
from __future__ import annotations

import io
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import write_png_dataset

CATALOG_URL = ("https://huggingface.co/datasets/juliensimon/gravitational-lenses/"
               "resolve/main/data/gravitational_lenses.parquet")
CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.jpg"


def _load_catalog(path: Optional[str] = None) -> List[dict]:
    """Return the catalogue as a list of dict rows (keys: name, ra_deg, dec_deg,
    lens_type, grading, ...). Downloads from HF unless a local file is given."""
    if path and path.endswith(".csv"):
        import csv
        with open(path) as f:
            return list(csv.DictReader(f))
    import pandas as pd
    return pd.read_parquet(path or CATALOG_URL).to_dict("records")


def _fetch_cutout(ra: float, dec: float, layer: str, pixscale: float,
                  size: int, timeout: int = 30) -> np.ndarray:
    import requests
    from PIL import Image
    params = {"ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "layer": layer,
              "pixscale": pixscale, "size": size}
    r = requests.get(CUTOUT_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))


def _is_blank(img: np.ndarray, std_thresh: float = 3.0) -> bool:
    """Out-of-footprint cutouts come back uniform/near-constant -> drop them."""
    return float(np.asarray(img, dtype=float).std()) < std_thresh


def _collect(coords: Iterable[Tuple[str, float, float]], label: int, n: int,
             layer: str, pixscale: float, size: int, sleep: float,
             log_every: int = 50) -> List[Tuple[int, np.ndarray, str]]:
    items: List[Tuple[int, np.ndarray, str]] = []
    for name, ra, dec in coords:
        if len(items) >= n:
            break
        try:
            img = _fetch_cutout(ra, dec, layer, pixscale, size)
        except Exception:
            continue
        if _is_blank(img):
            continue
        items.append((label, img, name))
        if len(items) % log_every == 0:
            print(f"  label {label}: {len(items)}/{n}")
        if sleep:
            time.sleep(sleep)
    return items


def build_lenscat_dataset(out_dir: str, n_per_class: int = 300,
                          layer: str = "ls-dr10", pixscale: float = 0.262,
                          size: int = 101,
                          grading: Sequence[str] = ("confident", "probable"),
                          lens_type: str = "galaxy", val_frac: float = 0.2,
                          seed: int = 42, catalog_path: Optional[str] = None,
                          sleep: float = 0.05) -> str:
    """Fetch real lens cutouts (positives) + random-field cutouts (negatives) and
    write the standard PNG + index.csv. Returns the index path."""
    rng = np.random.default_rng(seed)
    rows = _load_catalog(catalog_path)

    def _to_float(v):
        try:
            x = float(v)
            return None if np.isnan(x) else x
        except (TypeError, ValueError):
            return None

    positives: List[Tuple[str, float, float]] = []
    for r in rows:
        if lens_type and str(r.get("lens_type")) != lens_type:
            continue
        if grading and str(r.get("grading")) not in grading:
            continue
        ra, dec = _to_float(r.get("ra_deg")), _to_float(r.get("dec_deg"))
        if ra is None or dec is None:
            continue
        positives.append((str(r.get("name")), ra, dec))
    rng.shuffle(positives)
    print(f"lenscat: {len(positives)} catalogue positives match filters; "
          f"fetching up to {n_per_class}/class from {layer} (this hits the network)...")

    pos_items = _collect(positives, 1, n_per_class, layer, pixscale, size, sleep)

    def neg_coords():
        i = 0
        while True:
            i += 1
            yield (f"rand{i:06d}", float(rng.uniform(0, 360)), float(rng.uniform(-25, 32)))

    neg_items = _collect(neg_coords(), 0, len(pos_items), layer, pixscale, size, sleep)

    all_items = pos_items + neg_items

    def with_split():
        for i, (label, img, _name) in enumerate(all_items):
            split = "val" if rng.random() < val_frac else "train"
            yield split, label, img, f"{i:06d}"

    index, written = write_png_dataset(out_dir, with_split())
    n_lens = sum(w["label"] for w in written)
    print(f"lenscat: wrote {len(written)} images "
          f"({n_lens} lenses, {len(written) - n_lens} non-lenses) -> {index}")
    if n_lens < n_per_class:
        print("  note: fewer positives than requested (footprint/coverage); "
              "try --layer ls-dr9 (north), add 'probable' grading, or lower --n-per-class.")
    return index
