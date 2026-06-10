"""Physically-motivated strong-lens simulator built on lenstronomy.

A singular isothermal ellipsoid (SIE) + external shear deflects a Sersic background
source into arcs/Einstein rings, on top of a Sersic deflector galaxy. Three light
components are composed into an RGB cutout:

  * deflector galaxy + red neighbours (unlensed)        -> mostly RED
  * unlensed blue companions (hard negatives, BOTH classes)
  * the lensed background source (arcs / ring)          -> mostly BLUE  [lens only]

Positives and negatives differ ONLY by the lensed source, and blue companions
appear in both classes, so the model must learn arc morphology, not colour.

Requires the "sim" extra:  pip install -e ".[sim]"   (lenstronomy + astropy)
"""
from __future__ import annotations

import numpy as np

from ..utils import write_png_dataset


def _ell(rng, emax=0.35):
    while True:
        e1, e2 = rng.uniform(-emax, emax, 2)
        if e1 * e1 + e2 * e2 < emax * emax:
            return float(e1), float(e2)


def _sersic(rng, amp, x, y, R, n, emax=0.3):
    e1, e2 = _ell(rng, emax)
    return {"amp": amp, "R_sersic": R, "n_sersic": n,
            "e1": e1, "e2": e2, "center_x": x, "center_y": y}


def generate_lens_cutout(size: int = 64, deltaPix: float = 0.1, lens: bool = True,
                         rng=None, seed=None) -> np.ndarray:
    """Return an (size, size, 3) uint8 RGB cutout.

    deltaPix: arcsec/pixel (~0.05 space-based, 0.1 Euclid VIS, 0.2 ground)."""
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    import lenstronomy.Util.simulation_util as sim_util
    import lenstronomy.Util.image_util as image_util

    rng = rng if rng is not None else np.random.default_rng(seed)
    exp_time = float(rng.uniform(90, 200))
    bkg_rms = float(rng.uniform(0.03, 0.08))

    data = ImageData(**sim_util.data_configure_simple(size, deltaPix, exp_time, bkg_rms))
    psf = PSF(psf_type="GAUSSIAN", fwhm=float(rng.uniform(0.6, 1.6)) * deltaPix * 2.0,
              pixel_size=deltaPix)
    numerics = {"supersampling_factor": 1}

    theta_E = float(rng.uniform(0.7, 1.8))
    e1, e2 = _ell(rng, 0.3)
    kwargs_lens = [
        {"theta_E": theta_E, "e1": e1, "e2": e2, "center_x": 0.0, "center_y": 0.0},
        {"gamma1": float(rng.uniform(-0.05, 0.05)),
         "gamma2": float(rng.uniform(-0.05, 0.05)), "ra_0": 0, "dec_0": 0},
    ]
    lens_model = LensModel(["SIE", "SHEAR"])

    def render_unlensed(kwargs_light):
        lm = LightModel(["SERSIC_ELLIPSE"] * len(kwargs_light))
        im = ImageModel(data, psf, lens_model_class=lens_model,
                        lens_light_model_class=lm, kwargs_numerics=numerics)
        return im.image(kwargs_lens, None, kwargs_light, source_add=False,
                        lens_light_add=True, point_source_add=False)

    def render_lensed(kwargs_src):
        sm = LightModel(["SERSIC_ELLIPSE"])
        im = ImageModel(data, psf, lens_model_class=lens_model,
                        source_model_class=sm, kwargs_numerics=numerics)
        return im.image(kwargs_lens, kwargs_src, None, source_add=True,
                        lens_light_add=False, point_source_add=False)

    fov = size * deltaPix * 0.45
    gal_kwargs = [_sersic(rng, amp=rng.uniform(20, 50),
                          x=rng.uniform(-0.05, 0.05), y=rng.uniform(-0.05, 0.05),
                          R=rng.uniform(0.25, 0.5), n=rng.uniform(3, 5))]
    for _ in range(int(rng.integers(0, 3))):
        gal_kwargs.append(_sersic(rng, amp=rng.uniform(5, 20),
                                  x=rng.uniform(-fov, fov), y=rng.uniform(-fov, fov),
                                  R=rng.uniform(0.1, 0.3), n=rng.uniform(1, 4)))
    gal = render_unlensed(gal_kwargs)

    blue = np.zeros_like(gal)
    if rng.random() < 0.5:
        blue_kwargs = [_sersic(rng, amp=rng.uniform(5, 15),
                               x=rng.uniform(-fov, fov), y=rng.uniform(-fov, fov),
                               R=rng.uniform(0.05, 0.15), n=1.0)
                       for _ in range(int(rng.integers(1, 3)))]
        blue = render_unlensed(blue_kwargs)

    arc = np.zeros_like(gal)
    if lens:
        r = rng.uniform(0, 0.3 * theta_E)
        phi = rng.uniform(0, 2 * np.pi)
        src = [_sersic(rng, amp=rng.uniform(15, 40), x=r * np.cos(phi), y=r * np.sin(phi),
                       R=rng.uniform(0.05, 0.15), n=rng.uniform(1, 1.5), emax=0.2)]
        arc = render_lensed(src)

    R = 1.0 * gal + 0.35 * arc + 0.25 * blue
    G = 0.6 * gal + 0.70 * arc + 0.60 * blue
    B = 0.3 * gal + 1.00 * arc + 1.00 * blue
    img = np.stack([R, G, B], axis=-1).astype(float)
    for c in range(3):
        img[..., c] = (img[..., c]
                       + image_util.add_poisson(img[..., c], exp_time=exp_time)
                       + image_util.add_background(img[..., c], sigma_bkd=bkg_rms))
    img = np.clip(img, 0, None)
    img = np.arcsinh(img / (np.median(img) + img.std() + 1e-6))
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    return (img * 255).astype(np.uint8)


def build_dataset(out_dir: str, n_train: int = 2000, n_val: int = 500,
                  pos_frac: float = 0.1, size: int = 64, deltaPix: float = 0.1,
                  seed: int = 42) -> str:
    rng = np.random.default_rng(seed)

    def items():
        for split, n in [("train", n_train), ("val", n_val)]:
            for i in range(n):
                is_lens = bool(rng.random() < pos_frac)
                img = generate_lens_cutout(size, deltaPix, is_lens, rng)
                if (i + 1) % 200 == 0:
                    print(f"  {split}: {i + 1}/{n}")
                yield split, int(is_lens), img, f"{i:05d}"

    index, rows = write_png_dataset(out_dir, items())
    print(f"lenstronomy: wrote {len(rows)} images "
          f"({sum(r['label'] for r in rows)} lenses) -> {index}")
    return index
