# DINOv3 Strong-Lens Finder

[![CI](https://github.com/fabio-ag-silveira/DINOv3-Strong-Lens-Finder/actions/workflows/ci.yml/badge.svg)](https://github.com/fabio-ag-silveira/DINOv3-Strong-Lens-Finder/actions/workflows/ci.yml)

Find **strong gravitational lenses** (arcs, Einstein rings) in wide-field survey
imagery by fine-tuning a **DINOv3** self-supervised vision backbone. Lenses are
~1 in tens of thousands of galaxies yet invaluable for measuring dark matter and
the Hubble constant — a textbook **needle-in-a-haystack ranking** problem and a
clean showcase for modern transfer learning.

The project's story: **train on physical simulations, evaluate on a public
benchmark** (Bologna Lens Finding Challenge).

![Simulated examples — green = lens, red = non-lens](assets/lens_examples.png)

## Why DINOv3
- **Gram anchoring** keeps dense features sharp → optionally *localize* the arc with
  a segmentation head, not just classify.
- A **satellite-pretrained** variant narrows the domain gap to astronomical imaging.
- A full **size menu** (ViT-S/B/L + ConvNeXt) — small variants fit an 8 GB GPU.

## Install
```bash
git clone git@github.com:fabio-ag-silveira/DINOv3-Strong-Lens-Finder.git
cd DINOv3-Strong-Lens-Finder
pip install -e ".[sim,demo,notebook]"   # editable install with all extras
# or, without installing the package:  pip install -r requirements.txt
```

## Quickstart
```bash
# 1) physically-simulated training data (lenstronomy)
dino-lens simulate-physical --n-train 4000 --n-val 1000 --pos-frac 0.1

# 2) verify DINOv3 access (after `hf auth login` + accepting the licence)
dino-lens check-backbone

# 3) train, then evaluate -> metrics + ranked candidate list
dino-lens train  --config configs/default.yaml
dino-lens eval   --ckpt runs/exp1/best.pt --split val

# 4) evaluate on the benchmark (download the Bologna challenge first)
dino-lens make-bologna --image-dir /path/to/bologna/fits --catalog /path/to/catalog.csv
#   then set data.index -> data/bologna/index.csv and re-run `dino-lens eval`

# 5) interactive demo
LENS_CKPT=runs/exp1/best.pt python app/gradio_app.py
```
No GPU yet? `pytest -q` runs a CPU end-to-end check with a tiny timm backbone.

## Results

Run [`notebooks/results.ipynb`](notebooks/results.ipynb) end-to-end **on CPU**
(timm fallback backbone — no DINOv3 download, no GPU) to reproduce a ROC curve and
a ranked-candidate grid. Example output:

![ROC curve](assets/example_roc.png)

![Top-ranked candidates](assets/example_ranked_grid.png)

Swap in DINOv3 + a GPU (see the notebook's last section) for publication-grade numbers.

## Docs
- [docs/usage.md](docs/usage.md) — full usage, config reference, DINOv3 access, 8 GB tips.
- [docs/architecture.md](docs/architecture.md) — module map and design decisions.

## Layout
```
src/dino_lens_finder/   installable package (see docs/architecture.md)
configs/default.yaml    typed config (maps to dino_lens_finder/config.py)
app/gradio_app.py       interactive demo
tests/                  CPU smoke tests
assets/                 example imagery
```

## Notes
The lenstronomy simulator is a real lensing forward model (SIE + shear); the toy
generator (`simulation/toy.py`) exists only for the dependency-free test. DINOv3
weights use Meta's **DINOv3 License** (more restrictive than DINOv2's Apache-2.0) —
check it before commercial use.

## License
Code is released under the **MIT License** (see [`LICENSE`](LICENSE)). DINOv3 *weights* are governed by Meta's separate **DINOv3 License** — check it before commercial use.
