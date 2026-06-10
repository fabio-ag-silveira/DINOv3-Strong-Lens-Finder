# Benchmark: Bologna Strong Lens Finding Challenge (sim → real)

The headline, portfolio-worthy number for this project is **sim→real**: train on the
lenstronomy simulations, then evaluate on a *real public benchmark* — the **Strong
Gravitational Lens Finding Challenge** (Metcalf et al. 2019,
[A&A 625, A119](https://www.aanda.org/articles/aa/full_html/2019/05/aa32797-18/aa32797-18.html)
· [arXiv:1802.03609](https://arxiv.org/abs/1802.03609)).

> ⚠️ If the Bologna portal is unreachable from your network, use the server-free
> alternative in **[lenscat.md](lenscat.md)** (real lenses + Legacy Survey cutouts).

## 0. Download (no registration — the challenge is over)
Everything is on the **Bologna Lens Factory** portal:

**http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html**

Recommended starting point — **Challenge 1.0, space-based** (single band, simplest,
and closest to DINOv3's VIS-style pretraining):

- Under **Challenge 1.0 (ended)**, click **“Space Based Training Set”** — 20,000
  cutouts (101×101 FITS) plus an ASCII truth log (lens/non-lens + lens properties).
  Lens-only and source-only images are also included for analysis.
- Multi-band option: **“Ground Based Training Set”** — 20,000 × 4 bands (I, G, R, U);
  note masked regions are set to pixel value 100.
- Bigger / harder (Euclid-like VIS + NISP J/Y/H, 100k × 4, ~19 GB): **Challenge 2.0**
  → “Data pack of images” + “Log of lens properties”.

Tracks share the **101×101 px** cutout size; pixel scale ≈ 0.1″ (space VIS) / 0.2″
(ground). Extract the archive and note (a) the folder of FITS and (b) the truth-log path.

## 1. Look at what you got
```bash
ls <fits_dir> | head           # see the file-name pattern (the ID is in the name)
python -c "import pandas as pd; d=pd.read_csv('truth_log.txt', sep=None, engine='python'); print(d.columns.tolist()); print(d.head())"
```

## 2. Build the binary label
The official definition of a lens (Challenge 2.0; 1.0 is analogous) is:

```
n_source_im > 0   AND   mag_eff > 1.6   AND   n_pix_source > 20
```

Turn the truth log into a CSV with an `is_lens` column our loader reads (`sep=None`
auto-detects whitespace vs comma):

```python
import pandas as pd
df = pd.read_csv("truth_log.txt", sep=None, engine="python")
print(df.columns.tolist())                       # confirm exact names first!
df["is_lens"] = ((df["n_source_im"] > 0) &
                 (df["mag_eff"] > 1.6) &
                 (df["n_pix_source"] > 20)).astype(int)
df.to_csv("catalog.csv", index=False)
print(df["is_lens"].value_counts())
```
Adjust the column/ID names to match your header. (Quick alternative without this
step: pass `--label-col n_source_im --label-threshold 0` to `make-bologna`.)

## 3. Convert to the pipeline format
`make-bologna` writes the same `PNG + index.csv` the trainer/evaluator already read.
`{id}` is replaced by each catalogue ID; `--pattern` uses shell globs, so the default
`*{id}*.fits` works whenever the ID is a unique substring of the file name.

**Space-based (single band):**
```bash
dino-lens make-bologna --image-dir <fits_dir> --catalog catalog.csv \
  --id-col ID --label-col is_lens --pattern "*{id}*.fits" --out data/bologna
```

**Ground-based (compose RGB from 3 of the 4 bands):**
```bash
dino-lens make-bologna --image-dir <fits_dir> --catalog catalog.csv \
  --id-col ID --label-col is_lens --out data/bologna \
  --bands "*_R-{id}.fits" "*_G-{id}.fits" "*_I-{id}.fits"
```
(Run `python -m dino_lens_finder.cli` instead of `dino-lens` if the console script
isn't on your PATH.)

## 4. Evaluate your simulation-trained model on it
Point the validation split at the benchmark and evaluate the existing checkpoint —
**no retraining**:
```yaml
# configs/default.yaml
data:
  index: data/bologna/index.csv
```
```bash
dino-lens eval --ckpt runs/exp1/best.pt --split val
```
Cutouts are 101 px; transforms resize to the backbone's 224 px automatically. Expect
a **lower AUC than on simulations** — that gap is the scientifically interesting
result (domain shift sim→real). Send me the number and we'll interpret it.

## 5. Going further
- **Fine-tune on the Bologna train split** (optionally mixed with simulations) and
  re-evaluate to measure how much real labels close the gap.
- **Threshold studies**: the challenge scores ROC for sub-samples cut on lensed-image
  brightness, size, and Einstein radius — reproduce these with the quality columns.
- **True 4-band input**: DINOv3 takes 3 channels; for all four bands, adapt the
  patch-embed conv (see [usage.md](usage.md) → Multi-band inputs).
