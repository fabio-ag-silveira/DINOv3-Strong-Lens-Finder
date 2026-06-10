# Real benchmark without the Bologna server: lenscat + Legacy Survey

If the Bologna portal is unreachable from your network (it is geo/route-blocked from
some countries), use this **server-free** path. It builds a real lens / non-lens set
from two reliable, globally-reachable sources:

- **Positives** — confirmed/probable lenses from the **lenscat** catalogue
  (Hugging Face: `juliensimon/gravitational-lenses`, 32,838 real systems with RA/Dec).
- **Images** — *grz* colour cutouts from the **Legacy Survey** viewer at each lens
  position.
- **Negatives** — cutouts at random sky positions in the same footprint (strong
  lenses are rare, so random fields are overwhelmingly non-lenses).

## 0. Install + connectivity check
```bash
pip install -e ".[benchmark]"      # pandas, pyarrow, requests
# confirm the cutout service is reachable from your machine:
curl.exe -I "https://www.legacysurvey.org/viewer/cutout.jpg?ra=150.1&dec=2.2&layer=ls-dr10&pixscale=0.262&size=101"
```
A `200 OK` means you're good. (This host is US-based and reachable from Brazil,
unlike the Bologna server.)

## 1. Build the benchmark
```bash
dino-lens make-lenscat --n-per-class 100 --size 101 --out data/lenscat
# scale up once it works:
dino-lens make-lenscat --n-per-class 500 --out data/lenscat
```
Useful flags:
- `--layer ls-dr10` (south, dec ≲ +32) or `--layer ls-dr9` (north). Positives outside
  the chosen footprint return blank cutouts and are auto-skipped.
- `--grading confident` for the cleanest positives (fewer), or the default
  `confident probable` for quantity.
- `--lens-type galaxy` (default) excludes cluster-scale lenses.
- `--catalog-path lenscat.csv` to use a local copy instead of downloading.

Output: `data/lenscat/{train,val}/{lens,nonlens}/*.png` + `index.csv`.

## 2. Evaluate your simulation-trained model
```yaml
# configs/default.yaml
data:
  index: data/lenscat/index.csv
```
```bash
dino-lens eval --ckpt runs/exp1/best.pt --split val
```
Cutouts are 101 px; transforms resize to 224 automatically. Expect a **lower AUC
than on simulations** — that sim→real gap is the result worth reporting.

## Honest caveats
This is a real but **noisier** benchmark than Bologna: lenscat aggregates many
surveys/resolutions (heterogeneous positives), and the negatives are random fields
rather than vetted non-lenses. Treat the number as indicative, and state these
caveats when you report it. For the cleanest comparison, fine-tune/evaluate within a
single `--layer` and `--grading confident`.
