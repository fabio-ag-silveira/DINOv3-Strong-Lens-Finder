# Usage

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -e ".[sim,demo]"     # core + lenstronomy/astropy (sim) + gradio (demo)
```
Extras: `sim` enables `simulate-physical` and `make-bologna`; `demo` enables the app;
`dev` adds `pytest`/`pyflakes`.

## 2. Get DINOv3 (gated, free)
1. Create a **read** token at <https://huggingface.co/settings/tokens>, then:
   ```bash
   pip install -U "huggingface_hub[cli]"
   hf auth login          # paste the token (or: export HF_TOKEN=hf_xxx)
   ```
2. Accept the licence on the model page (once per repo, saved to your account):
   <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
3. `transformers>=4.56` is required (already pinned).
4. Verify:
   ```bash
   dino-lens check-backbone
   # ViT-B @224 -> embed_dim 768, patches (2,196,768), grid (14,14)
   ```
For the satellite backbone use `facebook/dinov3-vitl16-pretrain-sat493m` (ships only
as ViT-L / ViT-7B); `prefix_tokens` stays 5, `embed_dim` is detected automatically.

## 3. Build training data (simulation)
```bash
dino-lens simulate-physical --n-train 4000 --n-val 1000 --pos-frac 0.1 \
                            --size 64 --deltapix 0.1
```
- `--deltapix` arcsec/pixel: ~0.05 space-based, 0.1 Euclid VIS, 0.2 ground.
- `--pos-frac` ~0.01–0.1 mimics the real rarity of lenses.

Output → `data/lenstronomy/{train,val}/{lens,nonlens}/*.png` and `index.csv`.

## 4. Train
Edit `configs/default.yaml` (backbone, `finetune.mode`, `batch_size`), then:
```bash
dino-lens train --config configs/default.yaml
```
Best checkpoint by validation TPR@FPR → `runs/exp1/best.pt` (+ `metrics.json`).

## 5. Evaluate + ranked candidates
```bash
dino-lens eval --ckpt runs/exp1/best.pt --split val
# -> metrics + runs/exp1/ranked_candidates.csv (rank,score,label,path)
```

## 6. Benchmark on the Bologna Lens Challenge
Download the challenge data first (free registration; Metcalf et al. 2019). Then:
```bash
dino-lens make-bologna \
  --image-dir /path/to/bologna/fits \
  --catalog   /path/to/catalog.csv \
  --id-col ID --label-col is_lens --pattern "*{id}*.fits"
```
This writes `data/bologna/index.csv` in the same format. Point `data.index` at it and
re-run `dino-lens eval` to get a number comparable with the literature — i.e. *train
on simulation, evaluate on the real benchmark*.

Input assumptions (all overridable): one FITS per object (single band or several 2-D
image HDUs = bands), and a catalogue CSV with an ID column and a 0/1 label column.
Bands → RGB: first three HDUs map to R,G,B (a single band is replicated), with an
asinh + percentile stretch.

## Config reference (`configs/default.yaml`)
| group | key | meaning |
|------|-----|---------|
| backbone | `source` | `hf` (DINOv3/DINOv2) or `timm` |
| backbone | `name` | HF repo id or timm model name |
| backbone | `prefix_tokens` | CLS+registers to skip for patch tokens (DINOv3=5, DINOv2=1) |
| finetune | `mode` | `frozen` \| `lora` \| `full` |
| head | `seg_head` | add arc-localization head (hf source) |
| train | `precision` | `bf16` (Blackwell/Ada) or `fp16` |
| train | `batch_size`/`grad_accum` | effective batch = product |
| eval | `fpr_target`/`top_n` | ranking-metric settings |

## 8 GB VRAM tips
`amp: true` + `precision: bf16`, small `batch_size` with `grad_accum`, ViT-S/B with
`finetune.mode: lora`, 224 px cutouts. Prototype locally; rent an A100/4090
(RunPod / Vast.ai) only for larger final fine-tunes.

## Multi-band inputs
DINOv3 expects 3 channels. Either compose 3 survey bands into RGB (default), or
replace the patch-embedding conv with an N-channel conv and average-initialise it
from the pretrained 3-channel filters.

## Testing
```bash
pytest -q          # metric + toy tests always run; the end-to-end CPU test
                   # runs when torch is installed (tiny timm backbone, no downloads)
```

## Results notebook
```bash
pip install -e ".[notebook]"
jupyter lab notebooks/results.ipynb   # runs on CPU with a timm fallback backbone
```
It builds a small simulated set, trains a quick baseline, and plots the ROC curve
and the top-ranked candidates — a self-contained first run and a portfolio figure.
