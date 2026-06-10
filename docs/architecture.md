# Architecture

A small, layered package. Each layer depends only on the ones below it, so any
piece (backbone, data source, loss) can be swapped without touching the rest.

```
cli.py                      thin argument parsing -> calls into the layers below
│
├─ config.py                typed dataclasses (Config) + YAML (de)serialization
├─ utils.py                 seeding, device, logging, dataset writing
│
├─ simulation/              produce labelled cutouts (write PNG + index.csv)
│   ├─ toy.py               dependency-free analytic toy (smoke test only)
│   └─ lenstronomy_sim.py   physical SIE+shear forward model  [sim extra]
│
├─ data/                    turn images into tensors / benchmark ingestion
│   ├─ dataset.py           LensDataset, augmentation, imbalance sampler
│   ├─ bologna.py           Bologna Challenge FITS -> PNG + index.csv   [sim extra]
│   └─ lenscat.py           real lenses + Legacy Survey cutouts         [benchmark]
│
├─ models/                  the network
│   ├─ backbone.py          DINOv3/DINOv2 (HF) or timm + LoRA, uniform interface
│   └─ classifier.py        backbone + classification head (+ optional seg head)
│
├─ training/                the optimisation
│   ├─ losses.py            binary focal loss
│   ├─ metrics.py           ROC-AUC, TPR@FPR, precision@N (ranking metrics)
│   └─ trainer.py           Trainer: AMP + grad-accum + best-checkpointing
│
├─ evaluation.py            score a checkpoint -> metrics + ranked candidate CSV
└─ report.py                ROC + ranked-grid figures (matplotlib)        [notebook]
```

## Data contract
Every data source — simulation or benchmark — emits the **same artifact**: a
directory of PNG cutouts plus an `index.csv` with columns `path,label,split`.
Training and evaluation only ever read that contract, which is why "train on
simulation, test on Bologna" needs zero code changes — just repoint `data.index`.

## Backbone interface
`Backbone.forward(x)` always returns `{"cls", "patches", "grid"}`. The classification
head uses `cls`; the optional segmentation head uses `patches`+`grid`. This hides the
difference between HF DINOv3 (CLS + register + patch tokens) and timm (pooled vector).

## Design decisions
- **Typed config** (dataclasses) over raw dicts: defaults live in one place and access
  is checked (`cfg.train.lr`).
- **Frozen / LoRA / full** fine-tuning is a single config switch; LoRA is the 8 GB
  default.
- **Ranking metrics, not accuracy**, because the positive class is ~0.01–0.1 %.
- **Toy vs physical** simulators are separate modules so the test suite never needs
  heavy scientific dependencies.
