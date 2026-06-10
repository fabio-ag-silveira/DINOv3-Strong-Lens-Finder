"""Unified command-line interface:  `dino-lens <command> [options]`.

Commands
  simulate-physical   build a lenstronomy training set        (needs the 'sim' extra)
  simulate-toy        build a dependency-free toy set
  make-bologna        convert a local Bologna challenge copy   (needs the 'sim' extra)
  check-backbone      verify DINOv3 access + token layout
  train               fine-tune the model
  eval                evaluate a checkpoint + write ranked candidates
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dino-lens",
                                description="DINOv3 strong gravitational lens finder")
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_config(sp):
        sp.add_argument("--config", default="configs/default.yaml")

    sp = sub.add_parser("simulate-physical", help="lenstronomy training set")
    sp.add_argument("--out", default="data/lenstronomy")
    sp.add_argument("--n-train", type=int, default=2000)
    sp.add_argument("--n-val", type=int, default=500)
    sp.add_argument("--pos-frac", type=float, default=0.1)
    sp.add_argument("--size", type=int, default=64)
    sp.add_argument("--deltapix", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=42)

    sp = sub.add_parser("simulate-toy", help="dependency-free toy set")
    sp.add_argument("--out", default="data/synthetic")
    sp.add_argument("--n-train", type=int, default=400)
    sp.add_argument("--n-val", type=int, default=100)
    sp.add_argument("--pos-frac", type=float, default=0.3)
    sp.add_argument("--size", type=int, default=128)
    sp.add_argument("--seed", type=int, default=42)

    sp = sub.add_parser("make-bologna", help="convert a local Bologna challenge copy")
    sp.add_argument("--image-dir", required=True)
    sp.add_argument("--catalog", required=True)
    sp.add_argument("--out", default="data/bologna")
    sp.add_argument("--id-col", default="ID")
    sp.add_argument("--label-col", default="is_lens")
    sp.add_argument("--pattern", default="*{id}*.fits")
    sp.add_argument("--val-frac", type=float, default=0.2)
    sp.add_argument("--size", type=int, default=None)

    with_config(sub.add_parser("check-backbone", help="verify DINOv3 access"))
    with_config(sub.add_parser("train", help="fine-tune the model"))

    sp = sub.add_parser("eval", help="evaluate a checkpoint")
    with_config(sp)
    sp.add_argument("--ckpt", default="runs/exp1/best.pt")
    sp.add_argument("--split", default="val")
    sp.add_argument("--out", default="runs/exp1/ranked_candidates.csv")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.cmd == "simulate-physical":
        from .simulation.lenstronomy_sim import build_dataset
        build_dataset(args.out, args.n_train, args.n_val, args.pos_frac,
                      args.size, args.deltapix, args.seed)
    elif args.cmd == "simulate-toy":
        from .simulation.toy import build_dataset
        build_dataset(args.out, args.n_train, args.n_val, args.pos_frac,
                      args.size, args.seed)
    elif args.cmd == "make-bologna":
        from .data.bologna import build_bologna_index
        build_bologna_index(args.image_dir, args.catalog, args.out,
                            id_col=args.id_col, label_col=args.label_col,
                            pattern=args.pattern, val_frac=args.val_frac, size=args.size)
    elif args.cmd == "check-backbone":
        from .config import Config
        from .models.backbone import check_backbone
        check_backbone(Config.from_yaml(args.config))
    elif args.cmd == "train":
        from .config import Config
        from .training.trainer import Trainer
        Trainer(Config.from_yaml(args.config)).fit()
    elif args.cmd == "eval":
        from .config import Config
        from .evaluation import evaluate_checkpoint
        evaluate_checkpoint(Config.from_yaml(args.config), args.ckpt, args.split, args.out)


if __name__ == "__main__":
    main()
