"""Training loop tuned for 8 GB VRAM: mixed precision, gradient accumulation, and
imbalance-aware sampling. Saves the best checkpoint by validation TPR@FPR."""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..data.dataset import LensDataset, build_transforms, make_weighted_sampler
from ..models import build_model
from ..utils import count_trainable, get_device, get_logger
from .losses import BinaryFocalLoss
from .metrics import compute_metrics


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = get_logger()
        self.device = get_device()
        self.model = build_model(cfg).to(self.device)
        self.log.info(f"Device: {self.device} | trainable params: "
                      f"{count_trainable(self.model):,}")

    def _loaders(self) -> Tuple[DataLoader, DataLoader]:
        d, t = self.cfg.data, self.cfg.train
        tf_tr = build_transforms(d.img_size, d.mean, d.std, train=True)
        tf_va = build_transforms(d.img_size, d.mean, d.std, train=False)
        tr = LensDataset(d.index, "train", tf_tr)
        va = LensDataset(d.index, "val", tf_va)
        sampler = make_weighted_sampler(tr.labels()) if t.balance_sampler else None
        tr_loader = DataLoader(tr, batch_size=t.batch_size, sampler=sampler,
                               shuffle=sampler is None, num_workers=d.num_workers,
                               pin_memory=True, drop_last=True)
        va_loader = DataLoader(va, batch_size=t.batch_size, shuffle=False,
                               num_workers=d.num_workers, pin_memory=True)
        return tr_loader, va_loader

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[List[int], List[float]]:
        self.model.eval()
        ys, ss = [], []
        for x, y, _ in loader:
            logit = self.model(x.to(self.device))
            ss.extend(torch.sigmoid(logit).float().cpu().tolist())
            ys.extend(y.tolist())
        return ys, ss

    def fit(self) -> str:
        t = self.cfg.train
        tr_loader, va_loader = self._loaders()
        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=t.lr, weight_decay=t.weight_decay)
        crit = BinaryFocalLoss(t.focal_gamma, t.focal_alpha)

        use_amp = t.amp and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if t.precision == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
        accum = max(1, t.grad_accum)
        os.makedirs(t.out_dir, exist_ok=True)

        best = -1.0
        for epoch in range(t.epochs):
            self.model.train()
            opt.zero_grad()
            pbar = tqdm(tr_loader, desc=f"epoch {epoch}")
            for i, (x, y, _) in enumerate(pbar):
                x, y = x.to(self.device), y.float().to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    loss = crit(self.model(x), y) / accum
                scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
                # step every `accum` micro-batches, and on the last batch so
                # leftover gradients are never silently dropped
                if (i + 1) % accum == 0 or (i + 1) == len(tr_loader):
                    if scaler.is_enabled():
                        scaler.step(opt); scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad()
                pbar.set_postfix(loss=loss.detach().item() * accum)

            m = compute_metrics(*self.evaluate(va_loader),
                                fpr_target=self.cfg.eval.fpr_target,
                                top_n=self.cfg.eval.top_n)
            self.log.info(f"[val] epoch {epoch}: {m}")
            score = m["tpr_at_fpr"] if m["tpr_at_fpr"] == m["tpr_at_fpr"] else m["roc_auc"]
            if score == score and score > best:
                best = score
                torch.save({"model": self.model.state_dict(),
                            "cfg": self.cfg.to_dict(), "metrics": m},
                           os.path.join(t.out_dir, "best.pt"))
                with open(os.path.join(t.out_dir, "metrics.json"), "w") as f:
                    json.dump(m, f, indent=2)
        self.log.info(f"Best val score: {best:.4f} -> {t.out_dir}/best.pt")
        return os.path.join(t.out_dir, "best.pt")
