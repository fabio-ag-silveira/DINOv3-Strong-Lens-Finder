"""Interactive demo: upload a galaxy cutout -> lens score + saliency overlay.

    LENS_CKPT=runs/exp1/best.pt python app/gradio_app.py
"""
import os

import numpy as np
import torch
from PIL import Image
import gradio as gr

from dino_lens_finder.config import Config
from dino_lens_finder.data.dataset import build_transforms
from dino_lens_finder.models import build_model
from dino_lens_finder.utils import get_device

cfg = Config.from_yaml(os.environ.get("LENS_CONFIG", "configs/default.yaml"))
device = get_device()
model = build_model(cfg).to(device).eval()
ckpt = os.environ.get("LENS_CKPT", "runs/exp1/best.pt")
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    print(f"Loaded {ckpt}")
else:
    print(f"[warn] checkpoint {ckpt} not found - running with a random head.")

tf = build_transforms(cfg.data.img_size, cfg.data.mean, cfg.data.std, train=False)


def predict(img):
    x = tf(img.convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)
    logit = model(x)
    score = torch.sigmoid(logit).item()
    model.zero_grad()
    logit.sum().backward()
    sal = x.grad.abs().sum(1)[0].detach().cpu().numpy()
    sal = (sal - sal.min()) / (np.ptp(sal) + 1e-8)
    heat = Image.fromarray((sal * 255).astype("uint8")).resize(img.size)
    return {"lens": score, "not-lens": 1 - score}, heat


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Galaxy cutout"),
    outputs=[gr.Label(label="Score"), gr.Image(label="Saliency (where the model looks)")],
    title="DINOv3 Strong-Lens Finder",
    description="Upload a galaxy cutout; the model scores how lens-like it is.",
)

if __name__ == "__main__":
    demo.launch()
