"""Generate Grad-CAM for a stage-2 checkpoint."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnet.data import MessidorMultiTaskDataset, build_transforms
from drnet.engine import preds_from_outputs
from drnet.explain import gradcam_for, save_overlay
from drnet.models import MultiTaskNet
from drnet.utils import load_checkpoint, load_config

_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def _denorm(t: torch.Tensor) -> np.ndarray:
    img = t.permute(1, 2, 0).cpu().numpy() * _STD + _MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_finetune.yaml")
    ap.add_argument("--ckpt", default="checkpoints/stage2/best_qwk.pth")
    ap.add_argument("--head", default="dr", choices=["dr", "me"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/stage2/gradcam")
    ap.add_argument("--fold", type=int, help="Override data.fold from config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = cfg["data"]
    os.makedirs(args.out, exist_ok=True)

    tf_val = build_transforms(d["image_size"], train=False)
    ds = MessidorMultiTaskDataset(d["folds_csv"], d["root"], d["fold"], "val",
                                  transform=tf_val, image_size=d["image_size"])
    model = MultiTaskNet(cfg["model"]).to(device)
    load_checkpoint(model, args.ckpt, map_location=device, strict=False)
    model.eval()

    for i in range(min(args.n, len(ds))):
        img, _ = ds[i]
        x = img.unsqueeze(0).to(device)
        pred = int(preds_from_outputs(model(x)[args.head], cfg["model"]["head"])[0])
        cam = gradcam_for(model, x, args.head, pred)
        out_path = os.path.join(args.out, f"{i:03d}_{args.head}_pred{pred}.png")
        save_overlay(_denorm(img), cam, out_path)
        print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
