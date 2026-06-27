"""Generate Grad-CAM for a stage-2 checkpoint."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
    ap.add_argument("--resize", choices=("on", "off"),
                    help="Override data.resize for validation transforms")
    ap.add_argument("--indices", nargs="*", type=int,
                    help="Dataset indices within the validation fold. Defaults to first n.")
    ap.add_argument("--target-mode", choices=("full", "binary"), default="full",
                    help="Use full-grade targets or 0-vs-positive binary targets.")
    ap.add_argument("--save-original", action="store_true",
                    help="Also save denormalized input images for side-by-side comparison.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    if args.resize is not None:
        cfg["data"]["resize"] = args.resize == "on"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = cfg["data"]
    os.makedirs(args.out, exist_ok=True)

    tf_val = build_transforms(d["image_size"], train=False, resize=d.get("resize", True))
    ds = MessidorMultiTaskDataset(
        d["folds_csv"],
        d["root"],
        d["fold"],
        "val",
        transform=tf_val,
        image_size=d["image_size"],
        target_mode=args.target_mode,
    )
    model = MultiTaskNet(cfg["model"]).to(device)
    load_checkpoint(model, args.ckpt, map_location=device, strict=False)
    model.eval()

    indices = args.indices if args.indices else list(range(min(args.n, len(ds))))
    rows = []
    for i in indices:
        if i < 0 or i >= len(ds):
            raise IndexError(f"index {i} out of range for fold val set of size {len(ds)}")
        img, target = ds[i]
        x = img.unsqueeze(0).to(device)
        logits = model(x)[args.head]
        pred = int(preds_from_outputs(logits, cfg["model"]["head"])[0])
        if cfg["model"]["head"] in ("corn", "coral"):
            # Ordinal heads expose K-1 threshold logits, not K class logits.
            cam_target = min(max(pred - 1, 0), logits.shape[1] - 1)
        else:
            cam_target = pred
        cam = gradcam_for(model, x, args.head, cam_target)
        row = ds.df.iloc[i]
        true = int(target[args.head])
        stem = f"{i:03d}_{args.head}_true{true}_pred{pred}"
        out_path = os.path.join(args.out, f"{stem}.png")
        rgb = _denorm(img)
        if args.save_original:
            Image.fromarray(rgb).save(os.path.join(args.out, f"{stem}_orig.png"))
        save_overlay(rgb, cam, out_path)
        rows.append({
            "index": i,
            "image": row["image"],
            "patient_id": row.get("patient_id", ""),
            "head": args.head,
            "true": true,
            "pred": pred,
            "cam_target": int(cam_target),
            "dr_grade": int(row["dr_grade"]),
            "me_risk": int(row["me_risk"]),
            "path": out_path,
        })
        print(f"saved {out_path}")
    with open(os.path.join(args.out, f"manifest_{args.head}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index", "image", "patient_id", "head", "true", "pred", "cam_target",
                "dr_grade", "me_risk", "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
