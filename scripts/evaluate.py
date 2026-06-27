"""Evaluate a stage-2 checkpoint on one fold."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnet.data import MessidorMultiTaskDataset, build_transforms
from drnet.engine import preds_from_outputs
from drnet.engine.metrics import confusion, per_class_recall, quadratic_weighted_kappa
from drnet.models import MultiTaskNet
from drnet.utils import load_checkpoint, load_config


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_finetune.yaml")
    ap.add_argument("--ckpt", default="checkpoints/stage2/best_qwk.pth")
    ap.add_argument("--out", default="checkpoints/stage2")
    ap.add_argument("--fold", type=int, help="Override data.fold from config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = cfg["data"]
    head_mode = cfg["model"]["head"]
    nc = cfg["model"]["num_classes"]

    tf_val = build_transforms(d["image_size"], train=False)
    ds = MessidorMultiTaskDataset(d["folds_csv"], d["root"], d["fold"], "val",
                                  transform=tf_val, image_size=d["image_size"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=d.get("num_workers", 4))

    model = MultiTaskNet(cfg["model"]).to(device)
    load_checkpoint(model, args.ckpt, map_location=device, strict=False)
    model.eval()

    preds = {"dr": [], "me": []}
    gts = {"dr": [], "me": []}
    metrics = {}
    for img, tg in loader:
        out = model(img.to(device))
        for t in ("dr", "me"):
            preds[t].append(preds_from_outputs(out[t], head_mode).cpu())
            gts[t].append(tg[t])

    for t in ("dr", "me"):
        p = torch.cat(preds[t]).numpy()
        g = torch.cat(gts[t]).numpy()
        qwk = quadratic_weighted_kappa(g, p)
        acc = accuracy_score(g, p)
        prec = precision_score(g, p, labels=list(range(nc[t])), average="macro", zero_division=0)
        rec = per_class_recall(g, p, nc[t])
        cm = confusion(g, p, nc[t])
        metrics[t] = {
            "qwk": float(qwk),
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_per_class": [float(r) for r in rec],
        }
        print(
            f"== {t.upper()} ==  QWK={qwk:.4f}  ACC={acc:.4f}  "
            f"PREC(macro)={prec:.4f}  per-class recall={[round(r, 3) for r in rec]}"
        )
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("pred")
        plt.ylabel("true")
        plt.title(f"{t.upper()} confusion (QWK={qwk:.3f})")
        plt.tight_layout()
        path = os.path.join(args.out, f"confusion_{t}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"saved {path}")
    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"saved {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
