"""Summarize held-out predictions from stage-2 cross-validation checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnet.data import MessidorMultiTaskDataset, build_transforms
from drnet.engine import preds_from_outputs
from drnet.engine.metrics import confusion, per_class_recall, quadratic_weighted_kappa
from drnet.models import MultiTaskNet
from drnet.utils import load_checkpoint, load_config


@torch.no_grad()
def _predict_fold(cfg: dict, fold: int, ckpt: Path, device: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data_cfg = cfg["data"]
    tf_val = build_transforms(data_cfg["image_size"], train=False, resize=data_cfg.get("resize", True))
    ds = MessidorMultiTaskDataset(
        data_cfg["folds_csv"],
        data_cfg["root"],
        fold,
        "val",
        transform=tf_val,
        image_size=data_cfg["image_size"],
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
    )

    model = MultiTaskNet(cfg["model"]).to(device)
    load_checkpoint(model, str(ckpt), map_location=device, strict=False)
    model.eval()

    head_mode = cfg["model"]["head"]
    preds = {"dr": [], "me": []}
    gts = {"dr": [], "me": []}
    for images, targets in loader:
        outputs = model(images.to(device))
        for task in ("dr", "me"):
            preds[task].append(preds_from_outputs(outputs[task], head_mode).cpu())
            gts[task].append(targets[task])

    return {
        task: (torch.cat(gts[task]).numpy(), torch.cat(preds[task]).numpy())
        for task in ("dr", "me")
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_finetune.yaml")
    ap.add_argument("--cv-root", default="checkpoints/stage2_cv")
    ap.add_argument("--out", default="checkpoints/stage2_cv/heldout_eval_summary.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cv_root = Path(args.cv_root)
    folds = sorted(int(p.name.split("_", 1)[1]) for p in cv_root.glob("fold_*") if p.is_dir())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_gt = {"dr": [], "me": []}
    all_pred = {"dr": [], "me": []}
    per_fold = []
    for fold in folds:
        ckpt = cv_root / f"fold_{fold}" / "best_qwk.pth"
        result = _predict_fold(cfg, fold, ckpt, device)
        row = {"fold": fold}
        for task in ("dr", "me"):
            gt, pred = result[task]
            all_gt[task].append(gt)
            all_pred[task].append(pred)
            row[f"{task}_qwk"] = quadratic_weighted_kappa(gt, pred)
            row[f"{task}_recall_per_class"] = per_class_recall(
                gt,
                pred,
                cfg["model"]["num_classes"][task],
            )
        per_fold.append(row)

    summary = {"folds": folds, "per_fold": per_fold, "pooled": {}}
    for task in ("dr", "me"):
        gt = np.concatenate(all_gt[task])
        pred = np.concatenate(all_pred[task])
        num_classes = cfg["model"]["num_classes"][task]
        summary["pooled"][task] = {
            "n": int(len(gt)),
            "qwk": quadratic_weighted_kappa(gt, pred),
            "confusion_matrix": confusion(gt, pred, num_classes).tolist(),
            "recall_per_class": per_class_recall(gt, pred, num_classes),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "task", "qwk", "recall_per_class"])
        for row in per_fold:
            for task in ("dr", "me"):
                writer.writerow([
                    row["fold"],
                    task,
                    f"{row[f'{task}_qwk']:.6f}",
                    json.dumps(row[f"{task}_recall_per_class"]),
                ])

    for task in ("dr", "me"):
        item = summary["pooled"][task]
        print(f"== {task.upper()} pooled held-out ==")
        print(f"n={item['n']} qwk={item['qwk']:.4f}")
        print("confusion_matrix:")
        print(np.array(item["confusion_matrix"]))
        print(f"recall_per_class={[round(x, 4) for x in item['recall_per_class']]}")
    print(f"saved {out_path}")
    print(f"saved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
