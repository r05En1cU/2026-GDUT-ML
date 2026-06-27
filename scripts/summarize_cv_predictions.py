"""Summarize held-out predictions from stage-2 cross-validation checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnet.data import MessidorMultiTaskDataset, build_transforms
from drnet.engine import preds_from_outputs
from drnet.engine.metrics import confusion, per_class_recall, quadratic_weighted_kappa
from drnet.models import MultiTaskNet
from drnet.utils import load_checkpoint, load_config


@torch.no_grad()
def _predict_fold(cfg: dict, fold: int, ckpt: Path, device: str) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    list[dict],
]:
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
    rows = []
    for images, targets in loader:
        outputs = model(images.to(device))
        batch_pred = {}
        for task in ("dr", "me"):
            batch_pred[task] = preds_from_outputs(outputs[task], head_mode).cpu()
            preds[task].append(batch_pred[task])
            gts[task].append(targets[task])
        start = len(rows)
        for j in range(images.shape[0]):
            row = ds.df.iloc[start + j]
            rows.append({
                "fold": fold,
                "image": row["image"],
                "patient_id": row.get("patient_id", ""),
                "dr_true": int(targets["dr"][j]),
                "dr_pred": int(batch_pred["dr"][j]),
                "me_true": int(targets["me"][j]),
                "me_pred": int(batch_pred["me"][j]),
            })

    result = {
        task: (torch.cat(gts[task]).numpy(), torch.cat(preds[task]).numpy())
        for task in ("dr", "me")
    }
    return result, rows


def _binary_detection_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_bin = gt > 0
    pred_bin = pred > 0
    tp = int((gt_bin & pred_bin).sum())
    tn = int((~gt_bin & ~pred_bin).sum())
    fp = int((~gt_bin & pred_bin).sum())
    fn = int((gt_bin & ~pred_bin).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "balanced": float(0.5 * (sensitivity + specificity)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _save_confusion_plot(cm: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(4.8, 4.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_finetune.yaml")
    ap.add_argument("--cv-root", default="checkpoints/stage2_cv")
    ap.add_argument("--out", default="checkpoints/stage2_cv/heldout_eval_summary.json")
    ap.add_argument("--resize", choices=("on", "off"),
                    help="Override data.resize in config for validation transforms")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.resize is not None:
        cfg["data"]["resize"] = args.resize == "on"
    cv_root = Path(args.cv_root)
    folds = sorted(int(p.name.split("_", 1)[1]) for p in cv_root.glob("fold_*") if p.is_dir())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_gt = {"dr": [], "me": []}
    all_pred = {"dr": [], "me": []}
    per_fold = []
    pred_rows = []
    for fold in folds:
        ckpt = cv_root / f"fold_{fold}" / "best_qwk.pth"
        result, rows = _predict_fold(cfg, fold, ckpt, device)
        pred_rows.extend(rows)
        row = {"fold": fold}
        for task in ("dr", "me"):
            gt, pred = result[task]
            all_gt[task].append(gt)
            all_pred[task].append(pred)
            row[f"{task}_accuracy"] = float(accuracy_score(gt, pred))
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
            "accuracy": float(accuracy_score(gt, pred)),
            "qwk": quadratic_weighted_kappa(gt, pred),
            "confusion_matrix": confusion(gt, pred, num_classes).tolist(),
            "recall_per_class": per_class_recall(gt, pred, num_classes),
            "positive_detection": _binary_detection_metrics(gt, pred),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "task", "accuracy", "qwk", "recall_per_class"])
        for row in per_fold:
            for task in ("dr", "me"):
                writer.writerow([
                    row["fold"],
                    task,
                    f"{row[f'{task}_accuracy']:.6f}",
                    f"{row[f'{task}_qwk']:.6f}",
                    json.dumps(row[f"{task}_recall_per_class"]),
                ])

    pred_path = out_path.with_name("heldout_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)

    for task in ("dr", "me"):
        item = summary["pooled"][task]
        cm = np.array(item["confusion_matrix"])
        _save_confusion_plot(
            cm,
            out_path.with_name(f"confusion_{task}.png"),
            f"{task.upper()} pooled confusion (ACC={item['accuracy']:.3f}, QWK={item['qwk']:.3f})",
        )
        print(f"== {task.upper()} pooled held-out ==")
        print(f"n={item['n']} acc={item['accuracy']:.4f} qwk={item['qwk']:.4f}")
        print("confusion_matrix:")
        print(cm)
        print(f"recall_per_class={[round(x, 4) for x in item['recall_per_class']]}")
        det = item["positive_detection"]
        print(
            "positive_detection="
            f"sens={det['sensitivity']:.4f} spec={det['specificity']:.4f} "
            f"fpr={det['fpr']:.4f} fnr={det['fnr']:.4f}"
        )
    print(f"saved {out_path}")
    print(f"saved {csv_path}")
    print(f"saved {pred_path}")
    print(f"saved {out_path.with_name('confusion_dr.png')}")
    print(f"saved {out_path.with_name('confusion_me.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
