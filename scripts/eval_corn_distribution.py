"""Evaluate CORN checkpoints via implied class-probability argmax."""
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
from drnet.engine.metrics import confusion, per_class_recall, quadratic_weighted_kappa
from drnet.losses.ordinal import ordinal_logits_to_label
from drnet.models import MultiTaskNet
from drnet.utils import load_checkpoint, load_config


def _corn_class_probs(logits: torch.Tensor) -> torch.Tensor:
    cond = torch.sigmoid(logits)
    tails = torch.cumprod(cond, dim=1)
    p0 = 1.0 - tails[:, :1]
    middle = tails[:, :-1] - tails[:, 1:] if tails.size(1) > 1 else tails[:, :0]
    plast = tails[:, -1:]
    return torch.cat([p0, middle, plast], dim=1)


@torch.no_grad()
def _predict_fold(cfg: dict, fold: int, ckpt: Path, device: str) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
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

    strategies = ("corn_threshold", "corn_prob_argmax")
    out = {
        strategy: {"dr": {"gt": [], "pred": []}, "me": {"gt": [], "pred": []}}
        for strategy in strategies
    }

    for images, targets in loader:
        outputs = model(images.to(device))
        for task in ("dr", "me"):
            logits = outputs[task]
            preds = {
                "corn_threshold": ordinal_logits_to_label(logits, mode="corn"),
                "corn_prob_argmax": _corn_class_probs(logits).argmax(dim=1),
            }
            for strategy, pred in preds.items():
                out[strategy][task]["gt"].append(targets[task])
                out[strategy][task]["pred"].append(pred.cpu())

    return {
        strategy: {
            task: (
                torch.cat(out[strategy][task]["gt"]).numpy(),
                torch.cat(out[strategy][task]["pred"]).numpy(),
            )
            for task in ("dr", "me")
        }
        for strategy in strategies
    }


def _summarize(gt_parts: list[np.ndarray], pred_parts: list[np.ndarray], num_classes: int) -> dict:
    gt = np.concatenate(gt_parts)
    pred = np.concatenate(pred_parts)
    recall = per_class_recall(gt, pred, num_classes)
    qwk = quadratic_weighted_kappa(gt, pred)
    return {
        "n": int(len(gt)),
        "qwk": qwk,
        "macro_recall": float(np.mean(recall)),
        "balanced": 0.5 * qwk + 0.5 * float(np.mean(recall)),
        "confusion_matrix": confusion(gt, pred, num_classes).tolist(),
        "recall_per_class": recall,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_finetune.yaml")
    ap.add_argument("--cv-root", default="checkpoints/stage2_cv")
    ap.add_argument("--out", default="checkpoints/stage2_cv/corn_distribution_eval.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["model"]["head"] != "corn":
        raise ValueError("CORN distribution evaluation expects model.head=corn")

    cv_root = Path(args.cv_root)
    folds = sorted(int(p.name.split("_", 1)[1]) for p in cv_root.glob("fold_*") if p.is_dir())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strategies = ("corn_threshold", "corn_prob_argmax")
    all_parts = {
        strategy: {"dr": {"gt": [], "pred": []}, "me": {"gt": [], "pred": []}}
        for strategy in strategies
    }
    per_fold = []

    for fold in folds:
        ckpt = cv_root / f"fold_{fold}" / "best_qwk.pth"
        print(f"predict fold={fold} ckpt={ckpt}")
        result = _predict_fold(cfg, fold, ckpt, device)
        for strategy in strategies:
            row = {"fold": fold, "strategy": strategy}
            for task in ("dr", "me"):
                gt, pred = result[strategy][task]
                all_parts[strategy][task]["gt"].append(gt)
                all_parts[strategy][task]["pred"].append(pred)
                recall = per_class_recall(gt, pred, cfg["model"]["num_classes"][task])
                qwk = quadratic_weighted_kappa(gt, pred)
                row[f"{task}_qwk"] = qwk
                row[f"{task}_recall_per_class"] = recall
                row[f"{task}_macro_recall"] = float(np.mean(recall))
            per_fold.append(row)

    summary = {"folds": folds, "per_fold": per_fold, "pooled": {}}
    for strategy in strategies:
        summary["pooled"][strategy] = {}
        for task in ("dr", "me"):
            summary["pooled"][strategy][task] = _summarize(
                all_parts[strategy][task]["gt"],
                all_parts[strategy][task]["pred"],
                cfg["model"]["num_classes"][task],
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "task", "qwk", "macro_recall", "balanced", "recall_per_class", "confusion_matrix"])
        for strategy in strategies:
            for task in ("dr", "me"):
                item = summary["pooled"][strategy][task]
                writer.writerow([
                    strategy,
                    task,
                    f"{item['qwk']:.6f}",
                    f"{item['macro_recall']:.6f}",
                    f"{item['balanced']:.6f}",
                    json.dumps(item["recall_per_class"]),
                    json.dumps(item["confusion_matrix"]),
                ])

    for strategy in strategies:
        print(f"== {strategy} ==")
        for task in ("dr", "me"):
            item = summary["pooled"][strategy][task]
            print(
                f"{task.upper()} n={item['n']} qwk={item['qwk']:.4f} "
                f"macro_recall={item['macro_recall']:.4f} balanced={item['balanced']:.4f}"
            )
            print(np.array(item["confusion_matrix"]))
            print(f"recall_per_class={[round(x, 4) for x in item['recall_per_class']]}")
    print(f"saved {out_path}")
    print(f"saved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
