"""Evaluate softmax checkpoints with ordinal projection post-processing."""
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
from drnet.models import MultiTaskNet
from drnet.utils import load_checkpoint, load_config


def _raw_softmax(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)


def _ordinal_projection(logits: torch.Tensor, thresholds: list[float]) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    tails = torch.flip(torch.cumsum(torch.flip(probs, dims=(1,)), dim=1), dims=(1,))[:, 1:]
    th = torch.tensor(thresholds, device=logits.device, dtype=probs.dtype).view(1, -1)
    return (tails > th).sum(dim=1)


def _middle_class_guard(logits: torch.Tensor, thresholds: list[float]) -> torch.Tensor:
    raw = _raw_softmax(logits)
    proj = _ordinal_projection(logits, thresholds)
    max_class = logits.size(1) - 1
    raw_mid = (raw > 0) & (raw < max_class)
    proj_mid = (proj > 0) & (proj < max_class)
    return torch.where(raw_mid | proj_mid, proj, raw)


def _thresholds(spec: str, num_classes: int) -> list[float]:
    vals = [float(x) for x in spec.split(",")]
    if len(vals) == 1:
        vals = vals * (num_classes - 1)
    if len(vals) != num_classes - 1:
        raise ValueError(f"expected 1 or {num_classes - 1} thresholds, got {spec}")
    return vals


@torch.no_grad()
def _predict_fold(
    cfg: dict,
    fold: int,
    ckpt: Path,
    device: str,
    threshold_spec: str,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
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

    strategies = ("raw_softmax", "ordinal_projection", "middle_class_guard")
    out = {
        strategy: {"dr": {"gt": [], "pred": []}, "me": {"gt": [], "pred": []}}
        for strategy in strategies
    }
    thresholds = {
        task: _thresholds(threshold_spec, cfg["model"]["num_classes"][task])
        for task in ("dr", "me")
    }

    for images, targets in loader:
        outputs = model(images.to(device))
        for task in ("dr", "me"):
            logits = outputs[task]
            preds = {
                "raw_softmax": _raw_softmax(logits),
                "ordinal_projection": _ordinal_projection(logits, thresholds[task]),
                "middle_class_guard": _middle_class_guard(logits, thresholds[task]),
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
    return {
        "n": int(len(gt)),
        "qwk": quadratic_weighted_kappa(gt, pred),
        "confusion_matrix": confusion(gt, pred, num_classes).tolist(),
        "recall_per_class": per_class_recall(gt, pred, num_classes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_ablate_softmax_ldam.yaml")
    ap.add_argument("--cv-root", default="checkpoints/stage2_cv_ablate_softmax_ldam")
    ap.add_argument("--out", default="checkpoints/stage2_cv_ablate_softmax_ldam/projection_eval.json")
    ap.add_argument("--thresholds", default="0.5", help="One threshold or comma-separated K-1 thresholds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["model"]["head"] != "softmax":
        raise ValueError("ordinal projection expects model.head=softmax")

    cv_root = Path(args.cv_root)
    folds = sorted(int(p.name.split("_", 1)[1]) for p in cv_root.glob("fold_*") if p.is_dir())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strategies = ("raw_softmax", "ordinal_projection", "middle_class_guard")
    all_parts = {
        strategy: {"dr": {"gt": [], "pred": []}, "me": {"gt": [], "pred": []}}
        for strategy in strategies
    }
    per_fold = []

    for fold in folds:
        ckpt = cv_root / f"fold_{fold}" / "best_qwk.pth"
        result = _predict_fold(cfg, fold, ckpt, device, args.thresholds)
        for strategy in strategies:
            row = {"fold": fold, "strategy": strategy}
            for task in ("dr", "me"):
                gt, pred = result[strategy][task]
                all_parts[strategy][task]["gt"].append(gt)
                all_parts[strategy][task]["pred"].append(pred)
                row[f"{task}_qwk"] = quadratic_weighted_kappa(gt, pred)
                row[f"{task}_recall_per_class"] = per_class_recall(
                    gt,
                    pred,
                    cfg["model"]["num_classes"][task],
                )
            per_fold.append(row)

    summary = {"folds": folds, "thresholds": args.thresholds, "per_fold": per_fold, "pooled": {}}
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
        writer.writerow(["strategy", "task", "qwk", "recall_per_class", "confusion_matrix"])
        for strategy in strategies:
            for task in ("dr", "me"):
                item = summary["pooled"][strategy][task]
                writer.writerow([
                    strategy,
                    task,
                    f"{item['qwk']:.6f}",
                    json.dumps(item["recall_per_class"]),
                    json.dumps(item["confusion_matrix"]),
                ])

    for strategy in strategies:
        print(f"== {strategy} ==")
        for task in ("dr", "me"):
            item = summary["pooled"][strategy][task]
            print(f"{task.upper()} n={item['n']} qwk={item['qwk']:.4f}")
            print(np.array(item["confusion_matrix"]))
            print(f"recall_per_class={[round(x, 4) for x in item['recall_per_class']]}")
    print(f"saved {out_path}")
    print(f"saved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
