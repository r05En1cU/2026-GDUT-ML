"""Cross-fold threshold search for ordinal projection on softmax checkpoints."""
from __future__ import annotations

import argparse
import csv
import itertools
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def _raw_softmax(logits: np.ndarray) -> np.ndarray:
    return logits.argmax(axis=1)


def _ordinal_projection(logits: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    probs = _softmax(logits)
    tails = np.flip(np.cumsum(np.flip(probs, axis=1), axis=1), axis=1)[:, 1:]
    return (tails > np.asarray(thresholds, dtype=probs.dtype)[None, :]).sum(axis=1)


def _middle_class_guard(logits: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    raw = _raw_softmax(logits)
    proj = _ordinal_projection(logits, thresholds)
    max_class = logits.shape[1] - 1
    raw_mid = (raw > 0) & (raw < max_class)
    proj_mid = (proj > 0) & (proj < max_class)
    return np.where(raw_mid | proj_mid, proj, raw)


def _score(gt: np.ndarray, pred: np.ndarray, num_classes: int, objective: str) -> float:
    qwk = quadratic_weighted_kappa(gt, pred)
    macro_recall = float(np.mean(per_class_recall(gt, pred, num_classes)))
    if objective == "qwk":
        return qwk
    if objective == "macro_recall":
        return macro_recall
    if objective == "balanced":
        return 0.5 * qwk + 0.5 * macro_recall
    raise ValueError(f"unknown objective: {objective}")


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


def _threshold_grid(num_classes: int, values: list[float]) -> list[tuple[float, ...]]:
    return [tuple(x) for x in itertools.product(values, repeat=num_classes - 1)]


def _search_thresholds(
    gt: np.ndarray,
    logits: np.ndarray,
    num_classes: int,
    values: list[float],
    objective: str,
    strategy: str,
) -> tuple[tuple[float, ...], float]:
    best_thresholds: tuple[float, ...] | None = None
    best_score = -1.0
    for thresholds in _threshold_grid(num_classes, values):
        if strategy == "calibrated_projection":
            pred = _ordinal_projection(logits, thresholds)
        elif strategy == "calibrated_middle_guard":
            pred = _middle_class_guard(logits, thresholds)
        else:
            raise ValueError(f"cannot search strategy: {strategy}")
        score = _score(gt, pred, num_classes, objective)
        if score > best_score:
            best_score = score
            best_thresholds = thresholds
    assert best_thresholds is not None
    return best_thresholds, best_score


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

    logits = {"dr": [], "me": []}
    gts = {"dr": [], "me": []}
    for images, targets in loader:
        outputs = model(images.to(device))
        for task in ("dr", "me"):
            logits[task].append(outputs[task].float().cpu())
            gts[task].append(targets[task])

    return {
        task: (
            torch.cat(gts[task]).numpy(),
            torch.cat(logits[task]).numpy(),
        )
        for task in ("dr", "me")
    }


def _concat(parts: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]], folds: list[int], task: str):
    gt = np.concatenate([parts[fold][task][0] for fold in folds])
    logits = np.concatenate([parts[fold][task][1] for fold in folds])
    return gt, logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_ablate_softmax_ldam.yaml")
    ap.add_argument("--cv-root", default="checkpoints/stage2_cv_ablate_softmax_ldam")
    ap.add_argument("--out", default="checkpoints/stage2_cv_ablate_softmax_ldam/projection_search_eval.json")
    ap.add_argument("--grid-min", type=float, default=0.50)
    ap.add_argument("--grid-max", type=float, default=0.95)
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--objective", choices=("balanced", "qwk", "macro_recall"), default="balanced")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["model"]["head"] != "softmax":
        raise ValueError("threshold search expects model.head=softmax")

    values = np.arange(args.grid_min, args.grid_max + args.grid_step / 2, args.grid_step)
    values = [float(round(x, 6)) for x in values]
    cv_root = Path(args.cv_root)
    folds = sorted(int(p.name.split("_", 1)[1]) for p in cv_root.glob("fold_*") if p.is_dir())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_data = {}
    for fold in folds:
        ckpt = cv_root / f"fold_{fold}" / "best_qwk.pth"
        print(f"predict fold={fold} ckpt={ckpt}")
        fold_data[fold] = _predict_fold(cfg, fold, ckpt, device)

    strategies = ("raw_softmax", "calibrated_projection", "calibrated_middle_guard")
    all_parts = {
        strategy: {"dr": {"gt": [], "pred": []}, "me": {"gt": [], "pred": []}}
        for strategy in strategies
    }
    per_fold = []
    thresholds_by_fold = []

    for fold in folds:
        calibration_folds = [x for x in folds if x != fold]
        fold_thresholds = {"fold": fold, "calibration_folds": calibration_folds, "tasks": {}}
        searched = {"calibrated_projection": {}, "calibrated_middle_guard": {}}
        for task in ("dr", "me"):
            gt_cal, logits_cal = _concat(fold_data, calibration_folds, task)
            num_classes = cfg["model"]["num_classes"][task]
            for strategy in ("calibrated_projection", "calibrated_middle_guard"):
                thresholds, score = _search_thresholds(
                    gt_cal,
                    logits_cal,
                    num_classes,
                    values,
                    args.objective,
                    strategy,
                )
                searched[strategy][task] = thresholds
                fold_thresholds["tasks"].setdefault(task, {})[strategy] = {
                    "thresholds": list(thresholds),
                    f"calibration_{args.objective}": score,
                }
        thresholds_by_fold.append(fold_thresholds)

        for strategy in strategies:
            row = {"fold": fold, "strategy": strategy}
            for task in ("dr", "me"):
                gt, logits = fold_data[fold][task]
                if strategy == "raw_softmax":
                    pred = _raw_softmax(logits)
                elif strategy == "calibrated_projection":
                    pred = _ordinal_projection(logits, searched[strategy][task])
                elif strategy == "calibrated_middle_guard":
                    pred = _middle_class_guard(logits, searched[strategy][task])
                else:
                    raise ValueError(strategy)
                all_parts[strategy][task]["gt"].append(gt)
                all_parts[strategy][task]["pred"].append(pred)
                row[f"{task}_qwk"] = quadratic_weighted_kappa(gt, pred)
                row[f"{task}_recall_per_class"] = per_class_recall(
                    gt,
                    pred,
                    cfg["model"]["num_classes"][task],
                )
                row[f"{task}_macro_recall"] = float(np.mean(row[f"{task}_recall_per_class"]))
            per_fold.append(row)

    summary = {
        "folds": folds,
        "grid": values,
        "objective": args.objective,
        "note": "For each eval fold, thresholds are searched on the other 4 held-out folds only.",
        "thresholds_by_fold": thresholds_by_fold,
        "per_fold": per_fold,
        "pooled": {},
    }
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
