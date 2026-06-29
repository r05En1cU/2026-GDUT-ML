"""Run stage-2 5-fold cross-validation on Messidor."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._stage2_common import run_stage2_fold
from drnet.utils import load_config


def _infer_folds(folds_csv: str) -> list[int]:
    df = pd.read_csv(folds_csv, usecols=["fold"])
    folds = sorted(int(v) for v in df["fold"].dropna().unique().tolist())
    if not folds:
        raise ValueError(f"no folds found in {folds_csv}")
    return folds


def _fold_summary(results: list[dict]) -> dict:
    df = pd.DataFrame(results)
    summary = {"folds": df["fold"].tolist(), "n_folds": int(len(df))}
    for metric in ("accuracy_dr", "accuracy_me", "accuracy_mean",
                   "qwk_dr", "qwk_me", "qwk_mean",
                   "macro_recall_dr", "macro_recall_me", "macro_recall",
                   "min_recall_dr", "min_recall_me", "min_recall",
                   "balanced"):
        summary[metric] = {
            "mean": float(df[metric].mean()),
            "std": float(df[metric].std(ddof=1) if len(df) > 1 else 0.0),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_finetune.yaml")
    ap.add_argument("--folds", nargs="*", type=int, help="Folds to run. Defaults to folds in folds.csv.")
    ap.add_argument("--resize", choices=("on", "off"),
                    help="Override data.resize in config for train/val transforms")
    ap.add_argument("--output-root", default="checkpoints/stage2_cv")
    ap.add_argument("--log-root", default="runs/stage2_cv")
    ap.add_argument("--shared-cv-root",
                    help="Load per-fold shared init from <root>/fold_i/best_binary.pth")
    args = ap.parse_args()

    cfg = load_config(args.config)
    folds = args.folds if args.folds else _infer_folds(cfg["data"]["folds_csv"])
    output_root = Path(args.output_root)
    log_root = Path(args.log_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for fold in folds:
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["data"]["fold"] = int(fold)
        fold_cfg.setdefault("output", {})
        fold_cfg["output"]["dir"] = str(output_root / f"fold_{fold}")
        fold_cfg["output"]["log_dir"] = str(log_root / f"fold_{fold}")
        if args.shared_cv_root:
            fold_cfg["model"]["shared_ckpt"] = str(Path(args.shared_cv_root) / f"fold_{fold}" / "best_binary.pth")

        print(f"=== fold {fold} ===")
        result = run_stage2_fold(fold_cfg, resize_override=args.resize)
        results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(output_root / "fold_metrics.csv", index=False)
    summary = _fold_summary(results)
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== cross-val summary ===")
    print(df[["fold", "accuracy_dr", "accuracy_me", "accuracy_mean",
              "qwk_dr", "qwk_me", "qwk_mean",
              "macro_recall", "min_recall", "balanced", "epoch"]].to_string(index=False))
    for metric, stats in summary.items():
        if isinstance(stats, dict):
            print(f"{metric}: mean={stats['mean']:.4f} std={stats['std']:.4f}")
    print(f"saved: {output_root / 'fold_metrics.csv'}")
    print(f"saved: {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
