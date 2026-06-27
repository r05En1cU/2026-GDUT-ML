"""Run binary-init 5-fold cross-validation on Messidor."""
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

from drnet.utils import load_config
from scripts._binary_common import run_binary_fold
from scripts.stage2_crossval import _infer_folds


def _fold_summary(results: list[dict]) -> dict:
    df = pd.DataFrame(results)
    summary = {"folds": df["fold"].tolist(), "n_folds": int(len(df))}
    for metric in (
        "binary_balanced",
        "binary_sensitivity",
        "binary_specificity",
        "sensitivity_dr",
        "specificity_dr",
        "fpr_dr",
        "fnr_dr",
        "sensitivity_me",
        "specificity_me",
        "fpr_me",
        "fnr_me",
    ):
        summary[metric] = {
            "mean": float(df[metric].mean()),
            "std": float(df[metric].std(ddof=1) if len(df) > 1 else 0.0),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/binary_init.yaml")
    ap.add_argument("--folds", nargs="*", type=int, help="Folds to run. Defaults to folds in folds.csv.")
    ap.add_argument("--resize", choices=("on", "off"),
                    help="Override data.resize in config for train/val transforms")
    ap.add_argument("--output-root", default="checkpoints/binary_init_cv")
    ap.add_argument("--log-root", default="runs/binary_init_cv")
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

        print(f"=== binary fold {fold} ===")
        result = run_binary_fold(fold_cfg, resize_override=args.resize)
        results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(output_root / "fold_metrics.csv", index=False)
    summary = _fold_summary(results)
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== binary cross-val summary ===")
    print(df[[
        "fold", "binary_balanced", "binary_sensitivity", "binary_specificity",
        "sensitivity_dr", "specificity_dr", "fpr_dr",
        "sensitivity_me", "specificity_me", "fpr_me", "epoch",
    ]].to_string(index=False))
    for metric, stats in summary.items():
        if isinstance(stats, dict):
            print(f"{metric}: mean={stats['mean']:.4f} std={stats['std']:.4f}")
    print(f"saved: {output_root / 'fold_metrics.csv'}")
    print(f"saved: {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
