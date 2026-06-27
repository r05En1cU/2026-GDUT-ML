"""Stage-2 entry point for one fold of Messidor fine-tuning."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._stage2_common import run_stage2_fold
from drnet.utils import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_finetune.yaml")
    ap.add_argument("--fold", type=int, help="Override data.fold from config")
    ap.add_argument("--resize", choices=("on", "off"),
                    help="Override data.resize in config for train/val transforms")
    ap.add_argument("--output-dir", help="Override output.dir in config")
    ap.add_argument("--log-dir", help="Override output.log_dir in config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    if args.output_dir or args.log_dir:
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("output", {})
        if args.output_dir:
            cfg["output"]["dir"] = args.output_dir
        if args.log_dir:
            cfg["output"]["log_dir"] = args.log_dir

    best = run_stage2_fold(cfg, resize_override=args.resize)
    print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
