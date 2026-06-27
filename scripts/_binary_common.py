"""Shared helpers for Messidor binary-init workflows."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from drnet.data import MessidorMultiTaskDataset, build_transforms
from drnet.engine import fit_binary
from drnet.losses import MultiTaskLoss
from drnet.models import MultiTaskNet
from drnet.utils import set_seed
from scripts._stage2_common import resolve_resize


def run_binary_fold(cfg: dict, resize_override: str | None = None) -> dict[str, Any]:
    """Run one binary-init fold and return the best validation summary."""
    cfg = copy.deepcopy(cfg)
    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = cfg["data"]
    fold = int(d["fold"])
    resize = resolve_resize(d, resize_override)

    tf_train = build_transforms(d["image_size"], train=True, resize=resize)
    tf_val = build_transforms(d["image_size"], train=False, resize=resize)
    ds_train = MessidorMultiTaskDataset(
        d["folds_csv"],
        d["root"],
        fold,
        "train",
        transform=tf_train,
        image_size=d["image_size"],
        target_mode="binary",
    )
    ds_val = MessidorMultiTaskDataset(
        d["folds_csv"],
        d["root"],
        fold,
        "val",
        transform=tf_val,
        image_size=d["image_size"],
        target_mode="binary",
    )
    print(f"binary fold={fold} resize={'on' if resize else 'off'}")

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=d.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=d.get("num_workers", 4),
        pin_memory=True,
    )

    model = MultiTaskNet(cfg["model"]).to(device)
    if cfg["model"].get("backbone_ckpt"):
        missing, unexpected = model.load_backbone(cfg["model"]["backbone_ckpt"])
        print(f"loaded Stage-1 backbone (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("using ImageNet backbone init")
    if cfg["train"].get("channels_last"):
        model = model.to(memory_format=torch.channels_last)

    cls_num = {
        "dr": ds_train.class_counts("dr", cfg["model"]["num_classes"]["dr"]),
        "me": ds_train.class_counts("me", cfg["model"]["num_classes"]["me"]),
    }
    print(f"binary class counts DR={cls_num['dr']} ME={cls_num['me']}")

    loss_cfg = {**cfg["loss"], "head_mode": cfg["model"]["head"],
                "num_classes": cfg["model"]["num_classes"]}
    loss_fn = MultiTaskLoss(loss_cfg, cls_num).to(device)

    best = fit_binary(cfg, model, loss_fn, train_loader, val_loader, device)
    result = dict(best)
    result.update({
        "fold": fold,
        "resize": resize,
        "device": device,
        "train_size": len(ds_train),
        "val_size": len(ds_val),
        "output_dir": cfg["output"]["dir"],
        "log_dir": cfg["output"].get("log_dir"),
        "checkpoint_path": str(Path(cfg["output"]["dir"]) / "best_binary.pth"),
    })
    return result
