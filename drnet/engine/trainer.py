"""训练循环:AMP、梯度累积、cosine LR(含 warmup)、按 val QWK 早停。"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..utils.checkpoint import save_checkpoint
from .metrics import per_class_recall, preds_from_outputs, quadratic_weighted_kappa


def _to_device_targets(targets: dict, device):
    return {k: v.to(device) for k, v in targets.items()}


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, epoch, device,
                    grad_accum: int = 1, channels_last: bool = False,
                    scheduler=None, log=print) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    pending = 0
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        targets = _to_device_targets(targets, device)
        with torch.autocast(device_type=device.split(":")[0], enabled=scaler.is_enabled()):
            outputs = model(images)
            loss, _ = loss_fn(outputs, targets, epoch)
            loss = loss / grad_accum
        scaler.scale(loss).backward()
        pending += 1
        if pending == grad_accum:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            pending = 0
        running += loss.item() * grad_accum
    if pending > 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
    return {"train_loss": running / max(len(loader), 1)}


@torch.no_grad()
def validate(model, loader, device, head_mode, num_classes, channels_last=False) -> dict:
    model.eval()
    preds = {"dr": [], "me": []}
    gts = {"dr": [], "me": []}
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        outputs = model(images)
        for t in ("dr", "me"):
            preds[t].append(preds_from_outputs(outputs[t], head_mode).cpu())
            gts[t].append(targets[t])
    res = {}
    for t in ("dr", "me"):
        p = torch.cat(preds[t]).numpy()
        g = torch.cat(gts[t]).numpy()
        res[f"qwk_{t}"] = quadratic_weighted_kappa(g, p)
        res[f"recall_{t}"] = per_class_recall(g, p, num_classes[t])
        res[f"macro_recall_{t}"] = float(np.mean(res[f"recall_{t}"]))
    res["qwk_mean"] = (res["qwk_dr"] + res["qwk_me"]) / 2
    res["macro_recall"] = (res["macro_recall_dr"] + res["macro_recall_me"]) / 2
    res["balanced"] = 0.5 * res["qwk_mean"] + 0.5 * res["macro_recall"]
    return res


def _build_scheduler(optimizer, epochs, warmup, steps_per_epoch):
    total = epochs * steps_per_epoch
    wu = warmup * steps_per_epoch

    def lr_lambda(step):
        if step < wu:
            return (step + 1) / max(wu, 1)
        prog = (step - wu) / max(total - wu, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def fit(cfg, model, loss_fn, train_loader, val_loader, device):
    """主训练循环。返回 best 指标 dict。"""
    tr = cfg["train"]
    head_mode = cfg["model"]["head"]
    num_classes = cfg["model"]["num_classes"]
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(cfg["output"].get("log_dir", os.path.join(out_dir, "runs")))

    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                                  weight_decay=tr["weight_decay"])
    grad_accum = tr.get("grad_accum", 1)
    scheduler = _build_scheduler(
        optimizer,
        tr["epochs"],
        tr.get("warmup_epochs", 0),
        max(math.ceil(len(train_loader) / max(grad_accum, 1)), 1),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=tr.get("amp", False) and device.startswith("cuda"))

    metric_name = tr.get("early_stop_metric", "qwk_mean")
    best_metric = -1.0
    best = {"qwk_mean": -1.0}
    patience, bad = tr.get("early_stop_patience", 10**9), 0
    for epoch in range(tr["epochs"]):
        log = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, epoch, device,
                              grad_accum=grad_accum,
                              channels_last=tr.get("channels_last", False),
                              scheduler=scheduler)
        val = validate(model, val_loader, device, head_mode, num_classes,
                       channels_last=tr.get("channels_last", False))
        writer.add_scalar("train/loss", log["train_loss"], epoch)
        for k in ("qwk_dr", "qwk_me", "qwk_mean",
                  "macro_recall_dr", "macro_recall_me", "macro_recall",
                  "balanced"):
            writer.add_scalar(f"val/{k}", val[k], epoch)
        print(f"[epoch {epoch}] loss={log['train_loss']:.4f} "
              f"qwk_dr={val['qwk_dr']:.4f} qwk_me={val['qwk_me']:.4f} "
              f"qwk_mean={val['qwk_mean']:.4f} macro_recall={val['macro_recall']:.4f} "
              f"balanced={val['balanced']:.4f}")

        metric = val[metric_name]
        if metric > best_metric:
            best_metric = metric
            best = {**val, "epoch": epoch, "best_metric": metric, "best_metric_name": metric_name}
            save_checkpoint(model, os.path.join(out_dir, "best_qwk.pth"),
                            extra={"epoch": epoch, "val": val,
                                   "best_metric": metric, "best_metric_name": metric_name})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"早停于 epoch {epoch}(patience={patience})")
                break
    writer.close()
    return best
