"""免数据冒烟测试:用随机张量跑通 前向 -> 双头损失 -> 反传 -> 指标。

装好环境后一条命令自检(默认 CPU、不下载预训练权重):
    python scripts/smoke_test.py
    python scripts/smoke_test.py --head coral --attention cross
    python scripts/smoke_test.py --head softmax --loss ldam

通过标准:三种 head × 三种 attention 均能前向、损失有限、可反传,指标可计算。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnet.engine import preds_from_outputs
from drnet.engine.metrics import per_class_recall, quadratic_weighted_kappa
from drnet.losses import MultiTaskLoss
from drnet.models import MultiTaskNet


def _fake_batch(b, size, nc, device):
    x = torch.randn(b, 3, size, size, device=device)
    targets = {
        "dr": torch.randint(0, nc["dr"], (b,), device=device),
        "me": torch.randint(0, nc["me"], (b,), device=device),
    }
    return x, targets


def run_one(head, attention, loss_type, device, size=128, b=4):
    nc = {"dr": 4, "me": 3}
    model_cfg = {
        "backbone": "convnext_tiny",
        "pretrained_imagenet": False,   # 不下载权重
        "grad_checkpoint": False,
        "attention": attention,
        "head": head,
        "num_classes": nc,
    }
    model = MultiTaskNet(model_cfg).to(device)
    # softmax 头才用 ldam/focal;序数头走序数损失
    eff_loss = loss_type if head == "softmax" else "ce"
    cls_num = {"dr": [10, 6, 3, 2], "me": [10, 5, 2]}
    loss_cfg = {"type": eff_loss, "head_mode": head, "num_classes": nc,
                "drw_defer_epoch": 0, "task_weights": {"dr": 1.0, "me": 1.0},
                "ldam_max_m": 0.5, "ldam_s": 30.0}
    loss_fn = MultiTaskLoss(loss_cfg, cls_num).to(device)

    x, targets = _fake_batch(b, size, nc, device)
    out = model(x)
    # 形状检查
    exp_dr = nc["dr"] if head == "softmax" else nc["dr"] - 1
    exp_me = nc["me"] if head == "softmax" else nc["me"] - 1
    assert out["dr"].shape == (b, exp_dr), out["dr"].shape
    assert out["me"].shape == (b, exp_me), out["me"].shape

    loss, parts = loss_fn(out, targets, epoch=1)
    assert torch.isfinite(loss), "loss 非有限"
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    assert grad_ok, "无有效梯度"

    # 指标可计算
    p_dr = preds_from_outputs(out["dr"], head).cpu().numpy()
    g_dr = targets["dr"].cpu().numpy()
    qwk = quadratic_weighted_kappa(g_dr, p_dr) if len(set(g_dr)) > 1 else float("nan")
    _ = per_class_recall(g_dr, p_dr, nc["dr"])
    return loss.item(), parts, qwk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None, choices=["corn", "coral", "softmax"])
    ap.add_argument("--attention", default=None, choices=["none", "specific", "cross"])
    ap.add_argument("--loss", default="ldam", choices=["ce", "focal", "ldam"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    heads = [args.head] if args.head else ["corn", "coral", "softmax"]
    atts = [args.attention] if args.attention else ["none", "specific", "cross"]

    n_pass = n_total = 0
    for head in heads:
        for att in atts:
            n_total += 1
            try:
                loss, parts, qwk = run_one(head, att, args.loss, device)
                print(f"[PASS] head={head:7s} att={att:8s} "
                      f"loss={loss:.4f} (dr={parts['loss_dr']:.3f}, me={parts['loss_me']:.3f}) "
                      f"qwk_dr={qwk:.3f}")
                n_pass += 1
            except Exception as e:  # noqa
                print(f"[FAIL] head={head:7s} att={att:8s} -> {type(e).__name__}: {e}")

    print(f"\n== {n_pass}/{n_total} 组合通过 ==")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
