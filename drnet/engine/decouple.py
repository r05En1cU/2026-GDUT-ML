"""解耦训练(cRT / LWS / tau-norm)。参考 reference/classifier-balancing。

Stage-2 训完若 R3/M2 召回仍弱,冻结主干、只重训/校准分类头。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def freeze_backbone(model) -> None:
    """冻结主干与注意力,仅留两个头可训(cRT 用)。"""
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for name in ("att_dr", "att_me", "cross"):
        mod = getattr(model, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(False)


def crt_finetune(model, balanced_loader, loss_fn, device, epochs: int = 10, lr: float = 1e-3):
    """cRT:冻主干,在类别均衡 loader 上重训分类头若干 epoch。"""
    freeze_backbone(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    for ep in range(epochs):
        for images, targets in balanced_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            outputs = model(images)
            loss, _ = loss_fn(outputs, targets, epoch=10**9)  # 已是均衡数据,无需 DRW
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def lws_calibrate(model, tau: float = 1.0):
    """LWS / tau-norm:按权重范数缩放各头的分类权重(无需再训练)。

    对每个头的输出权重矩阵 W,做 W <- W / ||W_row||^tau。
    """
    for head in (model.head_dr, model.head_me):
        fc = getattr(head, "fc", None)
        if fc is None:  # CORAL 头是 proj,跳过
            continue
        w = fc.weight.data
        norm = w.norm(dim=1, keepdim=True).clamp_min(1e-8)
        fc.weight.data = w / norm.pow(tau)
    return model
