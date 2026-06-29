"""LDAM loss and deferred re-weighting utilities."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss for softmax heads."""

    def __init__(
        self,
        cls_num_list: list[int],
        max_m: float = 0.5,
        s: float = 30.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.maximum(cls_num_list, 1)))
        m_list = m_list * (max_m / m_list.max())
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s
        self.weight = weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        margin_scale: float = 1.0,
    ) -> torch.Tensor:
        margin = self.m_list.to(device=logits.device, dtype=logits.dtype)[target]
        margin = margin * float(margin_scale)
        logits_m = logits.clone()
        idx = torch.arange(logits.size(0), device=logits.device)
        logits_m[idx, target] = logits[idx, target] - margin

        weight = None
        if self.weight is not None:
            weight = self.weight.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(self.s * logits_m, target, weight=weight)


def schedule_alpha(
    progress: float,
    defer_epoch: float,
    ramp_epochs: float = 0.0,
    schedule: str = "step",
) -> float:
    """Return a smooth 0..1 coefficient for deferred loss components."""
    if progress < defer_epoch:
        return 0.0
    if ramp_epochs <= 0 or schedule == "step":
        return 1.0

    x = (progress - defer_epoch) / max(ramp_epochs, 1e-8)
    x = float(np.clip(x, 0.0, 1.0))
    if schedule == "linear":
        return x
    if schedule == "cosine":
        return float(0.5 - 0.5 * np.cos(np.pi * x))
    if schedule == "smoothstep":
        return float(x * x * (3.0 - 2.0 * x))
    raise ValueError(f"unknown DRW schedule: {schedule}")


def effective_number_weights(
    cls_num_list: list[int],
    beta: float = 0.9999,
) -> torch.Tensor:
    """Class-balanced effective-number weights normalized to mean 1."""
    cls = np.array(cls_num_list, dtype=np.float64)
    eff_num = 1.0 - np.power(beta, cls)
    weights = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    weights = weights / weights.sum() * len(cls)
    return torch.tensor(weights, dtype=torch.float32)


def drw_weights(
    cls_num_list: list[int],
    epoch: float,
    defer_epoch: float,
    beta: float = 0.9999,
    ramp_epochs: float = 0.0,
    schedule: str = "step",
) -> torch.Tensor | None:
    """Return deferred effective-number weights with optional smooth ramp."""
    alpha = schedule_alpha(epoch, defer_epoch, ramp_epochs, schedule)
    if alpha <= 0:
        return None
    target = effective_number_weights(cls_num_list, beta)
    if alpha >= 1:
        return target
    uniform = torch.ones_like(target)
    return (1.0 - alpha) * uniform + alpha * target
