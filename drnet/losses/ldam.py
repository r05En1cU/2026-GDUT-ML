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

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        margin = self.m_list.to(device=logits.device, dtype=logits.dtype)[target]
        logits_m = logits.clone()
        idx = torch.arange(logits.size(0), device=logits.device)
        logits_m[idx, target] = logits[idx, target] - margin

        weight = None
        if self.weight is not None:
            weight = self.weight.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(self.s * logits_m, target, weight=weight)


def drw_weights(
    cls_num_list: list[int],
    epoch: int,
    defer_epoch: int,
    beta: float = 0.9999,
) -> torch.Tensor | None:
    """Return effective-number class weights after ``defer_epoch``."""
    if epoch < defer_epoch:
        return None
    cls = np.array(cls_num_list, dtype=np.float64)
    eff_num = 1.0 - np.power(beta, cls)
    weights = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    weights = weights / weights.sum() * len(cls)
    return torch.tensor(weights, dtype=torch.float32)
