"""LDAM 损失 + DRW(延迟重加权)。参考 reference/LDAM-DRW/losses.py。

仅用于 softmax 头(需要 K 类 logits)。序数头(corn/coral)走 ordinal.py 的损失,
长尾通过 DRW 的类别权重传入 binary 任务的 pos_weight 间接处理(见 multitask.py)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss(K 类 softmax)。"""

    def __init__(self, cls_num_list: list[int], max_m: float = 0.5, s: float = 30.0,
                 weight: torch.Tensor | None = None):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.maximum(cls_num_list, 1)))
        m_list = m_list * (max_m / m_list.max())
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        m = self.m_list.to(logits.device)[target]            # [B] 每样本对应类的间隔
        x_m = logits.clone()
        idx = torch.arange(logits.size(0), device=logits.device)
        x_m[idx, target] = logits[idx, target] - m           # 真类 logit 减间隔
        return F.cross_entropy(self.s * x_m, target,
                               weight=self.weight.to(logits.device) if self.weight is not None else None)


def drw_weights(cls_num_list: list[int], epoch: int, defer_epoch: int,
                beta: float = 0.9999) -> torch.Tensor | None:
    """延迟重加权:defer_epoch 之前返回 None(不加权),之后返回有效样本数的类别权重。"""
    if epoch < defer_epoch:
        return None
    cls = np.array(cls_num_list, dtype=np.float64)
    eff_num = 1.0 - np.power(beta, cls)
    w = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    w = w / w.sum() * len(cls)
    return torch.tensor(w, dtype=torch.float32)
