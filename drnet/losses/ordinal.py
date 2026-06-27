"""序数损失:CORAL 与 CORN。

约定:序数头输出 K-1 个 logits;真实等级 y in [0, K-1]。
扩展二元标签 levels[k] = 1[y > k], k=0..K-2。

CORAL: 各二分类用全部样本做加权 BCE。
CORN:  第 k 个二分类只在"满足 y > k-1"的子集上训练(k>=1),用条件概率链。
       预测时 P(y>k) = Π_{j<=k} sigmoid(logit_j)。

实现自包含;与 reference/coral-pytorch 等价,便于无依赖运行。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def label_to_levels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """y:[B] -> levels:[B, K-1], levels[:,k] = 1[y > k]。"""
    ks = torch.arange(num_classes - 1, device=y.device).view(1, -1)
    return (y.view(-1, 1) > ks).float()


def coral_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int,
               weights: torch.Tensor | None = None) -> torch.Tensor:
    """CORAL: 全样本加权 BCE。weights 可为各 level 的重要性权重 [K-1]。"""
    levels = label_to_levels(y, num_classes)
    losses = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")  # [B, K-1]
    if weights is not None:
        losses = losses * weights.view(1, -1)
    return losses.sum(dim=1).mean()


def corn_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int,
              weights: torch.Tensor | None = None) -> torch.Tensor:
    """CORN: 条件训练。第 k 列只在 y > k-1 的样本上算 BCE(k>=1);第 0 列用全部样本。"""
    total = logits.new_zeros(())
    n_terms = 0
    for k in range(num_classes - 1):
        if k == 0:
            mask = torch.ones_like(y, dtype=torch.bool)
        else:
            mask = y > (k - 1)         # 条件子集:已经 > k-1
        if mask.sum() == 0:
            continue
        target = (y[mask] > k).float()
        level_loss = F.binary_cross_entropy_with_logits(
            logits[mask, k], target, reduction="mean")
        if weights is not None:
            level_loss = level_loss * weights[k]
        total = total + level_loss
        n_terms += 1
    return total / max(n_terms, 1)


@torch.no_grad()
def ordinal_logits_to_label(logits: torch.Tensor, mode: str = "corn") -> torch.Tensor:
    """由序数 logits 还原等级。

    corn:  P(y>k) = 累乘 sigmoid; 等级 = Σ 1[P>0.5]
    coral: P(y>k) = sigmoid(logit_k); 等级 = Σ 1[P>0.5]
    """
    probs = torch.sigmoid(logits)
    if mode == "corn":
        probs = torch.cumprod(probs, dim=1)
    return (probs > 0.5).sum(dim=1)
