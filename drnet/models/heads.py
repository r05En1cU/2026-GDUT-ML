"""分类头:CORAL / CORN 序数头 + 普通 softmax 头(消融对照)。

CORAL: 共享投影向量 w,K-1 个独立偏置 -> 结构上保证单调一致。
CORN:  各阈值独立全权重,训练时用条件概率链保证一致(损失在 losses/ordinal.py 中实现)。
两者输出都是 [B, K-1] 的有序 logits;softmax 头输出 [B, K]。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OrdinalHead(nn.Module):
    """序数头,输出 K-1 个有序 logits。

    mode='coral': Linear(in,1) 给出共享投影 w^T x,再加 K-1 个可学习偏置。
    mode='corn':  Linear(in, K-1),每个阈值独立权重(配合 corn_loss 的条件训练)。
    """

    def __init__(self, in_features: int, num_classes: int, mode: str = "corn"):
        super().__init__()
        assert mode in ("coral", "corn")
        self.mode = mode
        self.num_classes = num_classes
        if mode == "coral":
            self.proj = nn.Linear(in_features, 1, bias=False)
            self.bias = nn.Parameter(torch.zeros(num_classes - 1))
        else:  # corn
            self.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "coral":
            return self.proj(x) + self.bias       # [B, K-1] 广播
        return self.fc(x)                          # [B, K-1]


class SoftmaxHead(nn.Module):
    """普通分类头,输出 [B, K]。消融对照用。"""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def build_head(mode: str, in_features: int, num_classes: int) -> nn.Module:
    """根据 mode 构建头:'corn'|'coral'->OrdinalHead, 'softmax'->SoftmaxHead。"""
    if mode in ("corn", "coral"):
        return OrdinalHead(in_features, num_classes, mode=mode)
    if mode == "softmax":
        return SoftmaxHead(in_features, num_classes)
    raise ValueError(f"未知 head mode: {mode}")
