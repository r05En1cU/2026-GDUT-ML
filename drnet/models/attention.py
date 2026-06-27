"""注意力模块(CANet 思路)。

- DiseaseSpecificAttention: 类 CBAM 的通道+空间注意力,为单个任务从共享特征中挑选有用特征。
- CrossDiseaseAttention: 用一个任务的全局特征调制另一个任务,显式建模 DR<->ME 关系。

参考 reference/CANet/models/cbam.py。这里给出简洁自包含实现。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * att


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class DiseaseSpecificAttention(nn.Module):
    """疾病特异注意力:CBAM 式通道+空间注意力,输出仍为特征图 [B,C,H,W]。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction)
        self.sa = _SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class CrossDiseaseAttention(nn.Module):
    """疾病依赖注意力:用对方任务的全局上下文生成通道门控,互相调制。

    输入两路特征图,输出两路被对方引导后的特征图。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate_dr = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )
        self.gate_me = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, feat_dr: torch.Tensor, feat_me: torch.Tensor):
        b, c, _, _ = feat_dr.shape
        ctx_dr = feat_dr.mean(dim=(2, 3))   # DR 全局上下文
        ctx_me = feat_me.mean(dim=(2, 3))   # ME 全局上下文
        # 用 ME 上下文调制 DR,用 DR 上下文调制 ME(交叉)
        feat_dr = feat_dr * self.gate_me(ctx_me).view(b, c, 1, 1)
        feat_me = feat_me * self.gate_dr(ctx_dr).view(b, c, 1, 1)
        return feat_dr, feat_me
