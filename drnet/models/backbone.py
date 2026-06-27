"""ConvNeXt-Tiny 主干封装(基于 timm)。输出特征图,供注意力与池化使用。"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ConvNeXtTinyBackbone(nn.Module):
    """timm convnext_tiny,去掉分类头,forward 返回最后一层特征图 [B, C, H, W]。"""

    def __init__(self, pretrained: bool = True, grad_checkpoint: bool = False):
        super().__init__()
        # features_only=False + num_classes=0, global_pool='' 保留空间维度的特征图
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        if grad_checkpoint and hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(True)
        self._out_channels = self.model.num_features  # 768 for tiny

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm convnext forward_features 返回 [B, C, H, W]
        return self.model.forward_features(x)
