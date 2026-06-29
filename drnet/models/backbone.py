"""Backbone wrappers built on timm."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class TimmFeatureBackbone(nn.Module):
    """Wrap a timm model and return its final feature map [B, C, H, W]."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        if grad_checkpoint and hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(True)
        self._out_channels = self.model.num_features

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model.forward_features(x)
        if feat.ndim != 4:
            raise RuntimeError(f"expected 4D feature map, got shape={tuple(feat.shape)}")
        return feat


class ConvNeXtTinyBackbone(TimmFeatureBackbone):
    """ConvNeXt-Tiny backbone."""

    def __init__(self, pretrained: bool = True, grad_checkpoint: bool = False):
        super().__init__(
            "convnext_tiny",
            pretrained=pretrained,
            grad_checkpoint=grad_checkpoint,
        )


class ResNet50Backbone(TimmFeatureBackbone):
    """ResNet-50 backbone."""

    def __init__(self, pretrained: bool = True, grad_checkpoint: bool = False):
        super().__init__(
            "resnet50",
            pretrained=pretrained,
            grad_checkpoint=grad_checkpoint,
        )
