"""多任务网络总装: backbone -> (DR/ME 特异注意力) -> 跨任务注意力 -> 池化 -> DR头 + ME头。"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossDiseaseAttention, DiseaseSpecificAttention
from .backbone import ConvNeXtTinyBackbone
from .heads import build_head


class MultiTaskNet(nn.Module):
    """输出 dict: {'dr': logits, 'me': logits}。

    Args (来自 config.model):
        backbone: 目前支持 'convnext_tiny'
        pretrained_imagenet, grad_checkpoint
        attention: 'none' | 'specific' | 'cross'
        head: 'corn' | 'coral' | 'softmax'
        num_classes: {'dr': 4, 'me': 3}
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg
        self.attention_mode = m.get("attention", "cross")
        self.head_mode = m.get("head", "corn")
        nc = m["num_classes"]

        self.backbone = ConvNeXtTinyBackbone(
            pretrained=m.get("pretrained_imagenet", True),
            grad_checkpoint=m.get("grad_checkpoint", False),
        )
        c = self.backbone.out_channels

        if self.attention_mode in ("specific", "cross"):
            self.att_dr = DiseaseSpecificAttention(c)
            self.att_me = DiseaseSpecificAttention(c)
        if self.attention_mode == "cross":
            self.cross = CrossDiseaseAttention(c)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_dr = build_head(self.head_mode, c, nc["dr"])
        self.head_me = build_head(self.head_mode, c, nc["me"])

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)                 # [B, C, H, W]

        if self.attention_mode == "none":
            f_dr = f_me = feat
        else:
            f_dr = self.att_dr(feat)
            f_me = self.att_me(feat)
            if self.attention_mode == "cross":
                f_dr, f_me = self.cross(f_dr, f_me)

        v_dr = self.pool(f_dr).flatten(1)        # [B, C]
        v_me = self.pool(f_me).flatten(1)
        return {"dr": self.head_dr(v_dr), "me": self.head_me(v_me)}

    @torch.no_grad()
    def load_backbone(self, ckpt_path: str, map_location="cpu"):
        """只加载主干权重(忽略 Stage-1 的 5 类头与任何不匹配键)。"""
        state = torch.load(ckpt_path, map_location=map_location)
        state = state.get("backbone", state.get("model", state))
        # 仅保留 backbone.* 键
        bb = {k.replace("backbone.", "", 1): v for k, v in state.items()
              if k.startswith("backbone.")} or state
        missing, unexpected = self.backbone.load_state_dict(bb, strict=False)
        return missing, unexpected

    @torch.no_grad()
    def load_shared(self, ckpt_path: str, map_location="cpu"):
        """加载 backbone + attention 共享权重,自动跳过任务头。"""
        state = torch.load(ckpt_path, map_location=map_location)
        state = state.get("model", state)
        prefixes = ("backbone.", "att_dr.", "att_me.", "cross.")
        shared = {k: v for k, v in state.items() if k.startswith(prefixes)}
        missing, unexpected = self.load_state_dict(shared, strict=False)
        return missing, unexpected
