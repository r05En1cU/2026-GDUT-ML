"""多任务总损失: L = w_dr*L_dr + w_me*L_me。

按 head_mode 选择每个头的损失:
  softmax -> CrossEntropy / Focal / LDAM(+DRW 类别权重)
  corn/coral -> 对应序数损失(长尾通过 DRW 类别权重缩放 per-level 重要性)
支持固定任务权重或 Kendall 不确定性自动加权。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ldam import LDAMLoss, drw_weights
from .ordinal import coral_loss, corn_loss


def _focal_loss(logits, target, gamma: float = 2.0, weight=None):
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: dict, cls_num: dict[str, list[int]]):
        """
        Args:
            cfg: config.loss 段 + 顶层 model.head。
            cls_num: {'dr': [...], 'me': [...]} 各任务训练集类别计数(用于 LDAM/DRW)。
        """
        super().__init__()
        self.head_mode = cfg["head_mode"]
        self.loss_type = cfg.get("type", "ce")
        self.num_classes = cfg["num_classes"]            # {'dr':4,'me':3}
        self.cls_num = cls_num
        self.drw_defer = cfg.get("drw_defer_epoch", 10**9)
        self.uncertainty = cfg.get("uncertainty", False)
        tw = cfg.get("task_weights", {"dr": 1.0, "me": 1.0})
        if self.uncertainty:
            # Kendall: 学习 log(sigma^2),损失自动平衡
            self.log_var = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer("w_dr", torch.tensor(float(tw["dr"])))
            self.register_buffer("w_me", torch.tensor(float(tw["me"])))
        if self.loss_type == "ldam":
            self.ldam = {t: LDAMLoss(cls_num[t], cfg.get("ldam_max_m", 0.5),
                                     cfg.get("ldam_s", 30.0)) for t in ("dr", "me")}

    def _head_loss(self, task: str, logits: torch.Tensor, target: torch.Tensor,
                   epoch: int) -> torch.Tensor:
        K = self.num_classes[task]
        if self.head_mode == "corn":
            w = drw_weights(self.cls_num[task], epoch, self.drw_defer)
            lvl_w = w[1:].to(logits.device) if w is not None else None
            return corn_loss(logits, target, K, weights=lvl_w)
        if self.head_mode == "coral":
            w = drw_weights(self.cls_num[task], epoch, self.drw_defer)
            lvl_w = w[1:].to(logits.device) if w is not None else None  # 近似:用类权重的尾部
            return coral_loss(logits, target, K, weights=lvl_w)
        # softmax 头
        weight = drw_weights(self.cls_num[task], epoch, self.drw_defer)
        weight = weight.to(logits.device) if weight is not None else None
        if self.loss_type == "ldam":
            self.ldam[task].weight = weight
            return self.ldam[task](logits, target)
        if self.loss_type == "focal":
            return _focal_loss(logits, target, weight=weight)
        return F.cross_entropy(logits, target, weight=weight)

    def forward(self, outputs: dict, targets: dict, epoch: int):
        l_dr = self._head_loss("dr", outputs["dr"], targets["dr"], epoch)
        l_me = self._head_loss("me", outputs["me"], targets["me"], epoch)
        if self.uncertainty:
            p_dr = torch.exp(-self.log_var[0]); p_me = torch.exp(-self.log_var[1])
            total = p_dr * l_dr + self.log_var[0] + p_me * l_me + self.log_var[1]
        else:
            total = self.w_dr * l_dr + self.w_me * l_me
        return total, {"loss_dr": l_dr.item(), "loss_me": l_me.item()}
