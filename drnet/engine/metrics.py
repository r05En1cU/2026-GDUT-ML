"""评估指标:QWK、每类 recall、混淆矩阵,以及从模型输出还原预测等级。"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, recall_score

from ..losses.ordinal import ordinal_logits_to_label


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def per_class_recall(y_true, y_pred, num_classes: int) -> list[float]:
    return recall_score(y_true, y_pred, labels=list(range(num_classes)),
                        average=None, zero_division=0).tolist()


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


@torch.no_grad()
def preds_from_outputs(logits: torch.Tensor, head_mode: str) -> torch.Tensor:
    """把一个头的 logits 转成预测等级 [B]。"""
    if head_mode in ("corn", "coral"):
        return ordinal_logits_to_label(logits, mode=head_mode)
    return logits.argmax(dim=1)
