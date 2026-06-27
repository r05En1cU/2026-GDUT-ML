"""Grad-CAM 可解释性。基于 pytorch-grad-cam(pip: grad-cam)。

对 ConvNeXt 最后一层卷积出热力图,验证模型是否关注真实病灶。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


def _target_layer(model):
    """取 ConvNeXt-Tiny 最后一个 stage 的最后一个 block 作为目标层。"""
    # timm convnext: model.backbone.model.stages[-1]
    return model.backbone.model.stages[-1]


class _HeadWrapper(torch.nn.Module):
    """把多任务输出包成单 logit 张量,供 grad-cam 取目标类别。"""

    def __init__(self, model, task: str):
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, x):
        return self.model(x)[self.task]


def gradcam_for(model, image: torch.Tensor, target_head: str, target_class: int):
    """对单张 image([1,3,H,W] 已归一化)生成 CAM。返回 [H,W] in [0,1]。"""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    wrapper = _HeadWrapper(model, target_head)
    cam = GradCAM(model=wrapper, target_layers=[_target_layer(model)])
    grayscale = cam(input_tensor=image,
                    targets=[ClassifierOutputTarget(target_class)])[0]
    return grayscale


def save_overlay(rgb_image: np.ndarray, cam: np.ndarray, out_path: str):
    """把 CAM 叠到原 RGB 图(uint8)上并保存。"""
    from pytorch_grad_cam.utils.image import show_cam_on_image
    img = rgb_image.astype(np.float32) / 255.0
    vis = show_cam_on_image(img, cam, use_rgb=True)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return out_path
