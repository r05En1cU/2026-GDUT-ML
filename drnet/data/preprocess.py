"""眼底图预处理:圆形裁剪 + Ben Graham 对比度归一化。

参考 Kaggle DR 竞赛(Ben Graham)的标准做法,让微动脉瘤、渗出等微小病灶更可见。
"""
from __future__ import annotations

import cv2
import numpy as np


def _crop_to_content(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """按灰度阈值裁掉眼底图四周黑边。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return img
    coords = np.ix_(mask.any(1), mask.any(0))
    return img[coords[0].ravel().min():coords[0].ravel().max() + 1,
               coords[1].ravel().min():coords[1].ravel().max() + 1]


def circle_crop(img: np.ndarray) -> np.ndarray:
    """裁掉黑边并用圆形掩膜把眼底裁成居中的方形。

    入参/出参均为 RGB uint8 (H, W, 3)。
    """
    img = _crop_to_content(img)
    h, w = img.shape[:2]
    side = min(h, w)
    # 居中裁成正方形
    top, left = (h - side) // 2, (w - side) // 2
    img = img[top:top + side, left:left + side]
    # 圆形掩膜
    mask = np.zeros((side, side), dtype=np.uint8)
    cv2.circle(mask, (side // 2, side // 2), side // 2, 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    return img


def ben_graham(img: np.ndarray, sigma: int = 10) -> np.ndarray:
    """Ben Graham 对比度归一化: 4*img - 4*GaussianBlur(img) + 128。"""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_image(path: str, size: int = 512, sigma: int = 10) -> np.ndarray:
    """完整管线: 读图 -> circle_crop -> resize(size) -> ben_graham。

    返回 RGB uint8 (size, size, 3)。
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = circle_crop(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = ben_graham(img, sigma=sigma)
    return img
