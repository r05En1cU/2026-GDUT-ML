"""albumentations 增广管线。预处理图已是 512 方形且对比度归一化,这里只做几何/光度扰动。"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet 统计量(timm convnext 默认)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool, strong: bool = False,
                     resize: bool = True):
    """构建 albumentations 变换。

    Args:
        image_size: 目标边长。
        train: 训练管线(含增广)还是验证管线(仅归一化)。
        strong: 对稀有类施加更强增广(见 datasets 的过采样逻辑)。
    """
    if not train:
        aug = []
        if resize:
            aug.append(A.Resize(image_size, image_size))
        aug += [
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
        return A.Compose(aug)

    aug = []
    if resize:
        aug.append(A.Resize(image_size, image_size))
    aug += [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=0, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, border_mode=0, p=0.5),
    ]
    if strong:
        aug += [
            A.ElasticTransform(alpha=1, sigma=20, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=image_size // 16,
                            max_width=image_size // 16, p=0.3),
        ]
    aug += [A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()]
    return A.Compose(aug)
