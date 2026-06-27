"""Dataset 封装。

预处理图建议由 scripts/prepare_data.py 预先生成并缓存(circle_crop + ben_graham),
这里默认直接读已处理好的图;若读到原图也可用 on_the_fly=True 现场处理。
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .preprocess import preprocess_image


def _load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class DRGradingDataset(Dataset):
    """Stage-1 预训练用:只含 DR 标签 (0-4)。

    csv 列: image(相对/绝对路径), dr_grade。
    """

    def __init__(self, csv_path: str, img_dir: str, transform=None,
                 on_the_fly: bool = False, image_size: int = 512):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.on_the_fly = on_the_fly
        self.image_size = image_size
        self.labels = self.df["dr_grade"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def class_counts(self, num_classes: int = 5) -> list[int]:
        return np.bincount(self.labels, minlength=num_classes).tolist()

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = os.path.join(self.img_dir, str(row["image"]))
        img = preprocess_image(path, self.image_size) if self.on_the_fly else _load_rgb(path)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, int(row["dr_grade"])


class MessidorMultiTaskDataset(Dataset):
    """Stage-2 微调用:原始 Messidor,含 DR(0-3) 与 ME(0-2) 双标签。

    csv 列: image, dr_grade(0-3), me_risk(0-2), patient_id, fold。
    split: 'train' 取 fold != 当前折; 'val' 取 fold == 当前折。
    """

    def __init__(self, csv_path: str, img_dir: str, fold: int, split: str,
                 transform=None, on_the_fly: bool = False, image_size: int = 512):
        df = pd.read_csv(csv_path)
        if split == "train":
            df = df[df["fold"] != fold]
        elif split == "val":
            df = df[df["fold"] == fold]
        else:
            raise ValueError(f"split 必须是 train/val, 收到 {split}")
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.on_the_fly = on_the_fly
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def class_counts(self, task: str, num_classes: int) -> list[int]:
        col = "dr_grade" if task == "dr" else "me_risk"
        return np.bincount(self.df[col].astype(int), minlength=num_classes).tolist()

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = os.path.join(self.img_dir, str(row["image"]))
        img = preprocess_image(path, self.image_size) if self.on_the_fly else _load_rgb(path)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        target = {"dr": int(row["dr_grade"]), "me": int(row["me_risk"])}
        return img, target
