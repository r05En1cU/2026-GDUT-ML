"""跨数据集去重 + 患者级分层 K 折。

防泄漏:Messidor-2 与原始 Messidor 在来源上有重叠,预训练集与 Messidor 测试集若含同一图
会使指标虚高。这里用感知哈希 (pHash) 做近重复检测。
"""
from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _phash(path: str, hash_size: int = 16) -> str:
    """简单感知哈希:缩放到 (hash_size+1) 灰度,按相邻像素差分生成位串的 md5。"""
    bgr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if bgr is None:
        return ""
    small = cv2.resize(bgr, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return hashlib.md5(diff.tobytes()).hexdigest()


def dedup_across_datasets(dfs: dict[str, pd.DataFrame], img_dirs: dict[str, str],
                          drop_from: str = "pretrain") -> dict[str, pd.DataFrame]:
    """对多个数据集做跨集去重。

    Args:
        dfs: {名称: DataFrame(含 'image' 列)}。
        img_dirs: {名称: 图像根目录}。
        drop_from: 发现重复时从哪个集合删除(默认从预训练集删,保住目标集 Messidor)。
    Returns:
        去重后的 dfs。
    """
    hashes: dict[str, set[str]] = {}
    for name, df in dfs.items():
        hs = set()
        for img in df["image"]:
            h = _phash(os.path.join(img_dirs[name], str(img)))
            if h:
                hs.add(h)
        hashes[name] = hs

    keep_hashes = set().union(*[h for n, h in hashes.items() if n != drop_from]) \
        if len(dfs) > 1 else set()

    df = dfs[drop_from].copy()
    mask = []
    for img in df["image"]:
        h = _phash(os.path.join(img_dirs[drop_from], str(img)))
        mask.append(h not in keep_hashes)
    dfs = dict(dfs)
    dfs[drop_from] = df[mask].reset_index(drop=True)
    return dfs


def make_patient_level_folds(df: pd.DataFrame, k: int = 5, seed: int = 42,
                             label_col: str = "dr_grade",
                             group_col: str = "patient_id") -> pd.DataFrame:
    """患者级分层 K 折(同一患者所有图在同一折),按 label_col 分层。

    返回带 'fold' 列(0..k-1)的副本。若无 patient_id 列则退化为按图像分层。
    """
    df = df.reset_index(drop=True).copy()
    if group_col not in df.columns:
        df[group_col] = np.arange(len(df))  # 退化:每图自成一组
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    y = df[label_col].astype(int).values
    groups = df[group_col].values
    for fold, (_, val_idx) in enumerate(sgkf.split(df, y, groups)):
        df.loc[val_idx, "fold"] = fold
    return df
