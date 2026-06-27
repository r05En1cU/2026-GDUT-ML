"""一次性数据准备:预处理(circle_crop + ben_graham)+ 跨集去重 + 患者级分层 K 折。

输入:各数据集的原始图目录 + 一个标注 csv。
输出:预处理图缓存 + 带 fold 列的 folds.csv。

注意:各数据集的标注 csv 需自行准备(列见下),本脚本不下载数据。
Messidor (目标):  image, dr_grade(0-3), me_risk(0-2), patient_id
预训练集 (源):    image, dr_grade(0-4), source
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnet.data.preprocess import preprocess_image
from drnet.data.splits import make_patient_level_folds


def _cache_preprocessed(df: pd.DataFrame, src_dir: str, dst_dir: str, size: int):
    os.makedirs(dst_dir, exist_ok=True)
    for img in tqdm(df["image"], desc=f"预处理 -> {dst_dir}"):
        out = os.path.join(dst_dir, os.path.basename(str(img)))
        if os.path.exists(out):
            continue
        try:
            arr = preprocess_image(os.path.join(src_dir, str(img)), size)
            cv2.imwrite(out, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        except FileNotFoundError:
            print(f"跳过缺失图: {img}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messidor_csv", required=True, help="原始 Messidor 标注 csv")
    ap.add_argument("--messidor_dir", required=True, help="原始 Messidor 图目录")
    ap.add_argument("--out_root", default="data_processed/messidor")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.messidor_csv)
    _cache_preprocessed(df, args.messidor_dir, args.out_root, args.image_size)
    # 预处理后图名统一为 basename
    df = df.copy()
    df["image"] = df["image"].map(lambda p: os.path.basename(str(p)))
    df = make_patient_level_folds(df, k=args.k, seed=args.seed, label_col="dr_grade")
    folds_csv = os.path.join(args.out_root, "folds.csv")
    df.to_csv(folds_csv, index=False)
    print(f"已写出折清单: {folds_csv}  (n={len(df)})")
    print(df.groupby("fold")[["dr_grade", "me_risk"]].count())


if __name__ == "__main__":
    main()
