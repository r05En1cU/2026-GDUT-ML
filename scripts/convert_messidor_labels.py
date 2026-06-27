"""把官方原始 Messidor 标注转成 prepare_data.py 所需的统一 csv。

官方原始 Messidor(1200 张)以 12 个 Excel 分发(Annotation_Base11.xls ... Base34.xls),
典型列名:
    Image name | Ophthalmologic department | Retinopathy grade | Risk of macular edema
本脚本自动匹配这些列(大小写/空格/变体不敏感),合并所有文件,输出:
    image, dr_grade(0-3), me_risk(0-2), patient_id

用法:
    python scripts/convert_messidor_labels.py \
        --ann_dir path/to/messidor_annotations \
        --out data/messidor_labels.csv

可选:
    --derive_patient   依据文件名第二字段粗略推断 patient/exam 分组(默认每图独立,见下方说明)
    --drop_duplicates  去掉重复 image 名(官方已知有少量重复图)
    --glob "Annotation_*.xls"   自定义匹配的标注文件名模式

说明:原始 Messidor 未公开双眼/患者配对信息,无法严格还原 patient_id;
默认让每张图自成一组(分折时退化为图像级)。--derive_patient 仅作粗略近似,慎用。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd


def _find_col(cols: list[str], *keywords: str) -> str | None:
    """在列名中找包含全部 keywords(小写子串)的第一列。"""
    for c in cols:
        lc = str(c).strip().lower()
        if all(k in lc for k in keywords):
            return c
    return None


def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if ext == ".tsv" else ",")
    try:
        return pd.read_excel(path)          # .xls 需 xlrd; .xlsx 需 openpyxl
    except ImportError as e:
        sys.exit(f"读取 {path} 需要 Excel 引擎: {e}\n"
                 f"  .xls -> pip install xlrd ;  .xlsx -> pip install openpyxl")


def convert(ann_dir: str, pattern: str, derive_patient: bool,
            drop_duplicates: bool) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(ann_dir, pattern)))
    # 兜底:若默认模式没命中,扫所有 excel/csv
    if not files:
        for p in ("*.xls", "*.xlsx", "*.csv"):
            files += glob.glob(os.path.join(ann_dir, p))
        files = sorted(set(files))
    if not files:
        sys.exit(f"在 {ann_dir} 未找到标注文件(模式 {pattern})")

    frames = []
    for f in files:
        df = _read_any(f)
        cols = list(df.columns)
        c_img = _find_col(cols, "image")
        c_dr = _find_col(cols, "retinopathy") or _find_col(cols, "grade")
        c_me = _find_col(cols, "macular") or _find_col(cols, "edema")
        if not (c_img and c_dr and c_me):
            print(f"[跳过] {os.path.basename(f)}: 缺列 "
                  f"(image={c_img}, dr={c_dr}, me={c_me}) 实际列={cols}")
            continue
        sub = pd.DataFrame({
            "image": df[c_img].astype(str).str.strip(),
            "dr_grade": pd.to_numeric(df[c_dr], errors="coerce").astype("Int64"),
            "me_risk": pd.to_numeric(df[c_me], errors="coerce").astype("Int64"),
        })
        sub["source_file"] = os.path.basename(f)
        frames.append(sub)
        print(f"[读取] {os.path.basename(f)}: {len(sub)} 行")

    if not frames:
        sys.exit("没有任何文件被成功解析,请检查列名。")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["dr_grade", "me_risk"]).reset_index(drop=True)
    out["dr_grade"] = out["dr_grade"].astype(int)
    out["me_risk"] = out["me_risk"].astype(int)

    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates(subset=["image"]).reset_index(drop=True)
        print(f"[去重] 删除重复 image 名 {before - len(out)} 行")

    # patient_id
    if derive_patient:
        # 文件名形如 20051019_38557_0100_PP.tif -> 取前两段作为粗略分组键
        out["patient_id"] = out["image"].map(
            lambda s: "_".join(os.path.basename(str(s)).split("_")[:2]))
        print("[patient] 已按文件名前两段粗略分组(近似,慎用)")
    else:
        out["patient_id"] = out["image"]   # 每图独立组

    # 基本校验
    assert out["dr_grade"].between(0, 3).all(), "dr_grade 超出 0-3,请检查列匹配"
    assert out["me_risk"].between(0, 2).all(), "me_risk 超出 0-2,请检查列匹配"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True, help="存放 Annotation_Base*.xls 的目录")
    ap.add_argument("--out", default="data/messidor_labels.csv")
    ap.add_argument("--glob", dest="pattern", default="Annotation_*.xls")
    ap.add_argument("--derive_patient", action="store_true")
    ap.add_argument("--drop_duplicates", action="store_true")
    args = ap.parse_args()

    df = convert(args.ann_dir, args.pattern, args.derive_patient, args.drop_duplicates)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df[["image", "dr_grade", "me_risk", "patient_id"]].to_csv(args.out, index=False)

    print(f"\n已写出: {args.out}  (共 {len(df)} 张)")
    print("DR  分布:", df["dr_grade"].value_counts().sort_index().to_dict())
    print("ME  分布:", df["me_risk"].value_counts().sort_index().to_dict())
    print(f"分组数(patient_id): {df['patient_id'].nunique()}")


if __name__ == "__main__":
    main()
