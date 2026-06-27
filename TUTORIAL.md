# DR + DME 多任务识别教程

这份文档给出从数据准备到训练/评估的最短路径。当前 stage-2 已支持：

- 单折训练：`scripts/stage2_finetune.py`
- 5-fold 交叉验证：`scripts/stage2_crossval.py`

## 1. 环境

```bash
conda env create -f environment.yml
conda activate dr-dme
```

或者手动安装：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 2. 数据准备

先把原始 Messidor 标注整理成统一 CSV，再做预处理和分折：

```bash
python scripts/convert_messidor_labels.py \
  --ann_dir path/to/messidor_annotations \
  --out data/messidor_labels.csv

python scripts/prepare_data.py \
  --messidor_csv data/messidor_labels.csv \
  --messidor_dir path/to/messidor_images \
  --out_root data_processed/messidor \
  --image_size 512 --k 5
```

产物：

- `data_processed/messidor/` 下的预处理图
- `data_processed/messidor/folds.csv`，含 `fold` 列

## 3. 训练

单折训练默认读取 `configs/stage2_finetune.yaml` 里的 `data.fold`：

```bash
python scripts/stage2_finetune.py --config configs/stage2_finetune.yaml
```

如果要显式覆盖折号：

```bash
python scripts/stage2_finetune.py --config configs/stage2_finetune.yaml --fold 3
```

如果缓存图已经是目标尺寸，可关闭重复 `Resize`：

```bash
python scripts/stage2_finetune.py --config configs/stage2_finetune.yaml --resize off
```

### 5-fold 交叉验证

直接跑总控脚本即可，默认会读取 `folds.csv` 中的所有折并汇总指标：

```bash
python scripts/stage2_crossval.py --config configs/stage2_finetune.yaml
```

输出会写到：

- `checkpoints/stage2_cv/fold_*/best_qwk.pth`
- `checkpoints/stage2_cv/fold_metrics.csv`
- `checkpoints/stage2_cv/summary.json`

## 4. 评估与可解释性

单折 checkpoint 可直接评估：

```bash
python scripts/evaluate.py --ckpt checkpoints/stage2/best_qwk.pth
python scripts/run_gradcam.py --ckpt checkpoints/stage2/best_qwk.pth --head dr
```

如果评估某个 fold，只需加 `--fold` 并指向对应 checkpoint：

```bash
python scripts/evaluate.py --fold 2 --ckpt checkpoints/stage2_cv/fold_2/best_qwk.pth
```

## 5. 配置要点

`configs/stage2_finetune.yaml` 里最常改的项：

- `data.fold`: 单折验证时使用的折号
- `model.attention`: `none | specific | cross`
- `model.head`: `corn | coral | softmax`
- `loss.type`: `ce | focal | ldam`
- `model.backbone_ckpt`: `null` 表示从 ImageNet 初始化

## 6. 常用命令

```bash
python scripts/smoke_test.py
python scripts/prepare_data.py --messidor_csv ... --messidor_dir ... --out_root data_processed/messidor --image_size 512 --k 5
python scripts/stage2_finetune.py --config configs/stage2_finetune.yaml
python scripts/stage2_crossval.py --config configs/stage2_finetune.yaml
python scripts/evaluate.py --ckpt checkpoints/stage2/best_qwk.pth
python scripts/run_gradcam.py --ckpt checkpoints/stage2/best_qwk.pth --head dr
```
