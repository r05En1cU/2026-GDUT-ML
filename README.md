# DR + DME 多任务识别系统

基于 Messidor 的糖尿病视网膜病变(DR)分级与黄斑水肿(DME)风险分级**联合识别**系统。
目标:`Task 2`(DR grade 0–3)+ `Task 4`(ME risk 0–2)。

技术路线见 `DR识别系统技术路线.md`,代码骨架与已确认决策见 `代码骨架.md`,参考实现见 `reference/INDEX.md`。

## 方法概要

`512px 输入 → ConvNeXt-Tiny 主干 → CANet 式双头(疾病特异 + 跨任务注意力)→ DR/ME 两个 CORN 序数头`,
LDAM+DRW 处理长尾,QWK 早停与评测。两阶段:大数据集预训练主干 → 原始 Messidor 微调双头。

## 安装

```bash
conda env create -f environment.yml
conda activate dr-dme
# 或纯 pip:先按 CUDA 装 torch,再 pip install -r requirements.txt
```

## 目录

```
drnet/        源码包(data / models / losses / engine / explain / utils)
configs/      stage1_pretrain.yaml, stage2_finetune.yaml
scripts/      prepare_data / stage1_pretrain / stage2_finetune / evaluate / run_gradcam
reference/    各方法官方实现(已去 .git,见 INDEX.md)
```

## 环境自检(免数据)

装好环境后先跑冒烟测试,确认前向 + 双头损失 + 反传 + 指标全通(随机张量,不下载权重):

```bash
python scripts/smoke_test.py            # 跑遍 corn/coral/softmax × none/specific/cross 共 9 组合
```

## 数据准备

自备各数据集图像与标注 csv(脚本不下载数据)。原始 Messidor 标注列:
`image, dr_grade(0-3), me_risk(0-2), patient_id`。

```bash
python scripts/prepare_data.py \
    --messidor_csv path/to/messidor_labels.csv \
    --messidor_dir path/to/messidor_images \
    --out_root data_processed/messidor --image_size 512 --k 5
```

## 训练(baseline:跳过 Stage-1,从 ImageNet 起)

```bash
# stage2_finetune.yaml 中 model.backbone_ckpt: null 即走 ImageNet
python scripts/stage2_finetune.py --config configs/stage2_finetune.yaml

# 5-fold 交叉验证
python scripts/stage2_crossval.py --config configs/stage2_finetune.yaml
```

可选 Stage-1 预训练(需准备合并的 DR 0–4 标注):

```bash
python scripts/stage1_pretrain.py --config configs/stage1_pretrain.yaml
# 再把 stage2 配置的 model.backbone_ckpt 指向 checkpoints/stage1/backbone.pth
```

## 评估与可解释性

```bash
python scripts/evaluate.py  --ckpt checkpoints/stage2/best_qwk.pth   # QWK + 混淆矩阵
python scripts/run_gradcam.py --ckpt checkpoints/stage2/best_qwk.pth --head dr
```

## 10G 显存

默认 `512px / batch 8 / grad_accum 2 / amp / grad_checkpoint`。OOM 阶梯:
降 `batch_size`(8→4)并提 `grad_accum` → 降 `image_size`(512→448→384)。

## 消融(对应技术路线 §7)

改 `configs/stage2_finetune.yaml`:
- `model.attention`: `none | specific | cross`
- `model.head`: `corn | coral | softmax`
- `loss.type`: `ce | focal | ldam`
- `model.backbone_ckpt`: 有/无 Stage-1 主干

> 各 `reference/` 子仓库 License 以其自带 LICENSE 为准。
