# After Softmax: 双基线与 binary-init 优化计划

本文记录在 `CORN baseline`、`softmax+LDAM baseline`、以及 softmax/CORN 后处理互相投影之后，下一步采用的研究报告实验路线。

相关发现记录：

```text
DISCOVERY.md
```

其中记录了 A1/A2 与 baseline 的 Grad-CAM 对比发现：softmax+LDAM 对分散病灶更敏感，但存在视盘/亮区误识别风险；CORN 更稳定但热力图更弥散。

## 1. 当前双基线

### 1.1 Baseline A: CORN + DRW

归档目录：

```text
archives/baseline_balanced_drw5_patience25_20260626
```

定位：

```text
backbone: ConvNeXt-Tiny
attention: cross
head: CORN
loss: CORN ordinal loss + DRW level weight
drw_defer_epoch: 5
early stop: balanced = 0.5 * qwk_mean + 0.5 * macro_recall
evaluation: 5-fold held-out
```

特点：

- QWK 稳定，符合有序等级任务。
- 0 类和高等级类整体稳定。
- 不容易出现 softmax 的类别尖峰和严重跨级跳变。
- 问题是 DR1 / ME1 recall 仍有提升空间。

### 1.2 Baseline B: Softmax + LDAM + DRW

归档目录：

```text
archives/baseline_softmax_ldam_drw5_patience25_20260627
```

定位：

```text
backbone: ConvNeXt-Tiny
attention: cross
head: softmax
loss: LDAM + DRW
drw_defer_epoch: 5
early stop: balanced = 0.5 * qwk_mean + 0.5 * macro_recall
evaluation: 5-fold held-out
```

5-fold mean：

```text
DR QWK:        0.9028 +/- 0.0159
ME QWK:        0.8386 +/- 0.0387
QWK mean:      0.8707 +/- 0.0183
macro recall:  0.7851 +/- 0.0405
balanced:      0.8279 +/- 0.0253
```

pooled held-out：

```text
DR QWK = 0.9026
DR recall = [0.9066, 0.5163, 0.6154, 0.8583]

ME QWK = 0.8388
ME recall = [0.9497, 0.7867, 0.7748]
```

特点：

- DR1、ME1 recall 明显更好。
- LDAM 直接优化类别边界，对中间类/少数类更敏感。
- ME QWK 和有序稳定性弱于 CORN。

Baseline B 的意义：

```text
它不是失败实验，而是第二基线。
后续方案应尽量保留 softmax+LDAM 的中间类 recall 优势，
同时恢复 CORN 的有序稳定性。
```

## 2. 后处理实验结论

### 2.1 softmax -> CORN 投影

尝试把 softmax 概率转成 ordinal tail probability：

```text
P(y > 0) = 1 - p0
P(y > 1) = p2 + p3
P(y > 2) = p3
```

固定阈值 `0.5` 会导致 0 类大量被推成 1 类，因为只要：

```text
p0 < 0.5
```

样本就会被判为至少 1 类。

跨折阈值搜索得到：

```text
DR threshold ~= [0.70, 0.50, 0.50]
ME threshold ~= [0.65, 0.50]
```

校准后 0 类崩塌被缓解，但整体仍不如 raw softmax，也不如 CORN baseline。

结论：

```text
softmax 概率不能直接当作有序尾概率使用。
softmax -> CORN 后处理投影不作为主路径。
```

### 2.2 CORN -> class distribution

尝试从 CORN 条件概率链反推出完整类分布：

```text
q0 = P(y > 0)
q1 = P(y > 1 | y > 0)
q2 = P(y > 2 | y > 1)

P(y = 0) = 1 - q0
P(y = 1) = q0 * (1 - q1)
P(y = 2) = q0 * q1 * (1 - q2)
P(y = 3) = q0 * q1 * q2
```

然后用 `argmax P(y=c)` 判类。结果相对原始 CORN threshold 几乎没有收益，ME1 还下降。

结论：

```text
CORN 当前 threshold 判类已经接近自身概率结构下的简单最优解。
问题不在判类函数，而在训练目标和特征初始化。
```

## 3. 当前 loss 的关键限制

当前多任务总损失：

```text
L = w_DR * L_DR + w_ME * L_ME
```

Baseline A：

```text
L_DR = CORN loss for DR
L_ME = CORN loss for ME
```

CORN loss 对每个 threshold 做 BCE：

```text
z_k predicts 1[y > k]
```

限制：

- 它优化的是 `是否超过某个等级阈值`，不是直接优化最终类别 recall。
- 中间类的训练信号被拆散到多个 threshold 中，最终类别边界不一定最优。
- DRW 只是对 level loss 加权，不是对最终输出类别的假阴性/假阳性做直接惩罚。

Baseline B：

```text
L = LDAM softmax loss + DRW
```

限制：

- 它直接优化类别边界，对少数类更敏感。
- 但它没有显式等级约束，容易出现有序不一致。

当前 loss 没有显式预测方差惩罚。`uncertainty: true` 对应 Kendall 多任务不确定性加权，只用于平衡 DR/ME 任务损失，不是惩罚预测分布过尖或 logit 方差。

## 4. 确认采用的主方案：binary init -> full-grade finetune

### 4.1 方案定义

采用“方案 A”：先用简单二分类学习是否患病/是否有风险，得到更好的 backbone + attention 初始化；随后丢弃 binary heads，换成完整分级头做最终 fine-tune。

关键点：

```text
binary 阶段只作为初始化。
最终模型仍直接输出原始完整标签。
不采用 positive-only 级联推理作为主实验。
```

选择该方案的原因：

- 研究报告口径更稳，最终任务定义不变。
- 不引入 gate 错误导致后续分级无法纠正的级联风险。
- 实验变量清晰：只比较是否做过 binary disease/risk pre-finetune。
- 可以同时验证 CORN 和 softmax+LDAM 是否都受益于更好的阴性/阳性边界初始化。

### 4.2 Stage-2A: binary disease/risk pre-finetune

数据：

```text
原始 Messidor，沿用患者级 5-fold 划分。
```

标签：

```text
DR_bin = 1[DR > 0]
ME_bin = 1[ME > 0]
```

模型：

```text
shared backbone
cross attention
DR binary head
ME binary head
```

损失：

```text
L_bin =
  BCEWithLogits(g_DR, DR_bin)
  + BCEWithLogits(g_ME, ME_bin)
```

可选：

```text
weighted BCE
focal BCE
positive-class reweighting
```

产物：

```text
backbone + attention 权重
```

丢弃：

```text
DR binary head
ME binary head
```

代码复用策略：

```text
复用 MessidorMultiTaskDataset 的患者级 fold 划分和图像 transform。
binary 阶段只把 target 动态转成 0 vs >0，不改原始 folds.csv。

复用 MultiTaskNet 的 backbone + cross attention 主体。
binary 阶段配置为:
  head: softmax
  num_classes:
    dr: 2
    me: 2

复用现有 AMP、梯度累积、cosine warmup、AdamW 训练循环。
新增 binary validation 和 binary early-stop，不复用 QWK 作为 Stage-2A 主指标。
```

Stage-2A 必须单独验证阴性/阳性边界：

```text
DR sensitivity = TP / (TP + FN)
DR specificity = TN / (TN + FP)
DR false positive rate = FP / (FP + TN)
DR false negative rate = FN / (FN + TP)

ME sensitivity
ME specificity
ME false positive rate
ME false negative rate
```

Stage-2A early stop 指标：

```text
binary_balanced =
  mean(
    0.5 * (DR sensitivity + DR specificity),
    0.5 * (ME sensitivity + ME specificity)
  )
```

原因：

```text
binary-init 的目标不是让 Stage-2A 本身成为最终模型，
而是确认共享特征先学到了稳定的 0 vs >0 边界。

只看假阳性率不够，因为模型可能通过保守预测降低 FPR，
但同时提高假阴性率，损害阳性检出。
因此报告 FPR/FNR，同时用 sensitivity/specificity 的 balanced 指标做早停。
```

### 4.3 Stage-2B: full-grade severity finetune

加载：

```text
Stage-2A 的 backbone + attention 权重
```

实现细节：

```text
Stage-2B 只加载 Stage-2A checkpoint 中的共享模块:
  backbone.*
  att_dr.*
  att_me.*
  cross.*

不加载:
  head_dr.*
  head_me.*

这样 binary heads 和 full-grade heads 的输出维度不匹配不会影响加载。
```

替换：

```text
DR full-grade head: 0/1/2/3
ME full-grade head: 0/1/2
```

最终任务保持不变：

```text
DR 输出完整 4 类
ME 输出完整 3 类
```

不做主路径：

```text
if gate negative:
    pred = 0
else:
    pred = positive_grade + 1
```

原因：

```text
positive-only 级联可能让指标更好看，
但 gate 一旦错判，分级头无法补救；
报告风险更高。
```

## 5. 研究报告实验矩阵

### 5.1 Baselines

```text
Baseline A:
  CORN + DRW

Baseline B:
  softmax + LDAM + DRW
```

### 5.2 Binary-init experiments

```text
Experiment A1:
  Stage-2A binary init
  -> Stage-2B CORN + DRW full-grade finetune
  Compare against Baseline A

Experiment A2:
  Stage-2A binary init
  -> Stage-2B softmax + LDAM + DRW full-grade finetune
  Compare against Baseline B
```

当前 A1 运行命令：

```text
python scripts/binary_init_crossval.py \
  --config configs/binary_init.yaml \
  --output-root checkpoints/binary_init_cv \
  --log-root runs/binary_init_cv

python scripts/stage2_crossval.py \
  --config configs/a1_binary_init_corn.yaml \
  --shared-cv-root checkpoints/binary_init_cv \
  --output-root checkpoints/a1_binary_init_corn_cv \
  --log-root runs/a1_binary_init_corn_cv
```

这两个实验回答两个问题：

```text
1. binary disease/risk pre-finetune 是否普遍有效？
   看 A1 vs Baseline A，A2 vs Baseline B。

2. binary init 更适合有序头还是类别敏感头？
   看 A1 vs A2。
```

### 5.3 评估指标

主指标保持不变：

```text
5-fold held-out pooled confusion matrix
per-class recall
QWK
macro recall
balanced = 0.5 * qwk_mean + 0.5 * macro_recall
```

额外增加 binary 派生指标：

```text
DR positive detection: 0 vs >0 sensitivity/specificity
ME positive detection: 0 vs >0 sensitivity/specificity
```

原因：

```text
binary init 的核心假设是先学稳阴性/阳性边界。
因此必须报告最终 full-grade 模型在 0 vs >0 上的派生表现。
```

### 5.4 判定标准

相对 Baseline A：

```text
A1 是否提升 DR1 / ME1 recall。
A1 是否保住 CORN 的 QWK 和有序稳定性。
```

相对 Baseline B：

```text
A2 是否保住或提升 DR1 / ME1 recall。
A2 是否改善 ME QWK、DR2 recall 或 0-vs-positive 边界。
```

整体目标：

```text
保留 softmax+LDAM 的类别敏感性，
同时逼近 CORN 的有序稳定性。
```

## 6. 暂不作为主路径的方案

### 6.1 positive-only 级联

形式：

```text
Stage A: 0 vs >0 gate
Stage B: positive-only grade classifier
Inference: gate -> positive grade
```

不作为主路径的原因：

- 数据可能更好看，但 gate 错误会直接截断后续分级。
- 最终模型不再是直接预测完整任务标签，和题目定义有偏移。
- 报告解释复杂度更高。

保留为后续可选实验，不进入当前主实验矩阵。

### 6.2 训练级 ordinal consistency

如果 A2 有收益但仍存在明显跨级错分，再考虑：

```text
L = L_LDAM + lambda_ord * L_ord
lambda_ord = 0.1 or 0.2
```

当前优先级低于 binary-init A1/A2。

### 6.3 confidence/variance penalty

最后再尝试：

- label smoothing
- entropy regularization
- temperature scaling
- logit norm penalty

这些用于缓解 softmax 过度自信，但不替代 binary-init 结构实验。

### 6.4 MAPLES-DR 局部 mask 与病灶结构增强

结合 `DISCOVERY.md` 的 Grad-CAM 发现，后续可以把 MAPLES-DR 放入计划，但不替代当前 A1/A2 主矩阵。

定位：

```text
MAPLES-DR 用于:
  局部病灶 mask 监督
  病灶结构增强
  CAM 与 lesion mask 的 overlap 量化
  softmax 视盘/亮区误识别分析

最终评估仍回到原始 Messidor 5-fold held-out。
```

可选实验：

```text
A2-maples-aux:
  binary softmax init
  -> softmax + LDAM + DRW
  + MAPLES lesion-aware auxiliary pre-finetune / auxiliary branch

A2-discaug-maples:
  binary softmax init
  -> softmax + LDAM + DRW
  + MAPLES-guided lesion/disc robustness augmentation

A1-maples-aux:
  binary softmax init
  -> CORN + DRW
  + lesion-aware representation
```

注意：

```text
MAPLES-DR 是 MESSIDOR 子集，必须按 patient/image 做 fold-aware filtering。
当前 fold 的 held-out 图像不能进入该 fold 的 MAPLES 辅助训练。
```

## 7. 当前判断

当前保留两个正式 baseline：

```text
Baseline A: CORN + DRW，用于代表有序稳定性。
Baseline B: softmax + LDAM + DRW，用于代表类别敏感 recall。
```

下一步主实验：

```text
A1: binary init -> CORN + DRW full-grade finetune
A2: binary init -> softmax + LDAM + DRW full-grade finetune
```

核心假设：

```text
Binary pre-finetuning forces the shared representation to first learn
the clinically important negative-positive boundary.

After replacing the binary heads with full-grade severity heads,
the final fine-tuning may benefit from a better disease-aware initialization
while preserving the original full-grade prediction task.
```
