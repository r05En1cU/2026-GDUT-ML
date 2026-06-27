# Discovery: Grad-CAM Findings and Next Experiments

本文记录在 Baseline A/B、A1、A2、以及 pure binary-init 模型上统一查看 Grad-CAM 后得到的阶段性发现。它不是最终结论，而是后续 loss、augmentation、解释性分析的实验依据。

## 1. 对比对象与热力图产物

统一热力图目录：

```text
checkpoints/gradcam_compare_resize_off
```

对比模型：

```text
baseline_corn:
  CORN + DRW

baseline_softmax_ldam:
  softmax + LDAM + DRW

binary_init_softmax:
  Stage-2A pure binary softmax init

a1_binary_init_corn:
  binary softmax init -> CORN + DRW full-grade finetune

a2_binary_init_softmax_ldam:
  binary softmax init -> softmax + LDAM + DRW full-grade finetune
```

统一设置：

```text
fold: 0
resize: off
samples: 同一批 8 张 held-out 图像
heads: DR + ME
```

总索引：

```text
checkpoints/gradcam_compare_resize_off/manifest_all.csv
```

## 2. 主要视觉发现

### 2.1 Softmax/LDAM 的特点

Softmax/LDAM 系列，包括 Baseline B 和 A2，在明显病灶区域内识别到病灶时，Grad-CAM 往往能较完整覆盖病灶或分散病灶区域。

观察到的优势：

- 对分散病灶更敏感。
- 对小片、多点、跨区域的异常区域覆盖更主动。
- 这与 A2 中 DR1/DR2 recall 的提升方向一致。

观察到的问题：

- 部分样本中，热力图中心落在视盘或视盘附近亮区。
- 当 CAM 主要集中在视盘附近时，预测准确度更容易下降。
- 这可能解释 softmax 系列 0-vs-positive FPR 较高，以及 ME1 边界不稳定的问题。

### 2.2 CORN 的特点

CORN 系列，包括 Baseline A 和 A1，整体有序稳定性更好，但热力图表现更弥散。

观察到的优势：

- 不容易出现特别尖锐的类别决策热点。
- QWK 和跨级稳定性通常更好。

观察到的问题：

- 热力图范围更大、更弥散。
- 对破损、小出血点、分散病灶的定位精度弱于 softmax。
- 有些样本存在未能精准覆盖可疑病灶区域的情况。

### 2.3 ME 任务也有类似趋势

ME head 的 Grad-CAM 也出现类似现象：

```text
softmax:
  更容易关注到疑似异常区域，但也更容易受视盘/亮区干扰。

CORN:
  更稳定，但病灶定位更弥散，对局部异常的抓取不够锐利。
```

## 3. 与指标结果的对应关系

### 3.1 A1: binary init -> CORN

A1 5-fold mean：

```text
DR QWK:        0.9072 +/- 0.0240
ME QWK:        0.8787 +/- 0.0182
QWK mean:      0.8929 +/- 0.0095
macro recall:  0.7837 +/- 0.0237
balanced:      0.8383 +/- 0.0105
```

A1 pooled held-out：

```text
DR recall = [0.9341, 0.5294, 0.7004, 0.8189]
ME recall = [0.9702, 0.6400, 0.8411]
```

解释：

```text
A1 保留了 CORN 的有序稳定性和高 QWK。
但 Grad-CAM 显示其病灶定位较弥散，可能限制 DR1/ME1 等细粒度边界提升。
```

### 3.2 A2: binary init -> softmax+LDAM

A2 5-fold mean：

```text
DR QWK:        0.9034 +/- 0.0124
ME QWK:        0.8416 +/- 0.0133
QWK mean:      0.8725 +/- 0.0117
macro recall:  0.7920 +/- 0.0157
balanced:      0.8323 +/- 0.0079
```

A2 pooled held-out：

```text
DR recall = [0.9011, 0.6078, 0.6680, 0.8504]
ME recall = [0.9322, 0.6933, 0.8609]
```

相对 Baseline B：

```text
DR1 recall 上升明显。
DR2 recall 也有提升。
ME2 recall 上升。
但 ME1 recall 下降，ME0 specificity 下降，0-vs-positive FPR 更高。
```

解释：

```text
A2 的 softmax/LDAM 头更擅长抓取分散病灶，因此 DR1/DR2 受益。
但 softmax 也更容易受视盘/亮区影响，导致部分阴性或中间风险样本被推向阳性/高风险。
```

## 4. 当前核心假设

### 4.1 Softmax 的主要瓶颈不是看不到病灶

Grad-CAM 显示，softmax/LDAM 在很多样本上能看到病灶，尤其是分散病灶。

更可能的问题是：

```text
softmax 学到的高亮/圆形结构捷径与视盘区域重叠，
导致部分视盘附近 CAM 被误当成病灶证据。
```

### 4.2 CORN 的主要瓶颈是定位不够锐利

CORN 更稳定，但 Grad-CAM 经常更弥散。

更可能的问题是：

```text
CORN 的 threshold loss 优化有序边界，
但没有直接强化最终类别的局部病灶判别边界。
```

### 4.3 Binary-init 带来的是 0-vs-positive 表示初始化

Pure binary-init 的作用更像是让共享特征先学到阴性/阳性边界。

但 Stage-2B 的最终行为仍由 full-grade head 决定：

```text
CORN head:
  更稳但定位弥散。

softmax+LDAM head:
  更敏感但更容易视盘误识别。
```

## 5. 后续可验证指标

单张 Grad-CAM 图只能说明现象，不能充分证明原因。后续报告应尽量把热力图分析升级为定量归因、反事实遮挡和 targeted intervention。

### 5.1 视盘 CAM overlap

建议先做分析指标，不急于直接加 penalty。

定义：

```text
disc_overlap = sum(CAM * disc_mask) / sum(CAM)
```

需要粗视盘 mask，可以先用传统图像方法近似：

```text
1. 使用亮度图或绿色通道。
2. 寻找高亮、近圆形连通区域。
3. 结合位置先验过滤。
```

比较对象：

```text
Baseline B vs A2
A1 vs A2
正确样本 vs 错误样本
0 类正确样本 vs 0 类误报阳性样本
ME1 正确样本 vs ME1 错分样本
```

判定：

```text
如果 softmax 错误样本的 disc_overlap 显著更高，
则视盘误识别假设成立。
```

### 5.2 反事实遮挡验证

比 Grad-CAM 更强的证据是遮挡反事实实验。

对同一张图构造：

```text
original image
disc-occluded image
lesion-occluded image
random-region-occluded image
```

然后比较模型输出变化：

```text
delta_disc =
  P_positive(original) - P_positive(disc_occluded)

delta_lesion =
  P_positive(original) - P_positive(lesion_occluded)

delta_random =
  P_positive(original) - P_positive(random_occluded)
```

如果 softmax 的 0 类误报阳性样本在遮挡视盘后阳性概率显著下降，而随机遮挡没有同等效果，则说明模型确实依赖视盘/亮区证据。

推荐分组：

```text
true negative:
  y = 0, pred = 0

false positive:
  y = 0, pred > 0

true positive:
  y > 0, pred > 0

middle-class error:
  DR1 / ME1 被错分到相邻等级
```

报告表格：

```text
group             delta_disc   delta_lesion  delta_random
true negative     small        small         small
false positive    large        small/medium  small
true positive     small/medium large         small
```

解释：

```text
Grad-CAM 只能说明模型关注区域。
遮挡实验能进一步说明该区域对预测输出是否有因果影响。
```

### 5.3 CAM sharpness / lesion coverage

可选指标：

```text
CAM entropy:
  衡量热力图是否过于弥散。

top-k CAM area:
  衡量模型关注区域是否过大。

lesion overlap:
  若引入 MAPLES-DR 像素级病灶标注，可直接计算 CAM 与病灶 mask 的 overlap。
```

### 5.4 Targeted intervention 验证

在定量归因和遮挡实验之后，再做针对性干预会更有说服力。

实验：

```text
A2-discaug:
  binary softmax init
  -> softmax + LDAM + DRW
  + bright/disc robustness augmentation

A2-maples-aux:
  binary softmax init
  -> softmax + LDAM + DRW
  + MAPLES lesion-aware auxiliary supervision
```

成功标准不只看总分，而看与假设相关的指标：

```text
0-vs-positive FPR 下降
disc_overlap 下降
disc occlusion sensitivity 下降
lesion occlusion sensitivity 保持或上升
DR1/DR2 recall 保持
ME1 recall 恢复或不再下降
QWK / balanced 不明显下降
```

这样可以形成完整闭环：

```text
观察:
  softmax 热力图容易落在视盘/亮区。

量化:
  softmax 错误样本 disc_overlap 更高。

反事实:
  遮挡视盘后错误阳性概率下降。

干预:
  视盘/亮区增强或 MAPLES 病灶监督后 FPR 下降且病灶 recall 保持。
```

## 6. 后续改进方向

### 6.1 Disc/bright-region robustness augmentation

优先级最高，因为不需要额外标注。

实验：

```text
B-discaug:
  softmax + LDAM + DRW + bright/disc robustness augmentation

A2-discaug:
  binary softmax init
  -> softmax + LDAM + DRW + bright/disc robustness augmentation
```

增强方向：

```text
brightness/contrast jitter
local bright patch augmentation
random glare-like circle/ellipse
blur/sharpen variation
局部亮区 dropout 或 attenuation
```

目标：

```text
减少 softmax 对视盘/亮区的捷径依赖，
同时保留其对分散病灶的敏感性。
```

重点观察：

```text
DR/ME 0-vs-positive FPR 是否下降。
ME1 recall 是否恢复。
DR1/DR2 recall 是否保持。
QWK / balanced 是否不下降。
CAM disc_overlap 是否下降。
```

### 6.2 Disc-aware attention penalty

只有在 disc_overlap 分析确认视盘误识别后再做。

形式：

```text
L = L_softmax_ldam + lambda_disc * max(0, disc_overlap - tau)
```

风险：

```text
粗视盘 mask 错误会误伤真实病灶。
部分病灶可能位于视盘附近，强 penalty 可能降低敏感性。
```

### 6.3 CORN + auxiliary class boundary

针对 CORN 热力图弥散和中间类边界弱的问题，可考虑：

```text
L = L_CORN + lambda_aux * L_LDAM_aux
lambda_aux = 0.1 or 0.2
```

解释：

```text
CORN 主任务保持有序稳定性。
辅助 softmax/LDAM 直接提供类别边界监督，强化 DR1/ME1 等中间类。
```

### 6.4 MAPLES-DR 病灶监督

若需要进一步把热力图从“看起来合理”变成“可量化病灶对齐”，MAPLES-DR 是最合适的数据源。

定位：

```text
MAPLES-DR 不作为替代原始 Messidor 主评估集。
它作为局部病灶 mask 监督、病灶结构增强、CAM 对齐验证的数据源。
最终结论仍回到原始 Messidor 5-fold held-out。
```

用途：

```text
1. 训练 lesion-aware auxiliary branch。
2. 计算 CAM 与像素级病灶 mask 的 overlap。
3. 对 ME head 提供额外病灶监督。
4. 构造病灶结构增强，尤其是微动脉瘤、出血、渗出等局部模式。
5. 约束 softmax 避免把视盘/亮区当成病灶证据。
```

可做的训练形式：

```text
MAPLES auxiliary pre-finetune:
  image -> shared backbone
  auxiliary lesion segmentation / lesion presence heads
  save backbone + attention or lesion-aware backbone

Messidor full-grade finetune:
  load lesion-aware initialization
  train original DR 0/1/2/3 + ME 0/1/2 heads
```

或者作为联合辅助 loss：

```text
L = L_grade + lambda_lesion * L_lesion_mask

L_lesion_mask 可以是:
  BCE / Dice / focal Dice

lambda_lesion:
  0.05, 0.1, 0.2
```

病灶结构增强：

```text
lesion copy-paste:
  从 MAPLES mask 提取局部病灶 patch，贴到相近背景区域。

lesion-aware crop:
  训练时提高包含病灶区域 crop 的概率。

lesion-preserving augmentation:
  旋转、亮度、对比度增强时，保证病灶 mask 同步变换。

disc-vs-lesion contrast augmentation:
  用 MAPLES lesion mask 区分真实病灶亮区和视盘亮区，
  降低模型把视盘当成硬性阳性证据的风险。
```

必须注意：

```text
MAPLES-DR 是 MESSIDOR 子集，必须严格避免数据泄漏。
如果某张 MAPLES 图属于当前 held-out fold，则不能在该 fold 的训练或辅助监督中使用。
推荐做 patient/image-level fold-aware filtering。
```

建议实验：

```text
A2-maples-aux:
  binary softmax init
  -> softmax + LDAM + DRW
  + MAPLES lesion auxiliary pre-finetune or auxiliary branch

A2-discaug-maples:
  binary softmax init
  -> softmax + LDAM + DRW
  + MAPLES-guided lesion/disc robustness augmentation

A1-aux:
  binary softmax init
  -> CORN + DRW
  + auxiliary lesion-aware representation
```

主要观察：

```text
1. softmax 视盘误识别是否下降。
2. DR/ME 0-vs-positive FPR 是否下降。
3. DR1/DR2 recall 是否保持。
4. ME1 是否恢复，ME2 是否保持。
5. CAM 与 lesion mask 的 overlap 是否上升。
6. CAM 与 disc/bright area 的 overlap 是否下降。
```

## 7. 当前报告表述建议

可以写成：

```text
Grad-CAM analysis suggests that softmax+LDAM is more sensitive to scattered lesions,
which is consistent with its stronger DR1/DR2 recall.
However, several false or unstable predictions show optic-disc-centered activation,
indicating a possible shortcut through bright disc-like structures.

In contrast, CORN produces more stable ordinal predictions and higher QWK,
but its Grad-CAM maps are often more diffuse and less precisely aligned with
small hemorrhages or local lesion patterns.

These observations motivate a follow-up disc/bright-region robustness experiment:
reduce optic-disc false activation while preserving the lesion sensitivity of softmax.
```

中文报告可写成：

```text
热力图分析显示，softmax+LDAM 对分散病灶更敏感，能够解释其在 DR1/DR2
召回率上的提升；但部分错误样本的激活中心落在视盘或视盘附近亮区，提示模型可能
学习到视盘/亮区捷径。相较之下，CORN 的有序稳定性更强，但热力图更弥散，
对微小出血点和局部破损的定位不如 softmax 精准。

因此后续实验将围绕 softmax 的视盘误识别问题展开，通过亮区/视盘鲁棒性增强
或视盘 CAM overlap 分析，尝试降低假阳性并保留 softmax 对分散病灶的敏感性。
```
