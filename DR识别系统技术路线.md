# 糖尿病视网膜病变(DR)风险与病变程度识别系统 — 技术路线文档

> 目标任务:**Task 2 + Task 4 联合识别**
> Task 2 — 病变等级:`Retinopathy grade = 0, 1, 2, 3`
> Task 4 — 黄斑水肿危险程度:`Risk of macular edema = 0, 1, 2`
>
> 即同时完成 DR 分级与 DME(糖尿病性黄斑水肿)风险分级的**多任务有监督学习系统**。

---

## 1. 背景与问题定义

糖尿病视网膜病变(Diabetic Retinopathy, DR)是糖尿病的严重并发症,也是成年人视力下降乃至致盲的主要原因。合适的自动识别方法可辅助早期诊断与治疗,有效降低失明率。

本项目要求设计并实现一个**有监督学习系统**,对眼底图像进行分类。在题目给出的难度阶梯中,我们选择难度最高的一档:**同时输出 DR 病变等级(0–3)与黄斑水肿危险程度(0–2)**。这两个任务在临床上高度相关——DME 往往伴随一定程度的 DR——因此适合用**共享特征的多任务网络**联合建模,而不是训练两个孤立模型。

### 1.1 为什么是多任务联合建模

- **临床相关性**:DR 与 DME 共享大量眼底病灶特征(微动脉瘤、出血、硬性渗出),特征可复用。
- **数据效率**:在小数据规模下,共享主干能让两个任务互相正则化,缓解过拟合。
- **方法新颖性**:在两个任务头之间引入跨任务注意力(cross-disease attention),让 DR 与 DME 特征互相引导,是本题最自然、最有论文支撑的创新点(参见 CANet)。

---

## 2. 数据现实:必须先厘清的关键问题

> **这是整个方案能否落地的前提,务必先确认。**

题目给出的标签方案(`Retinopathy grade 0–3` + `Risk of macular edema 0–2`)对应的是 **原始 Messidor 数据集**(1,200 张图像),**而不是 Messidor-2**。

| 数据集 | 规模 | DR 标签 | DME / 黄斑水肿标签 | 备注 |
|---|---|---|---|---|
| **原始 Messidor** | 1,200 张 | grade 0–3(4 类) | Risk 0–2(3 类) | **唯一同时带有题目两个标签的数据集** |
| **Messidor-2** | 1,748 张 / 874 次检查 | 第三方裁定 grade(ICDR 0–4,5 类,1,744 张) | **无 ME 标签** | 无官方标注;DR 标签尺度与原始 Messidor 不同 |
| EyePACS(Kaggle) | ~88,000 张 | ICDR 0–4 | 无 | 大规模预训练用 |
| APTOS 2019 | ~3,600 张 | ICDR 0–4 | 无 | 预训练/增广用 |
| MAPLES-DR | MESSIDOR 子集 | DR + ME 分级 | **有像素级病灶 + ME 分级** | 想增强 ME 监督时的唯一外部来源 |

### 2.1 由此得到的硬约束

1. **Task 4(ME 0–2)只能在原始 Messidor 上监督。** 没有任何大规模外部数据集带 ME 标签。
2. **Messidor-2 / EyePACS / APTOS 的 DR 标签是 5 类(0–4),与原始 Messidor 的 4 类(0–3)尺度不同**,不能直接合并标签空间。它们只能用于**主干预训练**(表示学习),不能直接当作目标任务的标注。
3. **数据泄漏风险**:Messidor-2 与原始 Messidor 在图像来源上有重叠。若同一张图既进了预训练集又进了测试集,报告指标会虚高。**划分数据前必须按图像/患者去重。**

### 2.2 结论

- **目标(target):** 原始 Messidor —— 在其上 fine-tune 并评测 Task 2+4。
- **预训练源(source):** EyePACS + APTOS + Messidor-2 —— 仅用于主干的 DR 表示预训练。
- **可选 ME/病灶增强:** MAPLES-DR —— 若 ME 头精度或热力图病灶对齐成为瓶颈,可作为局部病灶 mask 监督、病灶结构增强与 CAM overlap 验证来源。

---

## 3. 整体技术路线(两阶段)

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1  主干表示预训练(只学特征,丢弃分类头)               │
│  数据: EyePACS + APTOS + Messidor-2 (DR grade 0–4, 5类)        │
│  目标: 得到「眼底/病灶感知」的 ConvNeXt-Tiny 主干              │
│  产物: 预训练 backbone 权重                                    │
└─────────────────────────────────────────────────────────────┘
                          │  加载 backbone,丢弃 5 类头
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2  目标任务 fine-tune(双头 + 跨任务注意力)            │
│  数据: 原始 Messidor (DR 0–3 + ME 0–2)                        │
│  结构: 共享主干 → cross-attention → DR 头(4类) + ME 头(3类)│
│  产物: 最终多任务模型,输出 Task 2 + Task 4                    │
└─────────────────────────────────────────────────────────────┘
```

**注意方向性**:原始 Messidor 是**目标**而非预训练源;Messidor-2 是**预训练源**而非目标。Stage 1 学到的是"看得懂眼底病灶"的通用特征,Stage 2 才真正对齐题目的两个标签。

---

## 4. 网络结构(本文档重点)

### 4.1 主干:ConvNeXt-Tiny

选择 **ConvNeXt-Tiny**(约 28M 参数)作为共享主干,理由:

- **现代纯卷积**:吸收了 Transformer 的训练技巧(大核深度卷积、LayerNorm、GELU、倒置瓶颈),在迁移学习上常优于同级 ResNet / EfficientNet。
- **数据规模匹配**:在 ~1.2k 量级的目标数据上,Tiny 级容量配合 ImageNet/DR 预训练,过拟合风险可控。
- **眼底任务有验证**:ConvNeXt + 注意力在 DR 检测上可达 AUC ≈ 0.94、Cohen's kappa ≈ 0.78 的水平(见参考文献)。

> **对照基线**:同时训练一个 ResNet-50 作为 baseline,用于报告里的消融对比。**ViT/Swin** 在本数据规模下从头训练困难,可作为"进阶尝试"列出但不作主力;**MobileNet/EfficientNet-B0 以下**太轻,难以分辨微动脉瘤等微小病灶,不推荐。

### 4.2 输入与预处理

- **输入分辨率 512×512**(DR 病灶微小,分辨率比网络深度更重要;512 通常显著优于 224)。
- **眼底专用预处理**:圆形裁剪(circle-crop)去黑边 + 对比度归一化(Ben Graham 方案,Kaggle DR 竞赛标准做法),让微动脉瘤、渗出等小病灶更可见。
- **数据增广**:随机旋转/翻转/缩放、亮度对比度抖动、轻度弹性形变;对稀有类别施加更强增广(见第 6 节长尾策略)。

### 4.3 双头 + 跨任务注意力(CANet 思路)

```
              输入眼底图 (512×512×3)
                      │
            ┌─────────▼─────────┐
            │  ConvNeXt-Tiny    │  ← 共享主干 (Stage 1 预训练)
            │   特征图 F        │
            └─────────┬─────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│ DR 专属注意力  │           │ ME 专属注意力  │  ← 疾病特异注意力
│ (disease-     │           │ (disease-     │    各自挑选有用特征
│  specific)    │           │  specific)    │
└───────┬───────┘           └───────┬───────┘
        │                           │
        │   ┌───────────────────┐   │
        └──▶│ 跨任务依赖注意力    │◀──┘  ← disease-dependent
            │ (cross-attention) │       捕捉 DR↔ME 内在关系
            └─────────┬─────────┘
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│  DR 头        │           │  ME 头        │
│  4 类 / 序数   │           │  3 类 / 序数   │
│  → Task 2     │           │  → Task 4     │
└───────────────┘           └───────────────┘
```

两类注意力模块(源自 CANet,TMI 2019,有公开代码):

- **疾病特异注意力(disease-specific attention)**:为 DR 和 ME 各自从共享特征中选择性提取有用特征。
- **疾病依赖注意力(disease-dependent / cross attention)**:显式建模 DR 与 ME 的内在关联,让一个任务的特征引导另一个——这是本系统相对单纯多头网络的核心增值点,且仅需图像级监督即可训练。

### 4.4 分类头:序数回归(可选但推荐)

DR grade(0→3)与 ME risk(0→2)**本质是有序类别**,不是互相独立的离散标签。建议用 **序数回归头(CORAL / CORN,或 cumulative-link)**替代普通 softmax:

- 把"把 3 预测成 2"和"把 3 预测成 0"区别对待,符合临床代价。
- 隐式缓解长尾:高等级类别可借用相邻等级的信号。
- 与评测指标 QWK(见第 7 节)天然对齐。

---

## 5. 损失函数

总损失为两个头的加权和:

```
L_total = w_DR · L_DR  +  w_ME · L_ME   (+ 可选注意力正则项)
```

- **每个头各自做类别均衡加权**(class-balanced / 有效样本数加权)。ME 头数据更少、更偏斜,`w_ME` 与其内部类别权重都应更重。
- 若用序数头,采用对应的序数损失(如 CORN 损失);若用普通分类,采用 **Focal Loss 或 LDAM + DRW**(延迟重加权)以压制多数类、增大稀有类间隔。
- `w_DR`、`w_ME` 可固定,或用不确定性加权(Kendall 多任务自动权重)自适应。

---

## 6. 长尾(类别不平衡)策略

DR/DME 数据天然长尾:grade 0 占大头,R3/增殖期与 M2 极少。按"性价比从高到低"组合使用:

1. **先换评测指标**(最重要,几乎零成本):不要用裸 accuracy,改用 **Quadratic Weighted Kappa(QWK)** + 每类 recall + 混淆矩阵;以 QWK 做早停。
2. **类别均衡损失加权**:逆频率 / 有效样本数加权,单行改动,常是最大单点收益。
3. **损失函数升级**:Focal 或 **LDAM+DRW**。
4. **重采样 + 针对性增广**:对稀有类过采样/类别均衡采样,并对稀有类施加更强增广,避免只是记住那几十张增殖期图。
5. **解耦训练(cRT / LWS)**:先在自然分布上训主干,再**只在均衡数据上重训分类头**(Kang et al.),便宜且对尾部很有效,报告里也好讲。
6. **外部数据是最强的长尾解药**:Stage 1 在 EyePACS+APTOS 上预训练,本身就注入了远多于 Messidor 的稀有等级样本——**对 DR 头**这是治本。
7. **ME 头的局限**:无大规模外部 ME 标签,ME 头几乎无法从 EyePACS/APTOS 获益。若 M2 精度、ME1/ME2 边界或热力图病灶对齐成为瓶颈,可引入 **MAPLES-DR** 补充 ME/病灶监督。
8. **局部 mask 训练与病灶结构增强**:MAPLES-DR 的像素级病灶标注可用于 lesion-aware auxiliary branch、lesion copy-paste、lesion-aware crop,以及 CAM 与病灶 mask 的 overlap 量化。

> 推荐执行顺序:QWK 评测 → 类别均衡/LDAM 损失 → 若 R3/M2 或热力图病灶对齐仍弱 → 解耦重训分类头 + MAPLES-DR 局部 mask 监督 / 病灶结构增强。

---

## 7. 评估方案

- **主指标**:两个任务各自的 **Quadratic Weighted Kappa(QWK)**——DR 分级的事实标准,奖励"接近"的有序错误。
- **辅助指标**:每类 precision/recall/F1、混淆矩阵、宏平均 F1(对长尾敏感)、整体 accuracy(仅作参考,不作为优化目标)。
- **验证协议**:在原始 Messidor 上做**患者级**的分层 K 折交叉验证(避免同一患者双眼分到不同折)。
- **消融实验**(报告核心):
  1. ResNet-50 baseline vs ConvNeXt-Tiny;
  2. 单任务两个独立模型 vs 多任务双头;
  3. 双头无注意力 vs 加 cross-attention;
  4. 有/无 Stage 1 预训练;
  5. 普通 softmax vs 序数头;
  6. 有/无长尾策略。
- **可解释性(加分项)**:Grad-CAM / 注意力热力图,展示模型关注的病灶区域是否与临床一致。当前 A1/A2 与 baseline 的热力图发现记录在 `DISCOVERY.md`。

---

## 8. 实施步骤

1. **数据准备**:下载原始 Messidor、Messidor-2、EyePACS、APTOS;统一预处理(圆裁剪+归一化);**跨数据集去重**;建立患者级划分清单。
2. **Stage 1 预训练**:在 EyePACS+APTOS+Messidor-2(DR 0–4)上训练 ConvNeXt-Tiny,保存主干权重。
3. **Stage 2 fine-tune**:加载主干,接双头 + cross-attention,在原始 Messidor 上训练 Task 2+4,各头类别均衡损失,QWK 早停。
4. **长尾调优**:按第 6 节顺序迭代,直到 R3/M2 recall 达标。
5. **评估与消融**:跑第 7 节全部实验,出混淆矩阵与 Grad-CAM。
6. **(可选)进阶**:引入 MAPLES-DR 做局部病灶 mask 监督、病灶结构增强与 CAM overlap 验证;或加生成式/VLM 分支做异常热力图与文本化报告作为"高级特性"。

---

## 9. 风险与备选

| 风险 | 影响 | 缓解 |
|---|---|---|
| 题目标签实为原始 Messidor 而非 Messidor-2 | 若坚持只用 Messidor-2,Task 4 无法监督 | 以原始 Messidor 为目标,Messidor-2 仅作预训练 |
| 数据泄漏(两数据集重叠) | 指标虚高、结论不可信 | 训练前按图像/患者去重 |
| ME 头数据稀少且长尾严重 | M2 召回低 | 更强增广+均衡损失;必要时引入 MAPLES-DR 局部病灶监督 |
| softmax 热力图误聚焦视盘/亮区 | 0-vs-positive 假阳性升高 | MAPLES-DR 病灶 mask overlap 分析;亮区/视盘鲁棒性增强 |
| MAPLES-DR 与 Messidor 图像重叠 | 辅助监督泄漏到 held-out fold | 按 patient/image 做 fold-aware filtering,只允许当前训练折使用非 held-out 图像 |
| ConvNeXt-Tiny 在 1.2k 上过拟合 | 泛化差 | 强预训练+强增广+早停;必要时退到 EfficientNet-B3 |
| 跨任务注意力调参复杂 | 训练不稳 | 先跑无注意力双头基线,再逐步加注意力 |

---

## 10. 一句话总结

**512px 输入 → ConvNeXt-Tiny 主干(在 EyePACS+APTOS+Messidor-2 上 Stage-1 预训练)→ 在原始 Messidor 上接 CANet 式 cross-attention 双序数头(DR 4 类 / ME 3 类)做 Stage-2 fine-tune,各头类别均衡损失,QWK 早停与评测。** 这条路线兼顾稳健性(成熟主干+迁移学习)、题目契合度(双头直出 Task 2+4)与方法新颖性(跨任务注意力)。

---

## 附录:名词解释

> 仅解释文档中出现、且属于专业术语的缩写;通用共识术语(如 AUC、F1、precision/recall、SOTA、K 折)与已知背景词(DR、ResNet、VAE、VLM)不再赘述。

### 临床 / 数据相关

- **DME(Diabetic Macular Edema,糖尿病性黄斑水肿)**:糖尿病引起的黄斑区液体积聚,与题目 Task 4 的"黄斑水肿危险程度"对应;常伴随一定程度的 DR。
- **ME(Macular Edema,黄斑水肿)**:黄斑水肿的统称,本文中即指 DME 风险分级。
- **ICDR(International Clinical Diabetic Retinopathy scale,国际临床 DR 分级)**:通行的 DR 5 级标准(0 无 / 1 轻度 / 2 中度 / 3 重度 / 4 增殖期)。EyePACS、APTOS、Messidor-2 的 DR 标签用此尺度(0–4),与原始 Messidor 的 0–3 不同。
- **MAPLES-DR**:在 MESSIDOR 子集上提供像素级病灶标注与 DR/ME 分级的外部数据集,可用于补充 ME 监督、局部 lesion mask 训练、病灶结构增强与 CAM overlap 量化。由于它是 MESSIDOR 子集,必须按 patient/image 做 fold-aware filtering 防止数据泄漏。

### 网络结构

- **ConvNeXt**:2022 年提出的现代纯卷积网络,把 Transformer 的训练技巧(大核深度卷积、LayerNorm、GELU、倒置瓶颈)移植回 CNN;本方案主干用其 Tiny 版本(约 28M 参数)。
- **EfficientNet**:通过复合缩放(深度/宽度/分辨率联合)兼顾精度与效率的 CNN 系列(B0–B7);本方案作为退路备选(B3)。
- **ViT(Vision Transformer)**:把图像切成 patch、用 Transformer 处理的视觉模型;数据量小时从头训练困难。
- **Swin(Swin Transformer)**:带滑动窗口、层级结构的 ViT 变体,计算更高效;本方案仅列为进阶尝试。
- **CANet(Cross-disease Attention Network)**:TMI 2019 提出的 DR+DME 联合分级网络,含"疾病特异注意力 + 疾病依赖(跨任务)注意力",仅需图像级监督;本方案双头跨任务注意力的设计来源。
- **TMI(IEEE Transactions on Medical Imaging)**:医学影像领域顶级期刊,CANet 发表于此。

### 评估指标

- **QWK(Quadratic Weighted Kappa,二次加权卡帕)**:衡量预测分级与真实分级一致性的指标,DR 分级的事实标准;错得越远扣分越狠(平方关系),对长尾比裸 accuracy 更靠谱。取值约 −1~1,1 为完美,0 约等于瞎猜。本方案的主指标与早停依据。
- **Cohen's kappa(科恩卡帕)**:一致性统计量,QWK 是其"按距离加权"的版本;文中引用 ConvNeXt 指标时出现。

### 长尾 / 训练方法

- **CORAL(Consistent Rank Logits)**:序数回归头,把 K 类分级拆成 K−1 个共享权重、仅偏置不同的二分类,保证预测单调一致。
- **CORN(Conditional Ordinal Regression for NN)**:CORAL 的改进版,用条件概率链放宽权重共享限制,通常更灵活、精度更好。
- **Focal Loss(焦点损失)**:在交叉熵上加调制因子,压低"容易的多数类"样本权重,聚焦难例与稀有类。
- **LDAM(Label-Distribution-Aware Margin Loss)**:按类别频率给稀有类更大的分类间隔(margin),让其决策边界留更宽安全区。
- **DRW(Deferred Re-Weighting,延迟重加权)**:训练前期正常训(先学好特征),后期才开启类别均衡权重的两段式策略;常与 LDAM 配合。
- **cRT(classifier Re-Training)**:解耦训练的一种——先在自然分布上训整网,再冻结主干、在均衡数据上重训分类头。
- **LWS(Learnable Weight Scaling)**:cRT 的轻量替代,冻结主干后只学习各类别权重向量的缩放因子,不改方向。

### 可解释性

- **Grad-CAM(Gradient-weighted Class Activation Mapping)**:用目标类别的梯度对最后一层卷积特征加权,生成叠加在原图上的热力图,显示模型决策依据的图像区域;用于验证模型是否在看真实病灶,也是答辩/报告的加分项。

---

## 参考文献

- CANet: Cross-disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading (TMI 2019) — [arXiv](https://arxiv.org/abs/1911.01376) · [代码](https://github.com/xmengli/CANet)
- ConvNeXt + attention 在 DR/多眼底病检测上的表现 — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11487407/)
- Messidor-2 数据集说明 — [ADCIS](https://www.adcis.net/en/third-party/messidor2/)
- MAPLES-DR(MESSIDOR 病灶/分级标注)— [Nature Scientific Data](https://www.nature.com/articles/s41597-024-03739-6)
- EyePACS→Messidor 预训练/迁移学习 — [对比学习预训练 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10102012/)

---

## Appendix: Softmax + Ordinal Projection Ablation

### Motivation

The current 5-fold ablations show a clear tradeoff:

- CORN baseline has better overall QWK and stability.
- Softmax+LDAM improves middle/minority recall, especially DR1 and ME1, but reduces ME QWK and majority-class stability.

This suggests softmax is better at direct class competition, while CORN is better at preserving ordinal structure. We therefore add a low-cost post-hoc branch before changing the training objective.

### Post-hoc ordinal projection

Given softmax probabilities:

```text
p0, p1, p2, p3
```

Construct tail probabilities:

```text
P(y > 0) = p1 + p2 + p3
P(y > 1) = p2 + p3
P(y > 2) = p3
```

Then project to an ordinal prediction:

```text
pred_ord = 1[P(y>0)>t0] + 1[P(y>1)>t1] + 1[P(y>2)>t2]
```

Default thresholds:

```text
t0 = t1 = t2 = 0.5
```

Compare three strategies:

```text
raw_softmax:
  argmax(p)

ordinal_projection:
  use pred_ord for every sample

middle_class_guard:
  use pred_ord only when raw_softmax or pred_ord is a middle class;
  otherwise keep raw_softmax
```

### Decision rule

Use held-out pooled metrics as the primary evidence:

- DR1 recall should improve over CORN baseline.
- DR2 recall should recover compared with raw softmax.
- ME1 recall should keep most of the softmax gain.
- QWK / balanced should not drop substantially below raw softmax.

If post-hoc projection works, upgrade to a training-time ordinal consistency loss:

```text
loss = LDAM_CE + lambda_ord * ordinal_consistency_loss
```

where:

```text
tail0 = p1 + p2 + p3
tail1 = p2 + p3
tail2 = p3

level0 = 1[y > 0]
level1 = 1[y > 1]
level2 = 1[y > 2]
```

Initial setting:

```text
lambda_ord = 0.2
```

---

## Appendix: Binary-Init Full-Grade Finetune Plan

### Decision

We choose the conservative "binary init" path instead of a positive-only cascade.

```text
Stage-2A:
  Train binary disease/risk heads:
    DR_bin = 1[DR > 0]
    ME_bin = 1[ME > 0]
  Save backbone + attention weights.
  Discard binary heads.

Stage-2B:
  Load Stage-2A backbone + attention.
  Reinitialize full-grade heads.
  Predict original labels:
    DR = 0/1/2/3
    ME = 0/1/2
```

The final model still predicts the original full-grade tasks directly. The binary stage is only used as representation initialization.

### Why not positive-only cascade as main path

```text
if gate says negative:
    pred = 0
else:
    pred = positive_grade + 1
```

This may produce better-looking numbers, but it has higher report risk:

- A gate false negative cannot be corrected by the grade head.
- The final predictor is no longer a direct full-label classifier.
- It introduces an additional threshold/calibration problem.

Therefore, positive-only cascade is kept as a possible later experiment, not the main path.

### Report experiment matrix

```text
Baseline A:
  CORN + DRW

Baseline B:
  softmax + LDAM + DRW

Experiment A1:
  binary init -> CORN + DRW full-grade finetune
  compare against Baseline A

Experiment A2:
  binary init -> softmax + LDAM + DRW full-grade finetune
  compare against Baseline B
```

This matrix answers:

```text
1. Does binary disease/risk pre-finetuning help generally?
   Compare A1 vs Baseline A and A2 vs Baseline B.

2. Is binary initialization more useful for ordinal heads or class-sensitive heads?
   Compare A1 vs A2.
```

### Additional metrics

Keep the original metrics:

```text
5-fold held-out confusion matrix
per-class recall
QWK
macro recall
balanced = 0.5 * qwk_mean + 0.5 * macro_recall
```

Add derived binary metrics from the final full-grade predictions:

```text
DR positive detection: 0 vs >0 sensitivity/specificity
ME positive detection: 0 vs >0 sensitivity/specificity
```

These directly test whether binary init improves the negative-positive boundary.
