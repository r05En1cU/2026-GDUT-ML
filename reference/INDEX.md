# Reference 代码索引

本目录收录技术路线文档(`../DR识别系统技术路线.md`)中各方法的官方/权威开源实现,**均已去除 `.git`**,仅保留纯源码与说明,作为本项目实现时的参考依据。

> 说明:`pytorch-grad-cam`、`coral-pytorch`、`classifier-balancing` 已裁剪掉仓库自带的演示图片/数据集文件列表/构建文档等**非代码大文件**(源码、README、代码示例完整保留)。
>
> ⚠️ 目录 `_DELETE_ME_locked_gitstub/` 是首次克隆失败残留的 `.git` 残骸,被宿主机文件锁占用、沙盒无法删除。**请在 Windows 资源管理器中手动删除该文件夹**(不影响任何源码)。

---

## 索引总表

| 仓库 | 对应方法 / 术语 | 对应文档章节 | 原始地址 | commit |
|---|---|---|---|---|
| `CANet/` | 跨任务注意力双头(DR+DME 联合分级) | §4.3 网络结构 | github.com/xmengli/CANet | `5a070e3` |
| `ConvNeXt/` | 主干网络 ConvNeXt-Tiny | §4.1 主干 | github.com/facebookresearch/ConvNeXt | `048efce` |
| `coral-pytorch/` | 序数回归头 CORAL / CORN | §4.4 分类头 | github.com/Raschka-research-group/coral-pytorch | `9129c60` |
| `LDAM-DRW/` | LDAM 损失 + 延迟重加权(DRW) | §5 损失 / §6 长尾 | github.com/kaidic/LDAM-DRW | `2536330` |
| `classifier-balancing/` | 解耦训练 cRT / LWS / τ-norm | §6 长尾(解耦) | github.com/facebookresearch/classifier-balancing | `f162ade` |
| `pytorch-grad-cam/` | Grad-CAM 等可解释性热力图 | §7 评估(可解释性) | github.com/jacobgil/pytorch-grad-cam | `084273b` |

---

## 各仓库详解

### 1. CANet/ —— 跨任务注意力双头(核心参考)
- **方法**:TMI 2019,DR + DME 联合分级网络,含"疾病特异注意力 + 疾病依赖(跨任务)注意力",仅需图像级监督。本项目双头 cross-attention 设计的直接来源。
- **对应文档**:§4.3。
- **关键文件**:
  - `baseline.py` / `test_baseline.py` —— 训练与评测主程序
  - `models/cbam.py`、`models/bam.py` —— 注意力模块(CBAM/BAM)
  - `models/resnet50.py`、`models/model_resnet.py` —— 主干(可替换为 ConvNeXt)
  - `messidor_scripts/train_fold.sh`、`eval_fold.sh` —— **Messidor 上的交叉验证脚本(与本项目数据一致,重点参考)**
  - `datasets/`、`data/` —— Messidor 数据加载与划分
  - `lr_scheduler.py`、`average_result.py`、`utils.py`
- **怎么用**:作为双头+跨任务注意力的结构蓝本;`messidor_scripts/` 的折划分与评测流程可直接借鉴。

### 2. ConvNeXt/ —— 主干网络
- **方法**:Facebook 官方 ConvNeXt 实现(2022),现代纯卷积主干。
- **对应文档**:§4.1。
- **关键文件**:
  - `models/convnext.py` —— **ConvNeXt-Tiny/Small/Base 定义(取 Tiny)**
  - `main.py`、`engine.py` —— 训练/验证循环
  - `optim_factory.py`、`datasets.py`、`utils.py`
  - `TRAINING.md`、`INSTALL.md` —— 训练配置与环境
- **怎么用**:取 `convnext_tiny` 作为共享主干,加载 ImageNet 预训练权重作为 Stage-1 起点;`models/object_detection`、`semantic_segmentation` 本项目用不到,可忽略。

### 3. coral-pytorch/ —— 序数回归头(CORAL / CORN)
- **方法**:把 K 类有序分级转成 K−1 个二分类,保证单调一致。用于 DR(0–3)与 ME(0–2)两个序数头。
- **对应文档**:§4.4。
- **关键文件**:
  - `coral_pytorch/layers.py` —— **CoralLayer(序数输出层)**
  - `coral_pytorch/losses.py` —— **coral_loss / corn_loss**
  - `coral_pytorch/dataset.py` —— 标签↔level 编码、由 logits 还原等级的工具
- **怎么用**:把两个分类头各自换成 CoralLayer + 对应损失;预测时用 `dataset.py` 的工具把累积概率转回 0–3 / 0–2 等级。

### 4. LDAM-DRW/ —— 长尾损失
- **方法**:LDAM(标签分布感知间隔损失)+ DRW(延迟重加权)。给稀有类更大间隔,后期才开启类别均衡权重。
- **对应文档**:§5、§6。
- **关键文件**:
  - `losses.py` —— **LDAMLoss 实现**
  - `cifar_train.py` —— **DRW 训练调度示例(关注 `train_rule` / per-class weight 切换逻辑)**
  - `imbalance_cifar.py` —— 构造长尾分布的示例(类比我们的 DR/ME 长尾)
- **怎么用**:把 `LDAMLoss` 套到 DR/ME 两个头;参照 `cifar_train.py` 的延迟重加权时序,在训练后段开启逆频率权重。

### 5. classifier-balancing/ —— 解耦训练(cRT / LWS)
- **方法**:Decoupling Representation and Classifier(Kang et al. 2020)。先在自然分布训主干,再冻结主干只重训分类头(cRT / LWS / τ-norm)。
- **对应文档**:§6(解耦)。
- **关键文件**:
  - `main.py`、`run_networks.py` —— 两阶段训练主流程
  - `tau_norm.py` —— **τ-normalization(LWS 的近亲,只缩放分类器权重)**
  - `layers/ModulatedAttLayer.py`、`loss/` —— 注意力层与损失
  - `config/` —— 各策略(cRT/LWS/τ-norm)配置示例
- **怎么用**:Stage-2 后若 R3/M2 召回仍弱,参照此仓库做"冻主干、重训分类头"的解耦微调。

### 6. pytorch-grad-cam/ —— 可解释性
- **方法**:Grad-CAM 及一系列 CAM 变体(EigenCAM、AblationCAM 等),生成决策依据热力图。
- **对应文档**:§7(可解释性 / 加分项)。
- **关键文件**:
  - `pytorch_grad_cam/` —— 各类 CAM 实现(`base_cam.py`、`grad_cam` 等)
  - `cam.py` —— 命令行入口示例
  - `usage_examples/` —— **最小可运行调用示例(代码,已保留)**
  - `README.md` —— 用法文档
- **怎么用**:训练完成后对 ConvNeXt 最后一层卷积做 Grad-CAM,叠加到眼底图,验证模型是否关注真实病灶。

---

## 实现建议:这些参考如何拼到一起

1. **主干** ← `ConvNeXt`(取 `convnext_tiny`)
2. **双头 + 跨任务注意力** ← `CANet`(注意力模块 + Messidor 折流程)
3. **两个分类头改序数** ← `coral-pytorch`(CoralLayer + corn_loss)
4. **长尾损失** ← `LDAM-DRW`(LDAMLoss + DRW 时序)
5. **(可选)解耦微调** ← `classifier-balancing`(cRT/LWS)
6. **可解释性出图** ← `pytorch-grad-cam`(Grad-CAM)

> 各仓库 License 以其自带 `LICENSE` 文件为准,二次使用/发表前请遵循原始许可。
