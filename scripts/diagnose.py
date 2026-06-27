"""分步定位原生段错误(0xC0000005)。每步 flush 打印,崩溃点 = 最后打印行的下一步。

把本文件放到项目的 scripts/ 下,在项目根目录运行:
    python scripts/diagnose.py
观察它停在第几步,把输出贴回即可定位。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def p(m):
    print(m, flush=True)


p("1) import torch ...")
import torch
p(f"   torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
p(f"   device={device}")

from drnet.utils import load_config
cfg = load_config("configs/stage2_finetune.yaml")

p("2) 数据读取(opencv + albumentations)单样本 ...")
from drnet.data import MessidorMultiTaskDataset, build_transforms
tf = build_transforms(cfg["data"]["image_size"], train=True)
ds = MessidorMultiTaskDataset(cfg["data"]["folds_csv"], cfg["data"]["root"],
                              cfg["data"]["fold"], "train", transform=tf,
                              image_size=cfg["data"]["image_size"])
x, y = ds[0]
p(f"   样本 OK: image={tuple(x.shape)} target={y}")

p("3) 构建模型(下载/加载 ImageNet 权重)...")
from drnet.models import MultiTaskNet
model = MultiTaskNet(cfg["model"])
p("   模型已建(CPU)")

p("4) CPU 前向 ...")
xb = x.unsqueeze(0)
with torch.no_grad():
    out = model(xb)
p(f"   CPU 前向 OK: dr={tuple(out['dr'].shape)} me={tuple(out['me'].shape)}")

if device == "cuda":
    p("5) 迁移到 CUDA 并前向 ...")
    model = model.to("cuda")
    xb = xb.to("cuda")
    with torch.no_grad():
        out = model(xb)
    torch.cuda.synchronize()
    p("   CUDA 前向 OK")

    p("6) CUDA autocast 前向 ...")
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        out = model(xb)
    torch.cuda.synchronize()
    p("   CUDA autocast 前向 OK")
else:
    p("5) (无 CUDA,跳过 GPU 测试)")

p("ALL OK —— 若到这里说明前向链路正常,问题在训练循环/反传")
