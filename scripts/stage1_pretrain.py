"""Stage-1 入口:在 EyePACS+APTOS+Messidor-2 (DR 0-4) 上预训练主干。

训完只保存 backbone 子状态(checkpoints/stage1/backbone.pth),供 Stage-2 加载。
用法:
    python scripts/stage1_pretrain.py --config configs/stage1_pretrain.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnet.data import DRGradingDataset, build_transforms
from drnet.models import ConvNeXtTinyBackbone
from drnet.utils import load_config, set_seed


class _PretrainNet(nn.Module):
    """主干 + 临时 5 类头(训完丢弃头)。"""

    def __init__(self, num_classes: int, pretrained: bool, grad_ckpt: bool):
        super().__init__()
        self.backbone = ConvNeXtTinyBackbone(pretrained, grad_ckpt)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x):
        f = self.pool(self.backbone(x)).flatten(1)
        return self.fc(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_pretrain.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = cfg["data"]
    tf = build_transforms(d["image_size"], train=True)
    ds = DRGradingDataset(d["csv"], d["root"], transform=tf, image_size=d["image_size"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=d.get("num_workers", 4), pin_memory=True, drop_last=True)

    model = _PretrainNet(cfg["model"]["num_classes"], cfg["model"]["pretrained_imagenet"],
                         cfg["model"].get("grad_checkpoint", False)).to(device)
    if cfg["train"].get("channels_last"):
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device == "cuda")
    accum = cfg["train"].get("grad_accum", 1)
    ce = nn.CrossEntropyLoss()

    for epoch in range(cfg["train"]["epochs"]):
        model.train(); opt.zero_grad(set_to_none=True); running = 0.0
        for step, (img, y) in enumerate(loader):
            img, y = img.to(device), y.to(device)
            with torch.autocast(device_type=device, enabled=scaler.is_enabled()):
                loss = ce(model(img), y) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            running += loss.item() * accum
        print(f"[stage1 epoch {epoch}] loss={running/max(len(loader),1):.4f}")

    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    out = os.path.join(cfg["output"]["dir"], "backbone.pth")
    torch.save({"backbone": {f"backbone.{k}": v
                             for k, v in model.backbone.state_dict().items()}}, out)
    print(f"已保存主干: {out}")


if __name__ == "__main__":
    main()
