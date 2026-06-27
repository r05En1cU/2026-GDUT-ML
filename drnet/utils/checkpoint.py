"""权重保存/加载。保存时同时存一份 backbone 子状态,便于 Stage-2 只载主干。"""
from __future__ import annotations

import torch


def save_checkpoint(model, path: str, extra: dict | None = None) -> None:
    state = {"model": model.state_dict()}
    if hasattr(model, "backbone"):
        state["backbone"] = {f"backbone.{k}": v
                             for k, v in model.backbone.state_dict().items()}
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(model, path: str, map_location="cpu", strict: bool = True):
    state = torch.load(path, map_location=map_location)
    sd = state.get("model", state)
    return model.load_state_dict(sd, strict=strict)
