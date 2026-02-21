from __future__ import annotations

import torch
from torch import nn


def init_weights_normal_02(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return


def clip_params_(module: nn.Module, clip: float) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.clamp_(-clip, clip)
