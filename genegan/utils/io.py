from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W] tensor, got {tuple(image.shape)}")
    x = image.detach().to(device="cpu", dtype=torch.float32)
    x = x.clamp(0.0, 255.0).to(torch.uint8)
    return x.permute(1, 2, 0).contiguous().numpy()


def save_image(image: torch.Tensor, path: str | Path) -> None:
    arr = tensor_to_uint8_hwc(image)
    Image.fromarray(arr).save(str(path), quality=95)


def save_concat_row(images: list[torch.Tensor], path: str | Path) -> None:
    if not images:
        raise ValueError("images must be non-empty")
    row = torch.cat(images, dim=2)
    save_image(row, path)


def save_checkpoint(
    path: str | Path,
    *,
    splitter: torch.nn.Module,
    joiner: torch.nn.Module,
    d_ax: torch.nn.Module,
    d_be: torch.nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    iteration: int,
    config: dict[str, Any],
) -> None:
    payload = {
        "G_splitter": splitter.state_dict(),
        "G_joiner": joiner.state_dict(),
        "D_Ax": d_ax.state_dict(),
        "D_Be": d_be.state_dict(),
        "opt_G": opt_g.state_dict(),
        "opt_D": opt_d.state_dict(),
        "iter": int(iteration),
        "config": config,
    }
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    *,
    splitter: torch.nn.Module,
    joiner: torch.nn.Module,
    d_ax: torch.nn.Module,
    d_be: torch.nn.Module,
    opt_g: torch.optim.Optimizer | None = None,
    opt_d: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location=map_location)
    splitter.load_state_dict(ckpt["G_splitter"])
    joiner.load_state_dict(ckpt["G_joiner"])
    d_ax.load_state_dict(ckpt["D_Ax"])
    d_be.load_state_dict(ckpt["D_Be"])
    if opt_g is not None and "opt_G" in ckpt:
        opt_g.load_state_dict(ckpt["opt_G"])
    if opt_d is not None and "opt_D" in ckpt:
        opt_d.load_state_dict(ckpt["opt_D"])
    return ckpt


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

