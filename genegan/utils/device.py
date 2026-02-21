from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSpec:
    device: torch.device
    type: str


def resolve_device(requested: str | None) -> DeviceSpec:
    if requested is not None:
        req = requested.lower()
        if req == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return DeviceSpec(torch.device("mps"), "mps")
            raise RuntimeError("Requested device=mps but PyTorch MPS is not available")
        if req == "cuda":
            if torch.cuda.is_available():
                return DeviceSpec(torch.device("cuda"), "cuda")
            raise RuntimeError("Requested device=cuda but CUDA is not available")
        if req == "cpu":
            return DeviceSpec(torch.device("cpu"), "cpu")
        raise ValueError(f"Unknown device: {requested}")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return DeviceSpec(torch.device("mps"), "mps")
    if torch.cuda.is_available():
        return DeviceSpec(torch.device("cuda"), "cuda")
    return DeviceSpec(torch.device("cpu"), "cpu")


def ensure_float32(module: torch.nn.Module) -> torch.nn.Module:
    return module.to(dtype=torch.float32)
