from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSpec:
    device: torch.device
    type: str


def resolve_device(requested: str | None) -> DeviceSpec:
    req = (requested or "").strip().lower()
    if req in {"", "auto"}:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return DeviceSpec(torch.device("mps"), "mps")
        if torch.cuda.is_available():
            return DeviceSpec(torch.device("cuda"), "cuda")
        return DeviceSpec(torch.device("cpu"), "cpu")

    if req == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return DeviceSpec(torch.device("mps"), "mps")
        raise RuntimeError("Requested device=mps but PyTorch MPS is not available")

    if req == "cpu":
        return DeviceSpec(torch.device("cpu"), "cpu")

    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda but CUDA is not available")
        dev = torch.device(req)
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device={requested} but only {torch.cuda.device_count()} CUDA device(s) are available"
            )
        return DeviceSpec(dev, "cuda")

    raise ValueError(f"Unknown device: {requested}")


def ensure_float32(module: torch.nn.Module) -> torch.nn.Module:
    return module.to(dtype=torch.float32)
