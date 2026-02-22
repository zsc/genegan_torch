from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class VQInitReport:
    loaded: list[tuple[str, str]]
    ambiguous: list[tuple[str, list[str]]]
    missing: list[str]


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "params"):
            v = obj.get(k)
            if isinstance(v, dict) and v and all(torch.is_tensor(t) for t in v.values()):
                return {str(kk): tt for kk, tt in v.items()}
        if obj and all(torch.is_tensor(t) for t in obj.values()):
            return {str(kk): tt for kk, tt in obj.items()}
    raise TypeError("Unsupported checkpoint format for VQVAE init (expected a state_dict-like dict).")


def _load_mapping(path: str | Path) -> dict[str, str]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
    ):
        raise ValueError("Mapping JSON must be an object of string->string entries.")
    return raw


def _collect_conv_weight_keys(module: nn.Module) -> list[str]:
    keys: list[str] = []
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            keys.append(f"{name}.weight" if name else "weight")
    return keys


def _init_by_mapping(
    dst: nn.Module,
    *,
    dst_prefix: str,
    src_sd: dict[str, torch.Tensor],
    mapping: dict[str, str],
) -> VQInitReport:
    dst_sd = dst.state_dict()
    loaded: list[tuple[str, str]] = []
    missing: list[str] = []
    for dst_key_short, src_key in mapping.items():
        dst_key = f"{dst_prefix}.{dst_key_short}" if dst_prefix else dst_key_short
        if dst_key not in dst_sd:
            missing.append(dst_key)
            continue
        if src_key not in src_sd:
            missing.append(src_key)
            continue
        src_t = src_sd[src_key]
        if not torch.is_tensor(src_t):
            missing.append(src_key)
            continue
        if tuple(src_t.shape) != tuple(dst_sd[dst_key].shape):
            raise ValueError(
                f"Shape mismatch for {dst_key} <- {src_key}: "
                f"{tuple(dst_sd[dst_key].shape)} vs {tuple(src_t.shape)}"
            )
        dst_sd[dst_key] = src_t.to(dtype=dst_sd[dst_key].dtype)
        loaded.append((dst_key, src_key))

    dst.load_state_dict(dst_sd, strict=False)
    return VQInitReport(loaded=loaded, ambiguous=[], missing=missing)


def _auto_subset(src_sd: dict[str, torch.Tensor], *, kind: str) -> dict[str, torch.Tensor]:
    kind = kind.lower()
    if kind not in {"encoder", "decoder"}:
        raise ValueError("kind must be 'encoder' or 'decoder'")
    keys = list(src_sd.keys())
    if kind == "encoder":
        subs = ("encoder", "enc")
    else:
        subs = ("decoder", "dec")
    sub = {k: src_sd[k] for k in keys if any(s in k.lower() for s in subs)}
    return sub if sub else src_sd


def _init_by_shape(
    dst: nn.Module,
    *,
    dst_prefix: str,
    src_sd: dict[str, torch.Tensor],
    restrict_to: set[str] | None = None,
) -> VQInitReport:
    dst_sd = dst.state_dict()
    src_by_shape: dict[tuple[int, ...], list[str]] = {}
    for k, v in src_sd.items():
        if not torch.is_tensor(v):
            continue
        src_by_shape.setdefault(tuple(v.shape), []).append(k)

    loaded: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, list[str]]] = []
    missing: list[str] = []

    if restrict_to is None:
        restrict_to = set(_collect_conv_weight_keys(dst))

    for dst_key_short in sorted(restrict_to):
        dst_key = f"{dst_prefix}.{dst_key_short}" if dst_prefix else dst_key_short
        if dst_key not in dst_sd:
            continue
        shape = tuple(dst_sd[dst_key].shape)
        cands = src_by_shape.get(shape, [])
        if len(cands) == 1:
            src_key = cands[0]
            dst_sd[dst_key] = src_sd[src_key].to(dtype=dst_sd[dst_key].dtype)
            loaded.append((dst_key, src_key))
        elif len(cands) == 0:
            missing.append(dst_key)
        else:
            ambiguous.append((dst_key, cands))

    dst.load_state_dict(dst_sd, strict=False)
    return VQInitReport(loaded=loaded, ambiguous=ambiguous, missing=missing)


def init_genegan_generator_from_vqvae(
    *,
    splitter: nn.Module,
    joiner: nn.Module,
    vqvae_ckpt: str | Path,
    mapping_json: str | Path | None = None,
) -> VQInitReport:
    """
    Initialize GeneGAN generator weights from a pretrained VQVAE checkpoint.

    We only touch generator conv/deconv weights (splitter/joiner). If `mapping_json`
    is provided, it must map our keys (prefixed with `splitter.` / `joiner.`) to
    VQVAE keys, e.g.:

      {"splitter.conv1.weight": "encoder.conv1.weight", "joiner.deconv1.weight": "decoder.deconv1.weight"}

    Without a mapping, we try an *unambiguous shape match* within encoder/decoder
    key subsets. Ambiguous shapes are skipped and reported.
    """
    raw = torch.load(str(vqvae_ckpt), map_location="cpu")
    src_sd = _extract_state_dict(raw)

    if mapping_json is not None:
        mapping = _load_mapping(mapping_json)
        # Apply explicit mapping to (splitter, joiner) by splitting destination keys.
        split_map = {k.split("splitter.", 1)[1]: v for k, v in mapping.items() if k.startswith("splitter.")}
        join_map = {k.split("joiner.", 1)[1]: v for k, v in mapping.items() if k.startswith("joiner.")}
        other = [k for k in mapping.keys() if not (k.startswith("splitter.") or k.startswith("joiner."))]
        if other:
            raise ValueError(
                "Mapping keys must be prefixed with 'splitter.' or 'joiner.': "
                + ", ".join(sorted(other)[:10])
            )
        rep_s = _init_by_mapping(splitter, dst_prefix="", src_sd=src_sd, mapping=split_map)
        rep_j = _init_by_mapping(joiner, dst_prefix="", src_sd=src_sd, mapping=join_map)
        return VQInitReport(
            loaded=rep_s.loaded + rep_j.loaded,
            ambiguous=[],
            missing=rep_s.missing + rep_j.missing,
        )

    rep_s = _init_by_shape(
        splitter,
        dst_prefix="",
        src_sd=_auto_subset(src_sd, kind="encoder"),
    )
    rep_j = _init_by_shape(
        joiner,
        dst_prefix="",
        src_sd=_auto_subset(src_sd, kind="decoder"),
    )
    return VQInitReport(
        loaded=rep_s.loaded + rep_j.loaded,
        ambiguous=rep_s.ambiguous + rep_j.ambiguous,
        missing=rep_s.missing + rep_j.missing,
    )
