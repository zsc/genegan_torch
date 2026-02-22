from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3

# Allow running as `python scripts/eval_fid.py ...` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genegan.data.celeba import build_celeba_attribute_datasets, seed_worker
from genegan.models.genegan import GeneGAN
from genegan.utils.device import resolve_device
from genegan.utils.io import load_checkpoint
from genegan.utils.seed import set_seed


@dataclass(frozen=True)
class FIDConfig:
    attribute: str
    data_root: str
    img_size: int
    n: int
    batch_size: int
    num_workers: int
    device: str | None
    cache_dir: str | None
    seed: int


class _RunningStats:
    def __init__(self, dim: int, *, device: torch.device) -> None:
        self.dim = int(dim)
        self.device = device
        self.n = 0
        self.sum = torch.zeros(self.dim, device=device, dtype=torch.float64)
        self.sum_outer = torch.zeros(self.dim, self.dim, device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, feats: torch.Tensor) -> None:
        if feats.ndim != 2 or feats.shape[1] != self.dim:
            raise ValueError(f"Expected [B,{self.dim}] feats, got {tuple(feats.shape)}")
        feats64 = feats.to(dtype=torch.float64)
        self.sum += feats64.sum(dim=0)
        self.sum_outer += feats64.t().mm(feats64)
        self.n += int(feats.shape[0])

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n < 2:
            raise ValueError("Need at least 2 samples to compute covariance.")
        mu = self.sum / float(self.n)
        outer = torch.outer(mu, mu)
        cov = (self.sum_outer - float(self.n) * outer) / float(self.n - 1)
        cov = (cov + cov.t()) * 0.5  # enforce symmetry
        return mu, cov


def _inception(device: torch.device) -> nn.Module:
    weights = Inception_V3_Weights.DEFAULT
    # torchvision requires aux_logits=True when loading pretrained weights.
    m = inception_v3(weights=weights, aux_logits=True, transform_input=False)
    m.fc = nn.Identity()
    m.eval()
    return m.to(device=device)


def _preprocess_for_inception(x_u8_255: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if x_u8_255.ndim != 4 or x_u8_255.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W], got {tuple(x_u8_255.shape)}")
    x = x_u8_255.to(device=device, dtype=torch.float32) / 255.0
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)[None, :, None, None]
    return (x - mean) / std


@torch.no_grad()
def _features(inception: nn.Module, x_u8_255: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    x = _preprocess_for_inception(x_u8_255, device=device)
    feats = inception(x)
    if feats.ndim != 2:
        feats = feats.flatten(1)
    return feats


def _save_stats(path: Path, mu: torch.Tensor, cov: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Torch->NumPy interop can produce arrays with odd dtype metadata in some
    # environments; force a real NumPy array with a concrete dtype.
    mu_np = np.asarray(mu.detach().cpu().numpy(), dtype=np.float32)
    cov_np = np.asarray(cov.detach().cpu().numpy(), dtype=np.float32)
    np.savez_compressed(
        str(path),
        mu=mu_np,
        cov=cov_np,
    )


def _load_stats(path: Path, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # `allow_pickle=True` for backward compatibility with earlier caches that
    # accidentally stored object arrays.
    arr = np.load(str(path), allow_pickle=True)
    mu_np = np.asarray(arr["mu"], dtype=np.float64)
    cov_np = np.asarray(arr["cov"], dtype=np.float64)
    # NOTE: In some environments, torch's NumPy bridge can be ABI-incompatible
    # with the runtime NumPy package, making `torch.from_numpy` fail even for
    # `numpy.ndarray`. Use a copying conversion for robustness.
    mu = torch.tensor(mu_np, device=device, dtype=torch.float64)
    cov = torch.tensor(cov_np, device=device, dtype=torch.float64)
    cov = (cov + cov.t()) * 0.5
    return mu, cov


def _stats_cache_path(
    *,
    cache_dir: Path,
    attribute: str,
    img_size: int,
    n: int,
    split: str,
) -> Path:
    safe_attr = "".join(c for c in attribute if c.isalnum() or c in ("-", "_"))
    return cache_dir / f"celeba_{safe_attr}_s{img_size}_n{n}_{split}_inceptionv3.npz"


def _frechet_distance(
    mu1: torch.Tensor,
    cov1: torch.Tensor,
    mu2: torch.Tensor,
    cov2: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> float:
    if mu1.shape != mu2.shape:
        raise ValueError("mu shapes do not match")
    if cov1.shape != cov2.shape:
        raise ValueError("cov shapes do not match")

    mu1 = mu1.to(dtype=torch.float64)
    mu2 = mu2.to(dtype=torch.float64)
    cov1 = cov1.to(dtype=torch.float64)
    cov2 = cov2.to(dtype=torch.float64)

    dim = cov1.shape[0]
    eye = torch.eye(dim, device=cov1.device, dtype=torch.float64)
    cov1 = cov1 + eps * eye
    cov2 = cov2 + eps * eye

    diff = mu1 - mu2

    # trace(sqrt(sqrt(cov1) * cov2 * sqrt(cov1))) for PSD covariances.
    e1, v1 = torch.linalg.eigh(cov1)
    e1 = torch.clamp(e1, min=0.0)
    sqrt_cov1 = (v1 * torch.sqrt(e1)).mm(v1.t())
    prod = sqrt_cov1.mm(cov2).mm(sqrt_cov1)
    prod = (prod + prod.t()) * 0.5
    e, _ = torch.linalg.eigh(prod)
    e = torch.clamp(e, min=0.0)
    tr_covmean = torch.sqrt(e).sum()

    fid = diff.dot(diff) + torch.trace(cov1) + torch.trace(cov2) - 2.0 * tr_covmean
    return float(fid.detach().cpu())


def _make_loader(dataset, *, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )


def _iter_n(loader: Iterable[torch.Tensor], n: int):
    seen = 0
    for batch in loader:
        if seen >= n:
            break
        if batch.ndim != 4:
            raise ValueError(f"Unexpected batch shape: {tuple(batch.shape)}")
        take = min(int(batch.shape[0]), n - seen)
        yield batch[:take]
        seen += take


def _load_genegan_from_ckpt(ckpt_path: Path, *, device: torch.device) -> tuple[GeneGAN, dict]:
    raw = torch.load(str(ckpt_path), map_location="cpu")
    cfg = raw.get("config", {}) if isinstance(raw, dict) else {}
    second_ratio = float(cfg.get("second_ratio", 0.25))
    obj_blockconv = bool(cfg.get("obj_blockconv", False))
    obj_block_size = int(cfg.get("obj_block_size", 4))

    model = GeneGAN(
        second_ratio=second_ratio,
        obj_blockconv=obj_blockconv,
        obj_block_size=obj_block_size,
    )
    load_checkpoint(
        ckpt_path,
        splitter=model.splitter,
        joiner=model.joiner,
        d_ax=model.d_ax,
        d_be=model.d_be,
        map_location="cpu",
    )
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("eval-fid")
    ap.add_argument("--ckpt", required=True, type=str, help="GeneGAN checkpoint (.pt)")
    ap.add_argument("--attribute", required=True, type=str)
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--img_size", default=128, type=int, choices=(64, 96, 128))
    ap.add_argument("--n", default=5000, type=int, help="Number of samples per distribution")
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument(
        "--cache_dir",
        default="outputs/_fid_cache",
        type=str,
        help="Where to cache real distribution stats (mu/cov)",
    )
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--out", default=None, type=str, help="Optional JSON output path")
    args = ap.parse_args(argv)

    set_seed(int(args.seed))
    dev = resolve_device(args.device).device
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg = FIDConfig(
        attribute=str(args.attribute),
        data_root=str(args.data_root),
        img_size=int(args.img_size),
        n=int(args.n),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        seed=int(args.seed),
    )

    ckpt_path = Path(args.ckpt)
    model, ckpt_cfg = _load_genegan_from_ckpt(ckpt_path, device=dev)

    inception = _inception(dev)

    splits = build_celeba_attribute_datasets(
        data_root=cfg.data_root,
        attribute=cfg.attribute,
        img_size=cfg.img_size,
    )
    loader_A = _make_loader(splits.with_attr, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed + 1000)
    loader_B = _make_loader(
        splits.without_attr, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed + 2000
    )

    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Real stats (with_attr / without_attr)
    if cache_dir is not None:
        path_real_with = _stats_cache_path(
            cache_dir=cache_dir,
            attribute=cfg.attribute,
            img_size=cfg.img_size,
            n=cfg.n,
            split="with_attr",
        )
        path_real_without = _stats_cache_path(
            cache_dir=cache_dir,
            attribute=cfg.attribute,
            img_size=cfg.img_size,
            n=cfg.n,
            split="without_attr",
        )
    else:
        path_real_with = None
        path_real_without = None

    if path_real_with is not None and path_real_with.exists():
        mu_with, cov_with = _load_stats(path_real_with, device=dev)
    else:
        stats_with = _RunningStats(2048, device=dev)
        for batch in _iter_n(loader_A, cfg.n):
            stats_with.update(_features(inception, batch, device=dev))
        mu_with, cov_with = stats_with.finalize()
        if path_real_with is not None:
            _save_stats(path_real_with, mu_with, cov_with)

    if path_real_without is not None and path_real_without.exists():
        mu_without, cov_without = _load_stats(path_real_without, device=dev)
    else:
        stats_without = _RunningStats(2048, device=dev)
        for batch in _iter_n(loader_B, cfg.n):
            stats_without.update(_features(inception, batch, device=dev))
        mu_without, cov_without = stats_without.finalize()
        if path_real_without is not None:
            _save_stats(path_real_without, mu_without, cov_without)

    # Generated stats (Bu should match with_attr; A0 should match without_attr).
    stats_Bu = _RunningStats(2048, device=dev)
    stats_A0 = _RunningStats(2048, device=dev)

    seen = 0
    for Au_b, B0_b in zip(_iter_n(loader_A, cfg.n), _iter_n(loader_B, cfg.n)):
        with torch.no_grad():
            outs = model(Au_b.to(device=dev), B0_b.to(device=dev))
        stats_Bu.update(_features(inception, outs.Bu, device=dev))
        stats_A0.update(_features(inception, outs.A0, device=dev))
        seen += int(Au_b.shape[0])
        if seen >= cfg.n:
            break

    mu_Bu, cov_Bu = stats_Bu.finalize()
    mu_A0, cov_A0 = stats_A0.finalize()

    fid_Bu = _frechet_distance(mu_with, cov_with, mu_Bu, cov_Bu)
    fid_A0 = _frechet_distance(mu_without, cov_without, mu_A0, cov_A0)

    iter_num = None
    try:
        raw = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(raw, dict) and "iter" in raw:
            iter_num = int(raw["iter"])
    except Exception:
        iter_num = None

    payload = {
        "fid_Bu_vs_with_attr": fid_Bu,
        "fid_A0_vs_without_attr": fid_A0,
        "ckpt": str(ckpt_path),
        "iter": iter_num,
        "fid_config": asdict(cfg),
        "ckpt_config": ckpt_cfg,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
