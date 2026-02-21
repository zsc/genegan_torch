from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from tqdm import tqdm

from genegan.data.celeba import build_celeba_attribute_datasets, make_dataloader
from genegan.losses import GLosses, compute_d_losses, compute_g_losses
from genegan.models.genegan import GeneGAN
from genegan.models.init import clip_params_, init_weights_normal_02
from genegan.utils.device import resolve_device
from genegan.utils.io import ensure_dir, load_checkpoint, save_checkpoint, save_concat_row
from genegan.utils.seed import set_seed


def _infinite(loader) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            yield batch


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(enabled)


def _optimizer_to_device_(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device=device)


@contextmanager
def _bn_momentum(module: nn.Module, momentum: float):
    bns: list[tuple[nn.modules.batchnorm._BatchNorm, float | None]] = []
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            bns.append((m, m.momentum))
            m.momentum = momentum
    try:
        yield
    finally:
        for m, old in bns:
            m.momentum = old


@dataclass(frozen=True)
class TrainConfig:
    attribute: str
    data_root: str
    image_dir: str | None
    attr_file: str | None
    exp_dir: str
    device: str | None
    seed: int
    num_workers: int

    max_images_per_split: int | None = None
    resume_ckpt: str | None = None

    max_iter: int = 100000
    batch_size: int = 64
    img_size: int = 64
    g_lr: float = 5e-5
    d_lr: float = 5e-5
    second_ratio: float = 0.25
    weight_decay: float = 5e-5
    critic_clip: float = 0.01
    critic_every: int = 500
    critic_n: int = 1
    critic_n_500: int = 100

    save_every: int = 500
    log_every: int = 20


def train(cfg: TrainConfig) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "tensorboard is required for training logs. Install it with `pip install tensorboard`."
        ) from e

    set_seed(cfg.seed)
    exp_dir = Path(cfg.exp_dir)
    ckpt_dir = ensure_dir(exp_dir / "checkpoints")
    sample_dir = ensure_dir(exp_dir / "samples")
    log_dir = ensure_dir(exp_dir / "logs")

    split = build_celeba_attribute_datasets(
        data_root=cfg.data_root,
        attribute=cfg.attribute,
        image_dir=cfg.image_dir,
        attr_file=cfg.attr_file,
        img_size=cfg.img_size,
        max_images_per_split=cfg.max_images_per_split,
    )

    dl_a = make_dataloader(
        split.with_attr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        seed=cfg.seed + 1,
    )
    dl_b = make_dataloader(
        split.without_attr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        seed=cfg.seed + 2,
    )

    it_a = _infinite(dl_a)
    it_b = _infinite(dl_b)

    model = GeneGAN(second_ratio=cfg.second_ratio)
    model.apply(init_weights_normal_02)

    device = resolve_device(cfg.device).device
    model.to(device=device, dtype=torch.float32)
    model.train()

    opt_g = torch.optim.RMSprop(
        list(model.splitter.parameters()) + list(model.joiner.parameters()),
        lr=cfg.g_lr,
        alpha=0.8,
        momentum=0.0,
    )
    opt_d = torch.optim.RMSprop(
        list(model.d_ax.parameters()) + list(model.d_be.parameters()),
        lr=cfg.d_lr,
        alpha=0.8,
        momentum=0.0,
    )

    writer = SummaryWriter(log_dir=str(log_dir))

    start_iter = 0
    if cfg.resume_ckpt is not None:
        ckpt_path = Path(cfg.resume_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = load_checkpoint(
            ckpt_path,
            splitter=model.splitter,
            joiner=model.joiner,
            d_ax=model.d_ax,
            d_be=model.d_be,
            opt_g=opt_g,
            opt_d=opt_d,
            map_location="cpu",
        )
        _optimizer_to_device_(opt_g, device)
        _optimizer_to_device_(opt_d, device)
        start_iter = int(ckpt.get("iter", 0)) + 1

    last_g: GLosses | None = None
    last_d = None

    pbar = tqdm(range(start_iter, cfg.max_iter), desc="train", dynamic_ncols=True)
    for i in pbar:
        d_num = cfg.critic_n_500 if i % cfg.critic_every == 0 else cfg.critic_n

        # update D (with clipping)
        for _ in range(d_num):
            Au = next(it_a).to(device=device, dtype=torch.float32)
            B0 = next(it_b).to(device=device, dtype=torch.float32)

            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                outs = model(Au, B0)
            d_losses = compute_d_losses(
                Au=Au,
                B0=B0,
                A0=outs.A0,
                Bu=outs.Bu,
                d_ax=model.d_ax,
                d_be=model.d_be,
            )
            d_losses.loss_D.backward()
            opt_d.step()
            clip_params_(model.d_ax, cfg.critic_clip)
            clip_params_(model.d_be, cfg.critic_clip)
            last_d = d_losses

        # update G
        Au = next(it_a).to(device=device, dtype=torch.float32)
        B0 = next(it_b).to(device=device, dtype=torch.float32)

        _set_requires_grad(model.d_ax, False)
        _set_requires_grad(model.d_be, False)
        try:
            opt_g.zero_grad(set_to_none=True)
            outs = model(Au, B0)
            g_losses = compute_g_losses(
                Au=Au,
                B0=B0,
                A0=outs.A0,
                Bu=outs.Bu,
                Au_hat=outs.Au_hat,
                B0_hat=outs.B0_hat,
                e=outs.e,
                d_ax=model.d_ax,
                d_be=model.d_be,
                splitter=model.splitter,
                joiner=model.joiner,
                weight_decay=cfg.weight_decay,
            )
            g_losses.loss_G.backward()
            opt_g.step()
            last_g = g_losses
        finally:
            _set_requires_grad(model.d_ax, True)
            _set_requires_grad(model.d_be, True)

        if last_g is not None and last_d is not None:
            pbar.set_postfix(
                g=float(last_g.loss_G.detach().cpu()),
                d=float(last_d.loss_D.detach().cpu()),
            )

        if i % cfg.log_every == 0 and last_g is not None and last_d is not None:
            # G losses
            writer.add_scalar("e", float(last_g.e.detach().cpu()), i)
            writer.add_scalar("cycle_Ax", float(last_g.cycle_Ax.detach().cpu()), i)
            writer.add_scalar("cycle_Be", float(last_g.cycle_Be.detach().cpu()), i)
            writer.add_scalar("Bx", float(last_g.Bx.detach().cpu()), i)
            writer.add_scalar("Ae", float(last_g.Ae.detach().cpu()), i)
            writer.add_scalar(
                "parallelogram", float(last_g.parallelogram.detach().cpu()), i
            )
            writer.add_scalar(
                "loss_G_nodecay", float(last_g.loss_G_nodecay.detach().cpu()), i
            )
            writer.add_scalar(
                "loss_G_decay", float(last_g.loss_G_decay.detach().cpu()), i
            )
            writer.add_scalar("loss_G", float(last_g.loss_G.detach().cpu()), i)

            # D losses
            writer.add_scalar("Ax_Bx", float(last_d.Ax_Bx.detach().cpu()), i)
            writer.add_scalar("Be_Ae", float(last_d.Be_Ae.detach().cpu()), i)
            writer.add_scalar("loss_D", float(last_d.loss_D.detach().cpu()), i)

            # learning rate
            writer.add_scalar("g_learning_rate", cfg.g_lr, i)
            writer.add_scalar("d_learning_rate", cfg.d_lr, i)

        if i % cfg.save_every == 0:
            ckpt_path = ckpt_dir / f"iter_{i:06d}.pt"
            save_checkpoint(
                ckpt_path,
                splitter=model.splitter,
                joiner=model.joiner,
                d_ax=model.d_ax,
                d_be=model.d_be,
                opt_g=opt_g,
                opt_d=opt_d,
                iteration=i,
                config=asdict(cfg),
            )
            save_checkpoint(
                ckpt_dir / "latest.pt",
                splitter=model.splitter,
                joiner=model.joiner,
                d_ax=model.d_ax,
                d_be=model.d_be,
                opt_g=opt_g,
                opt_d=opt_d,
                iteration=i,
                config=asdict(cfg),
            )

            # sample images: keep BN in train-mode, but avoid updating running stats
            with _bn_momentum(model, 0.0), torch.no_grad():
                Au_s = next(it_a).to(device=device, dtype=torch.float32)
                B0_s = next(it_b).to(device=device, dtype=torch.float32)
                outs_s = model(Au_s, B0_s)

                n = min(5, Au_s.shape[0])
                for j in range(n):
                    save_concat_row(
                        [
                            Au_s[j],
                            B0_s[j],
                            outs_s.A0[j],
                            outs_s.Bu[j],
                            outs_s.Au_hat[j],
                            outs_s.B0_hat[j],
                        ],
                        sample_dir / f"iter_{i:06d}_{j}.jpg",
                    )

    writer.close()
