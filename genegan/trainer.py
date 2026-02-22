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
from genegan.utils.io import (
    ensure_dir,
    load_checkpoint,
    point_latest_checkpoint,
    save_checkpoint,
    save_concat_row,
)
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

    max_iter: int = 300000
    batch_size: int = 64
    img_size: int = 128
    img_sizes: tuple[int, ...] | None = None
    g_lr: float = 5e-5
    d_lr: float = 5e-5
    second_ratio: float = 0.25
    obj_blockconv: bool = False
    obj_block_size: int = 4
    init_vqvae_ckpt: str | None = None
    init_vqvae_map: str | None = None
    weight_decay: float = 5e-5
    critic_clip: float = 0.01
    critic_every: int = 500
    critic_n: int = 1
    critic_n_500: int = 100

    save_every: int = 500
    sample_every: int = 500
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

    device_spec = resolve_device(cfg.device)
    device = device_spec.device
    use_cuda = device_spec.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    train_sizes = tuple(int(s) for s in (cfg.img_sizes or (cfg.img_size,)))
    if not train_sizes:
        raise ValueError("img_sizes must be non-empty")
    allowed = {64, 96, 128}
    bad = [s for s in train_sizes if s not in allowed]
    if bad:
        raise ValueError(f"Unsupported img_sizes: {bad}. Supported: {sorted(allowed)}")

    # Per-resolution data pipelines.
    it_a: dict[int, Iterator[torch.Tensor]] = {}
    it_b: dict[int, Iterator[torch.Tensor]] = {}
    for s in train_sizes:
        split_s = build_celeba_attribute_datasets(
            data_root=cfg.data_root,
            attribute=cfg.attribute,
            image_dir=cfg.image_dir,
            attr_file=cfg.attr_file,
            img_size=s,
            max_images_per_split=cfg.max_images_per_split,
        )
        dl_a_s = make_dataloader(
            split_s.with_attr,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            seed=cfg.seed + 1000 + s,
            pin_memory=use_cuda,
        )
        dl_b_s = make_dataloader(
            split_s.without_attr,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            seed=cfg.seed + 2000 + s,
            pin_memory=use_cuda,
        )
        it_a[s] = _infinite(dl_a_s)
        it_b[s] = _infinite(dl_b_s)

    sample_size = int(cfg.img_size)
    if sample_size not in it_a:
        sample_size = max(train_sizes)

    model = GeneGAN(
        second_ratio=cfg.second_ratio,
        obj_blockconv=bool(cfg.obj_blockconv),
        obj_block_size=int(cfg.obj_block_size),
    )
    model.apply(init_weights_normal_02)
    if cfg.init_vqvae_ckpt is not None:
        from genegan.utils.vq_init import init_genegan_generator_from_vqvae

        rep = init_genegan_generator_from_vqvae(
            splitter=model.splitter,
            joiner=model.joiner,
            vqvae_ckpt=cfg.init_vqvae_ckpt,
            mapping_json=cfg.init_vqvae_map,
        )
        print(
            f"[vq-init] loaded={len(rep.loaded)} ambiguous={len(rep.ambiguous)} missing={len(rep.missing)}"
        )

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
    last_iter: int | None = None

    def _save_ckpt(iteration: int) -> None:
        ckpt_path = ckpt_dir / f"iter_{iteration:06d}.pt"
        save_checkpoint(
            ckpt_path,
            splitter=model.splitter,
            joiner=model.joiner,
            d_ax=model.d_ax,
            d_be=model.d_be,
            opt_g=opt_g,
            opt_d=opt_d,
            iteration=iteration,
            config=asdict(cfg),
        )
        point_latest_checkpoint(ckpt_dir / "latest.pt", ckpt_path)

    def _save_samples(iteration: int) -> None:
        # sample images: keep BN in train-mode, but avoid updating running stats
        with _bn_momentum(model, 0.0), torch.no_grad():
            try:
                Au_s = next(it_a[sample_size]).to(
                    device=device, dtype=torch.float32, non_blocking=use_cuda
                )
                B0_s = next(it_b[sample_size]).to(
                    device=device, dtype=torch.float32, non_blocking=use_cuda
                )
            except (StopIteration, RuntimeError):
                # When shutting down (e.g. Ctrl+C), DataLoader workers may already be
                # stopped; avoid crashing while trying to write final samples.
                return
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
                    sample_dir / f"iter_{iteration:06d}_{j}.jpg",
                )

    interrupted = False
    try:
        for i in pbar:
            last_iter = i
            d_num = cfg.critic_n_500 if i % cfg.critic_every == 0 else cfg.critic_n

            # update D (with clipping)
            for _ in range(d_num):
                opt_d.zero_grad(set_to_none=True)
                d_acc = None
                for s in train_sizes:
                    Au = next(it_a[s]).to(
                        device=device, dtype=torch.float32, non_blocking=use_cuda
                    )
                    B0 = next(it_b[s]).to(
                        device=device, dtype=torch.float32, non_blocking=use_cuda
                    )
                    # Don't update generator BN running stats during critic steps.
                    with _bn_momentum(model.splitter, 0.0), _bn_momentum(
                        model.joiner, 0.0
                    ), torch.no_grad():
                        outs = model(Au, B0)
                    d_losses_s = compute_d_losses(
                        Au=Au,
                        B0=B0,
                        A0=outs.A0,
                        Bu=outs.Bu,
                        d_ax=model.d_ax,
                        d_be=model.d_be,
                    )
                    (d_losses_s.loss_D / float(len(train_sizes))).backward()
                    d_losses_s_det = type(d_losses_s)(
                        Ax_Bx=d_losses_s.Ax_Bx.detach(),
                        Be_Ae=d_losses_s.Be_Ae.detach(),
                        loss_D=d_losses_s.loss_D.detach(),
                    )
                    if d_acc is None:
                        d_acc = d_losses_s_det
                    else:
                        d_acc = type(d_acc)(
                            Ax_Bx=d_acc.Ax_Bx + d_losses_s_det.Ax_Bx,
                            Be_Ae=d_acc.Be_Ae + d_losses_s_det.Be_Ae,
                            loss_D=d_acc.loss_D + d_losses_s_det.loss_D,
                        )
                opt_d.step()
                clip_params_(model.d_ax, cfg.critic_clip)
                clip_params_(model.d_be, cfg.critic_clip)
                if d_acc is not None:
                    scale = float(len(train_sizes))
                    last_d = type(d_acc)(
                        Ax_Bx=d_acc.Ax_Bx / scale,
                        Be_Ae=d_acc.Be_Ae / scale,
                        loss_D=d_acc.loss_D / scale,
                    )

            # update G
            _set_requires_grad(model.d_ax, False)
            _set_requires_grad(model.d_be, False)
            try:
                opt_g.zero_grad(set_to_none=True)
                g_acc = None
                for s in train_sizes:
                    Au = next(it_a[s]).to(
                        device=device, dtype=torch.float32, non_blocking=use_cuda
                    )
                    B0 = next(it_b[s]).to(
                        device=device, dtype=torch.float32, non_blocking=use_cuda
                    )

                    outs = model(Au, B0)
                    g_losses_s = compute_g_losses(
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
                        weight_decay=0.0,  # apply once below for multi-res
                    )
                    (g_losses_s.loss_G_nodecay / float(len(train_sizes))).backward()
                    g_losses_s_det = type(g_losses_s)(
                        e=g_losses_s.e.detach(),
                        cycle_Ax=g_losses_s.cycle_Ax.detach(),
                        cycle_Be=g_losses_s.cycle_Be.detach(),
                        Bx=g_losses_s.Bx.detach(),
                        Ae=g_losses_s.Ae.detach(),
                        parallelogram=g_losses_s.parallelogram.detach(),
                        loss_G_nodecay=g_losses_s.loss_G_nodecay.detach(),
                        loss_G_decay=g_losses_s.loss_G_decay.detach(),
                        loss_G=g_losses_s.loss_G.detach(),
                    )
                    if g_acc is None:
                        g_acc = g_losses_s_det
                    else:
                        g_acc = type(g_acc)(
                            e=g_acc.e + g_losses_s_det.e,
                            cycle_Ax=g_acc.cycle_Ax + g_losses_s_det.cycle_Ax,
                            cycle_Be=g_acc.cycle_Be + g_losses_s_det.cycle_Be,
                            Bx=g_acc.Bx + g_losses_s_det.Bx,
                            Ae=g_acc.Ae + g_losses_s_det.Ae,
                            parallelogram=g_acc.parallelogram + g_losses_s_det.parallelogram,
                            loss_G_nodecay=g_acc.loss_G_nodecay + g_losses_s_det.loss_G_nodecay,
                            loss_G_decay=g_acc.loss_G_decay + g_losses_s_det.loss_G_decay,
                            loss_G=g_acc.loss_G + g_losses_s_det.loss_G,
                        )

                # Apply weight decay once on the full parameter set.
                if cfg.weight_decay > 0:
                    from genegan.losses import generator_weight_decay

                    decay = generator_weight_decay(
                        splitter=model.splitter,
                        joiner=model.joiner,
                        weight_decay=cfg.weight_decay,
                        img_size=None,
                    )
                    decay.backward()
                else:
                    decay = torch.zeros((), device=device)

                opt_g.step()
                if g_acc is not None:
                    scale = float(len(train_sizes))
                    last_g = type(g_acc)(
                        e=g_acc.e / scale,
                        cycle_Ax=g_acc.cycle_Ax / scale,
                        cycle_Be=g_acc.cycle_Be / scale,
                        Bx=g_acc.Bx / scale,
                        Ae=g_acc.Ae / scale,
                        parallelogram=g_acc.parallelogram / scale,
                        loss_G_nodecay=g_acc.loss_G_nodecay / scale,
                        loss_G_decay=decay.detach(),
                        loss_G=(g_acc.loss_G_nodecay / scale) + decay.detach(),
                    )
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
                _save_ckpt(i)
            if i % cfg.sample_every == 0:
                _save_samples(i)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        writer.close()
        if last_iter is not None and (interrupted or last_iter % cfg.save_every != 0):
            _save_ckpt(last_iter)
        if last_iter is not None and (interrupted or last_iter % cfg.sample_every != 0):
            _save_samples(last_iter)

        if interrupted:  # pragma: no cover
            raise KeyboardInterrupt
