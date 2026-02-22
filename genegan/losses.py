from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GLosses:
    e: torch.Tensor
    cycle_Ax: torch.Tensor
    cycle_Be: torch.Tensor
    Bx: torch.Tensor
    Ae: torch.Tensor
    parallelogram: torch.Tensor
    loss_G_nodecay: torch.Tensor
    loss_G_decay: torch.Tensor
    loss_G: torch.Tensor


@dataclass(frozen=True)
class DLosses:
    Ax_Bx: torch.Tensor
    Be_Ae: torch.Tensor
    loss_D: torch.Tensor


def generator_weight_decay(
    *,
    splitter: nn.Module,
    joiner: nn.Module,
    weight_decay: float,
    img_size: int | None = None,
) -> torch.Tensor:
    if weight_decay <= 0:
        return torch.tensor(0.0, device=next(splitter.parameters()).device)

    weights: list[torch.Tensor] = []

    # For the multi-res model we share most weights. To match single-resolution
    # behavior, we can optionally exclude blocks that are inactive for the given
    # resolution (e.g. extra down/up block for 96/128).
    if img_size in {64, 96, 128}:
        # Splitter
        for name in ("conv1", "conv2", "conv3"):
            m = getattr(splitter, name, None)
            if isinstance(m, nn.Conv2d):
                weights.append(m.weight)
        if img_size in {96, 128}:
            m = getattr(splitter, "conv4", None)
            if isinstance(m, nn.Conv2d):
                weights.append(m.weight)

        # Joiner
        for name in ("deconv1", "deconv2"):
            m = getattr(joiner, name, None)
            if isinstance(m, nn.ConvTranspose2d):
                weights.append(m.weight)
        if img_size in {96, 128}:
            m = getattr(joiner, "deconv3", None)
            if isinstance(m, nn.ConvTranspose2d):
                weights.append(m.weight)
        m = getattr(joiner, "to_rgb", None)
        if isinstance(m, nn.Conv2d):
            weights.append(m.weight)
    else:
        for m in list(splitter.modules()) + list(joiner.modules()):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                weights.append(m.weight)

    if not weights:
        return torch.tensor(0.0, device=next(splitter.parameters()).device)

    total = torch.zeros((), device=weights[0].device)
    for w in weights:
        total = total + 0.5 * weight_decay * (w.square().mean())
    return total


def compute_g_losses(
    *,
    Au: torch.Tensor,
    B0: torch.Tensor,
    A0: torch.Tensor,
    Bu: torch.Tensor,
    Au_hat: torch.Tensor,
    B0_hat: torch.Tensor,
    e: torch.Tensor,
    d_ax: nn.Module,
    d_be: nn.Module,
    splitter: nn.Module,
    joiner: nn.Module,
    weight_decay: float,
) -> GLosses:
    loss_e = e.abs().mean()
    loss_cycle_A = (Au - Au_hat).abs().mean() / 255.0
    loss_cycle_B = (B0 - B0_hat).abs().mean() / 255.0
    loss_adv_Bu = -d_ax(Bu).mean()
    loss_adv_A0 = -d_be(A0).mean()
    loss_para = 0.01 * (Au + B0 - Bu - A0).abs().mean()

    loss_nodecay = (
        loss_e
        + loss_cycle_A
        + loss_cycle_B
        + loss_adv_Bu
        + loss_adv_A0
        + loss_para
    )

    loss_decay = generator_weight_decay(
        splitter=splitter,
        joiner=joiner,
        weight_decay=weight_decay,
        img_size=int(Au.shape[-1]),
    )
    loss_total = loss_nodecay + loss_decay

    return GLosses(
        e=loss_e,
        cycle_Ax=loss_cycle_A,
        cycle_Be=loss_cycle_B,
        Bx=loss_adv_Bu,
        Ae=loss_adv_A0,
        parallelogram=loss_para,
        loss_G_nodecay=loss_nodecay,
        loss_G_decay=loss_decay,
        loss_G=loss_total,
    )


def compute_d_losses(
    *,
    Au: torch.Tensor,
    B0: torch.Tensor,
    A0: torch.Tensor,
    Bu: torch.Tensor,
    d_ax: nn.Module,
    d_be: nn.Module,
) -> DLosses:
    loss_ax = (d_ax(Bu) - d_ax(Au)).mean()
    loss_be = (d_be(A0) - d_be(B0)).mean()
    loss_total = loss_ax + loss_be
    return DLosses(Ax_Bx=loss_ax, Be_Ae=loss_be, loss_D=loss_total)
