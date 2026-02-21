from __future__ import annotations

import unittest

import torch

from genegan.losses import compute_d_losses, compute_g_losses
from genegan.models.genegan import GeneGAN


class TestForward(unittest.TestCase):
    def test_forward_and_losses(self) -> None:
        torch.manual_seed(0)
        model = GeneGAN(second_ratio=0.25)
        model.train()

        b = 2
        Au = torch.rand(b, 3, 128, 128, dtype=torch.float32) * 255.0
        B0 = torch.rand(b, 3, 128, 128, dtype=torch.float32) * 255.0

        outs = model(Au, B0)
        self.assertTrue(torch.isfinite(outs.Au_hat).all().item())
        self.assertTrue(torch.isfinite(outs.B0_hat).all().item())

        d_losses = compute_d_losses(
            Au=Au,
            B0=B0,
            A0=outs.A0.detach(),
            Bu=outs.Bu.detach(),
            d_ax=model.d_ax,
            d_be=model.d_be,
        )
        self.assertTrue(torch.isfinite(d_losses.loss_D).item())

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
            weight_decay=5e-5,
        )
        self.assertTrue(torch.isfinite(g_losses.loss_G).item())


if __name__ == "__main__":
    unittest.main()
