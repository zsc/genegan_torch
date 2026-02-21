from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _bn(num_features: int) -> nn.BatchNorm2d:
    # Match tf.layers.batch_normalization defaults more closely:
    # TF: moving = moving*0.99 + batch*0.01, eps=1e-3
    # PyTorch: running = (1-m)*running + m*batch
    return nn.BatchNorm2d(num_features, eps=1e-3, momentum=0.01)


class Splitter(nn.Module):
    def __init__(self, *, second_ratio: float = 0.25) -> None:
        super().__init__()
        if not (0.0 < second_ratio < 1.0):
            raise ValueError("second_ratio must be in (0,1)")
        self.second_ratio = float(second_ratio)
        self.num_ch = int(512 * self.second_ratio)
        if self.num_ch <= 0 or self.num_ch >= 512:
            raise ValueError(f"Invalid num_ch derived from second_ratio: {self.num_ch}")

        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = _bn(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = _bn(512)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(dtype=torch.float32) / 255.0
        h = self.act(self.conv1(x))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        bg, obj = torch.split(h, [512 - self.num_ch, self.num_ch], dim=1)
        return bg, obj


class Joiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn1 = _bn(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn2 = _bn(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 3, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.out_bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, bg: torch.Tensor, obj: torch.Tensor) -> torch.Tensor:
        h = torch.cat([bg, obj], dim=1)
        h = self.relu(self.bn1(self.deconv1(h)))
        h = self.relu(self.bn2(self.deconv2(h)))
        h = self.deconv3(h) + self.out_bias
        h = torch.tanh(h)
        return (h + 1.0) * 255.0 / 2.0


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = _bn(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = _bn(512)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = _bn(512)
        self.fc = nn.Linear(512 * 4 * 4, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32) / 255.0
        h = self.act(self.conv1(x))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = h.flatten(1)
        return self.fc(h)


@dataclass(frozen=True)
class GeneGANOutputs:
    A: torch.Tensor
    x: torch.Tensor
    B: torch.Tensor
    e: torch.Tensor
    A0: torch.Tensor
    Bu: torch.Tensor
    Au_hat: torch.Tensor
    B0_hat: torch.Tensor


class GeneGAN(nn.Module):
    def __init__(self, *, second_ratio: float = 0.25) -> None:
        super().__init__()
        self.splitter = Splitter(second_ratio=second_ratio)
        self.joiner = Joiner()
        self.d_ax = Discriminator()
        self.d_be = Discriminator()

    @property
    def num_ch(self) -> int:
        return self.splitter.num_ch

    def forward(self, Au: torch.Tensor, B0: torch.Tensor) -> GeneGANOutputs:
        A, x = self.splitter(Au)
        B, e = self.splitter(B0)
        zeros = torch.zeros_like(x)

        Au_hat = self.joiner(A, x)
        B0_hat = self.joiner(B, zeros)
        Bu = self.joiner(B, x)
        A0 = self.joiner(A, zeros)

        return GeneGANOutputs(A=A, x=x, B=B, e=e, A0=A0, Bu=Bu, Au_hat=Au_hat, B0_hat=B0_hat)
