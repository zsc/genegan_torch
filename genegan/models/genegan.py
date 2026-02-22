from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class SpatialBlockConv2d(nn.Module):
    """
    Spatial block-wise convolution.

    Split the feature map into non-overlapping blocks and apply a different Conv2d
    per block (shared within each block). This sits between fully shared conv and
    fully local conv.
    """

    def __init__(
        self,
        channels: int,
        *,
        block_size: int = 4,
        kernel_size: int = 3,
        supported_hw: tuple[int, ...] = (12, 16),
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.channels = int(channels)
        self.block_size = int(block_size)
        self.kernel_size = int(kernel_size)
        self.supported_hw = tuple(int(x) for x in supported_hw)

        pad = kernel_size // 2
        self.by_hw = nn.ModuleDict()
        for hw in self.supported_hw:
            if hw % self.block_size != 0:
                raise ValueError(
                    f"supported hw={hw} is not divisible by block_size={self.block_size}"
                )
            grid = hw // self.block_size
            convs = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=pad,
                        bias=False,
                    )
                    for _ in range(grid * grid)
                ]
            )
            self.by_hw[str(hw)] = convs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        b, c, h, w = x.shape
        if h != w:
            raise ValueError(f"Expected square feature map, got H={h}, W={w}")
        if c != self.channels:
            raise ValueError(f"Expected C={self.channels}, got C={c}")
        key = str(int(h))
        if key not in self.by_hw:
            raise ValueError(
                f"Unsupported feature map size {h} for SpatialBlockConv2d. Supported: {self.supported_hw}"
            )

        convs: nn.ModuleList = self.by_hw[key]
        bs = self.block_size
        grid = h // bs
        out = torch.empty_like(x)
        idx = 0
        for i in range(grid):
            for j in range(grid):
                patch = x[:, :, i * bs : (i + 1) * bs, j * bs : (j + 1) * bs]
                out[:, :, i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = convs[idx](patch)
                idx += 1
        return out


class SwitchableBN2d(nn.Module):
    """
    BatchNorm with per-resolution running stats (and affine params).

    For multi-resolution training, sharing BN running stats across resolutions can
    introduce interference. We keep conv weights shared, but maintain separate BN
    modules for each supported resolution.
    """

    def __init__(
        self,
        num_features: int,
        *,
        sizes: tuple[int, ...] = (64, 96, 128),
        eps: float = 1e-3,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.sizes = tuple(int(s) for s in sizes)
        self.bns = nn.ModuleDict(
            {
                str(s): nn.BatchNorm2d(
                    num_features,
                    eps=eps,
                    momentum=momentum,
                )
                for s in self.sizes
            }
        )

    def forward(self, x: torch.Tensor, *, size: int) -> torch.Tensor:
        key = str(int(size))
        if key not in self.bns:
            raise ValueError(f"Unsupported resolution for BN: {size}. Supported: {self.sizes}")
        return self.bns[key](x)


def _bn(num_features: int) -> SwitchableBN2d:
    # Match tf.layers.batch_normalization defaults more closely:
    # TF: moving = moving*0.99 + batch*0.01, eps=1e-3
    # PyTorch: running = (1-m)*running + m*batch
    return SwitchableBN2d(num_features, eps=1e-3, momentum=0.01)


class Splitter(nn.Module):
    """
    Encoder that supports 64x64 / 96x96 / 128x128 inputs.

    Shared conv trunk with a small resolution-dependent branch:
    - 64  -> 16x16 bottleneck
    - 96  -> 12x12 bottleneck
    - 128 -> 16x16 bottleneck
    """

    def __init__(
        self,
        *,
        second_ratio: float = 0.25,
        obj_blockconv: bool = False,
        obj_block_size: int = 4,
    ) -> None:
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
        # Keep spatial size after conv2; only increase channels.
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = _bn(512)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = _bn(512)

        self.obj_blockconv_enabled = bool(obj_blockconv)
        if self.obj_blockconv_enabled:
            self.obj_blockconv = SpatialBlockConv2d(
                self.num_ch,
                block_size=int(obj_block_size),
                kernel_size=3,
                supported_hw=(12, 16),
            )
            self.obj_block_bn = _bn(self.num_ch)
        else:
            self.obj_blockconv = None
            self.obj_block_bn = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W] input, got {tuple(x.shape)}")
        h_in, w_in = int(x.shape[-2]), int(x.shape[-1])
        if h_in != w_in:
            raise ValueError(f"Expected square input, got H={h_in}, W={w_in}")
        if h_in not in {64, 96, 128}:
            raise ValueError(f"Expected H=W to be 64, 96 or 128, got {h_in}")

        x = x.to(dtype=torch.float32) / 255.0
        h = self.act(self.conv1(x))
        h = self.act(self.bn2(self.conv2(h), size=h_in))
        h = self.act(self.bn3(self.conv3(h), size=h_in))

        # After conv3:
        # - 64  -> 16x16
        # - 96  -> 24x24 -> apply conv4 -> 12x12
        # - 128 -> 32x32 -> apply conv4 -> 16x16
        if h.shape[-1] in {24, 32}:
            h = self.act(self.bn4(self.conv4(h), size=h_in))

        if h.shape[-1] not in {12, 16}:
            raise RuntimeError(f"Unexpected feature map size after Splitter: {tuple(h.shape)}")

        bg, obj = torch.split(h, [512 - self.num_ch, self.num_ch], dim=1)

        if self.obj_blockconv is not None and self.obj_block_bn is not None:
            # Residual block-wise conv on the object subspace at bottleneck.
            obj_delta = self.obj_blockconv(obj)
            obj_delta = self.act(self.obj_block_bn(obj_delta, size=h_in))
            obj = obj + obj_delta

        return bg, obj


class Joiner(nn.Module):
    """
    Decoder that supports 64x64 / 96x96 / 128x128 outputs.

    Maximize sharing by using a shared upsampling trunk and a shared toRGB head.

    Input bottleneck spatial size must match the encoder:
    - out_size=64  expects 16x16 input features (2 upsample blocks)
    - out_size=96  expects 12x12 input features (3 upsample blocks)
    - out_size=128 expects 16x16 input features (3 upsample blocks)
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn1 = _bn(256)
        self.deconv2 = nn.ConvTranspose2d(
            256, 256, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn2 = _bn(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 256, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn3 = _bn(256)
        self.to_rgb = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.out_bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, bg: torch.Tensor, obj: torch.Tensor, *, out_size: int) -> torch.Tensor:
        if out_size not in {64, 96, 128}:
            raise ValueError("out_size must be 64, 96 or 128")

        h = torch.cat([bg, obj], dim=1)
        h = self.relu(self.bn1(self.deconv1(h), size=out_size))
        h = self.relu(self.bn2(self.deconv2(h), size=out_size))
        if out_size in {96, 128}:
            h = self.relu(self.bn3(self.deconv3(h), size=out_size))

        h = self.to_rgb(h) + self.out_bias
        h = torch.tanh(h)
        out = (h + 1.0) * 255.0 / 2.0
        if int(out.shape[-1]) != out_size or int(out.shape[-2]) != out_size:
            raise RuntimeError(
                f"Joiner produced {tuple(out.shape)} but out_size={out_size}. "
                "This usually means the latent spatial size does not match the requested resolution."
            )
        return out


class Discriminator(nn.Module):
    """
    Critic that supports 64x64 / 96x96 / 128x128 inputs with maximum sharing.

    All conv blocks are shared. Feature maps are pooled to 4x4 before a shared
    linear head. (WGAN-style critic: no sigmoid.)
    """

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
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W] input, got {tuple(x.shape)}")
        h_in, w_in = int(x.shape[-2]), int(x.shape[-1])
        if h_in != w_in:
            raise ValueError(f"Expected square input, got H={h_in}, W={w_in}")
        if h_in not in {64, 96, 128}:
            raise ValueError(f"Expected H=W to be 64, 96 or 128, got {h_in}")

        x = x.to(dtype=torch.float32) / 255.0
        h = self.act(self.conv1(x))
        h = self.act(self.bn2(self.conv2(h), size=h_in))
        h = self.act(self.bn3(self.conv3(h), size=h_in))
        h = self.act(self.bn4(self.conv4(h), size=h_in))

        h = F.adaptive_avg_pool2d(h, (4, 4))  # 64->4, 96->6, 128->8 => all to 4
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
    def __init__(
        self,
        *,
        second_ratio: float = 0.25,
        obj_blockconv: bool = False,
        obj_block_size: int = 4,
    ) -> None:
        super().__init__()
        self.splitter = Splitter(
            second_ratio=second_ratio,
            obj_blockconv=bool(obj_blockconv),
            obj_block_size=int(obj_block_size),
        )
        self.joiner = Joiner()
        self.d_ax = Discriminator()
        self.d_be = Discriminator()

    @property
    def num_ch(self) -> int:
        return self.splitter.num_ch

    def forward(self, Au: torch.Tensor, B0: torch.Tensor) -> GeneGANOutputs:
        if Au.shape[-2:] != B0.shape[-2:]:
            raise ValueError(
                f"Expected Au and B0 to have same H,W, got {tuple(Au.shape)} vs {tuple(B0.shape)}"
            )
        out_size = int(Au.shape[-1])

        A, x = self.splitter(Au)
        B, e = self.splitter(B0)
        zeros = torch.zeros_like(x)

        Au_hat = self.joiner(A, x, out_size=out_size)
        B0_hat = self.joiner(B, zeros, out_size=out_size)
        Bu = self.joiner(B, x, out_size=out_size)
        A0 = self.joiner(A, zeros, out_size=out_size)

        return GeneGANOutputs(A=A, x=x, B=B, e=e, A0=A0, Bu=Bu, Au_hat=Au_hat, B0_hat=B0_hat)
