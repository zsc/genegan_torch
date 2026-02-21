from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class CelebAImageTransform:
    img_size: int = 128

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if image.size != (self.img_size, self.img_size):
            image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        tensor_u8 = TF.pil_to_tensor(image)  # [3,H,W] uint8 in [0,255]
        return tensor_u8.to(dtype=torch.float32)


def load_pil_rgb(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")
