from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class CelebAImageTransform:
    img_size: int = 64

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if image.size != (self.img_size, self.img_size):
            image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32)  # [H,W,3] in [0,255]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]
        return tensor


def load_pil_rgb(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")

