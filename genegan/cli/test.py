from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from genegan.data.transforms import CelebAImageTransform, load_pil_rgb
from genegan.models.genegan import GeneGAN
from genegan.utils.device import resolve_device
from genegan.utils.io import load_checkpoint


def _load_image(path: str | Path, *, img_size: int = 64) -> torch.Tensor:
    img = load_pil_rgb(path)
    t = CelebAImageTransform(img_size=img_size)(img)
    return t.unsqueeze(0)


def _to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    x = image.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 255.0)
    x = x.to(torch.uint8)[0].permute(1, 2, 0).contiguous().numpy()
    return x


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("genegan-test")
    p.add_argument("--mode", required=True, choices=["swap", "interpolation", "matrix"])
    p.add_argument("--ckpt", required=True, type=str, help="Checkpoint .pt path")
    p.add_argument("--device", default=None, choices=["mps", "cuda", "cpu"])
    p.add_argument("--input", required=True, type=str, help="Source image (to change attribute)")

    p.add_argument("--target", type=str, help="Target image (has attribute)")
    p.add_argument("--num", default=5, type=int, help="Interpolation steps")
    p.add_argument(
        "--targets",
        nargs=4,
        type=str,
        help="Four target images (for matrix mode)",
    )
    p.add_argument("--size", nargs=2, default=[5, 5], type=int, help="Matrix size m n")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    device = resolve_device(args.device).device

    # reconstruct model (second_ratio from checkpoint config if available)
    ckpt_path = Path(args.ckpt)
    tmp_model = GeneGAN(second_ratio=0.25)
    ckpt = load_checkpoint(
        ckpt_path,
        splitter=tmp_model.splitter,
        joiner=tmp_model.joiner,
        d_ax=tmp_model.d_ax,
        d_be=tmp_model.d_be,
        map_location="cpu",
    )
    second_ratio = float(ckpt.get("config", {}).get("second_ratio", 0.25))
    model = tmp_model if second_ratio == 0.25 else GeneGAN(second_ratio=second_ratio)
    if model is not tmp_model:
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

    src = _load_image(args.input).to(device=device, dtype=torch.float32)

    if args.mode == "swap":
        if not args.target:
            raise SystemExit("--target is required for swap mode")
        att = _load_image(args.target).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            A, x = model.splitter(att)
            B, _ = model.splitter(src)
            zeros = torch.zeros_like(x)
            out1 = model.joiner(B, x)  # src with attribute
            out2 = model.joiner(A, zeros)  # target without attribute

        from PIL import Image

        Image.fromarray(_to_uint8_hwc(out1)).save("out1.jpg", quality=95)
        Image.fromarray(_to_uint8_hwc(out2)).save("out2.jpg", quality=95)
        return

    if args.mode == "interpolation":
        if not args.target:
            raise SystemExit("--target is required for interpolation mode")
        att = _load_image(args.target).to(device=device, dtype=torch.float32)
        from PIL import Image

        with torch.no_grad():
            B, _ = model.splitter(src)
            _, x = model.splitter(att)

            outs = [_to_uint8_hwc(src)]
            for i in range(1, args.num + 1):
                lam = i / float(args.num)
                out_i = model.joiner(B, x * lam)
                outs.append(_to_uint8_hwc(out_i))

        canvas = np.concatenate(outs, axis=1)
        Image.fromarray(canvas).save("interpolation.jpg", quality=95)
        return

    if args.mode == "matrix":
        if not args.targets or len(args.targets) != 4:
            raise SystemExit("--targets requires 4 images for matrix mode")
        m, n = int(args.size[0]), int(args.size[1])
        if m < 2 or n < 2:
            raise SystemExit("--size must be at least 2 2")

        att_imgs = torch.cat([_load_image(p) for p in args.targets], dim=0).to(
            device=device, dtype=torch.float32
        )

        from PIL import Image

        with torch.no_grad():
            attrs = []
            for i in range(4):
                _, x = model.splitter(att_imgs[i : i + 1])
                attrs.append(x)
            B, _ = model.splitter(src)

            rows = [[1 - i / float(m - 1), i / float(m - 1)] for i in range(m)]
            cols = [[1 - i / float(n - 1), i / float(n - 1)] for i in range(n)]
            four_tuple = []
            for row in rows:
                for col in cols:
                    four_tuple.append(
                        [row[0] * col[0], row[0] * col[1], row[1] * col[0], row[1] * col[1]]
                    )

            h = w = 64
            out = np.zeros((0, w * n, 3), dtype=np.uint8)
            cnt = 0
            for _i in range(m):
                out_row = np.zeros((h, 0, 3), dtype=np.uint8)
                for _j in range(n):
                    four = four_tuple[cnt]
                    attribute = sum(four[k] * attrs[k] for k in range(4))
                    img = model.joiner(B, attribute)
                    out_row = np.concatenate((out_row, _to_uint8_hwc(img)), axis=1)
                    cnt += 1
                out = np.concatenate((out, out_row), axis=0)

        att_np = [_to_uint8_hwc(att_imgs[i : i + 1]) for i in range(4)]
        if m > 2:
            pad = 255 * np.ones(((m - 2) * h, w, 3), dtype=np.uint8)
            first_col = np.concatenate((att_np[0], pad, att_np[2]), axis=0)
            last_col = np.concatenate((att_np[1], pad, att_np[3]), axis=0)
        else:
            first_col = np.concatenate((att_np[0], att_np[2]), axis=0)
            last_col = np.concatenate((att_np[1], att_np[3]), axis=0)

        canvas = np.concatenate((first_col, out, last_col), axis=1)
        Image.fromarray(canvas).save("four_matrix.jpg", quality=95)
        return


if __name__ == "__main__":
    main()
