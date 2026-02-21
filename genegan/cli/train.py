from __future__ import annotations

import argparse

from genegan.utils.device import resolve_device
from genegan.trainer import TrainConfig, train


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("genegan-train")
    p.add_argument("--attribute", required=True, type=str, help="CelebA attribute name")
    p.add_argument("--data_root", required=True, type=str, help="CelebA root directory")
    p.add_argument("--exp_dir", required=True, type=str, help="Experiment output dir")

    p.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device: auto|mps|cpu|cuda|cuda:N (default: auto mps->cuda->cpu)",
    )
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--num_workers", default=0, type=int)
    p.add_argument(
        "--max_images",
        default=None,
        type=int,
        help="Limit images per split for smoke runs (e.g. 2000)",
    )
    p.add_argument(
        "--resume_ckpt",
        default=None,
        type=str,
        help="Resume from a checkpoint .pt (loads model + optim state, continues at iter+1)",
    )

    p.add_argument("--max_iter", default=300000, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--img_size", default=128, type=int)

    p.add_argument("--g_lr", default=5e-5, type=float)
    p.add_argument("--d_lr", default=5e-5, type=float)
    p.add_argument("--second_ratio", default=0.25, type=float)
    p.add_argument("--weight_decay", default=5e-5, type=float)
    p.add_argument("--critic_clip", default=0.01, type=float)
    p.add_argument("--critic_every", default=500, type=int)
    p.add_argument("--critic_n", default=1, type=int)
    p.add_argument("--critic_n_500", default=100, type=int)

    p.add_argument(
        "--save_every",
        default=None,
        type=int,
        help="Checkpoint interval (default: 500 on mps/cpu, 20000 on cuda)",
    )
    p.add_argument(
        "--sample_every",
        default=None,
        type=int,
        help="Sample image interval (default: 500 on mps/cpu, 5000 on cuda)",
    )
    p.add_argument("--log_every", default=20, type=int)

    p.add_argument(
        "--image_dir",
        default=None,
        type=str,
        help="Image directory relative to data_root (default: align_5p/img_align_celeba)",
    )
    p.add_argument(
        "--attr_file",
        default=None,
        type=str,
        help="Attribute file path relative to data_root (default: list_attr_celeba.txt/csv)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    device_type = resolve_device(args.device).type
    save_every = args.save_every
    if save_every is None:
        save_every = 20000 if device_type == "cuda" else 500
    sample_every = args.sample_every
    if sample_every is None:
        sample_every = 5000 if device_type == "cuda" else 500

    cfg = TrainConfig(
        attribute=args.attribute,
        data_root=args.data_root,
        image_dir=args.image_dir,
        attr_file=args.attr_file,
        exp_dir=args.exp_dir,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        max_images_per_split=args.max_images,
        resume_ckpt=args.resume_ckpt,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        img_size=args.img_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        second_ratio=args.second_ratio,
        weight_decay=args.weight_decay,
        critic_clip=args.critic_clip,
        critic_every=args.critic_every,
        critic_n=args.critic_n,
        critic_n_500=args.critic_n_500,
        save_every=int(save_every),
        sample_every=int(sample_every),
        log_every=args.log_every,
    )
    train(cfg)


if __name__ == "__main__":
    main()
