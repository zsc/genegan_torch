from __future__ import annotations

import argparse
from pathlib import Path


def _iter_ckpts(ckpt_dir: Path) -> list[Path]:
    return sorted(ckpt_dir.glob("iter_*.pt"))


def prune_checkpoints(*, ckpt_dir: Path, keep: int, dry_run: bool) -> None:
    if keep < 1:
        raise ValueError("--keep must be >= 1")
    ckpt_dir = ckpt_dir.resolve()
    ckpts = _iter_ckpts(ckpt_dir)
    if len(ckpts) <= keep:
        return

    to_delete = ckpts[: -keep]
    for p in to_delete:
        if dry_run:
            print(f"[dry-run] delete {p}")
        else:
            p.unlink()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, type=str)
    ap.add_argument("--keep", default=2, type=int, help="Keep last N iter_*.pt (default: 2)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_dir():
        raise SystemExit(f"--ckpt_dir is not a directory: {ckpt_dir}")

    prune_checkpoints(ckpt_dir=ckpt_dir, keep=int(args.keep), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()

