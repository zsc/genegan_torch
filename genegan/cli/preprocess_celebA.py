from __future__ import annotations

import argparse
import csv
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import tqdm


# average landmarks (from official GeneGAN preprocess.py)
mean_face_lm5p = np.array(
    [
        [-0.17607, -0.172844],  # left eye pupil
        [0.1736, -0.17356],  # right eye pupil
        [-0.00182, 0.0357164],  # nose tip
        [-0.14617, 0.20185],  # left mouth corner
        [0.14496, 0.19943],  # right mouth corner
    ]
)


def _get_align_5p_mat23_size_256(lm: np.ndarray) -> np.ndarray:
    width = 256
    mf = mean_face_lm5p.copy()

    ratio = 70.0 / (256.0 * 0.34967)

    left_eye_pupil_y = mf[0][1]
    ratioy = (left_eye_pupil_y * ratio + 0.5) * (1 + 1.42)
    mf[:, 0] = (mf[:, 0] * ratio + 0.5) * width
    mf[:, 1] = (mf[:, 1] * ratio + 0.5) * width / ratioy
    mx = mf[:, 0].mean()
    my = mf[:, 1].mean()
    dmx = lm[:, 0].mean()
    dmy = lm[:, 1].mean()
    mat = np.zeros((3, 3), dtype=float)
    ux = mf[:, 0] - mx
    uy = mf[:, 1] - my
    dux = lm[:, 0] - dmx
    duy = lm[:, 1] - dmy
    c1 = (ux * dux + uy * duy).sum()
    c2 = (ux * duy - uy * dux).sum()
    c3 = (dux**2 + duy**2).sum()
    a = c1 / c3
    b = c2 / c3

    kx = 1
    ky = 1

    s = c3 / (c1**2 + c2**2)
    ka = c1 * s
    kb = c2 * s

    transform = np.zeros((2, 3), dtype=float)
    transform[0][0] = kx * a
    transform[0][1] = kx * b
    transform[0][2] = mx - kx * a * dmx - kx * b * dmy
    transform[1][0] = -ky * b
    transform[1][1] = ky * a
    transform[1][2] = my - ky * a * dmy + ky * b * dmx
    return transform


def get_align_5p_mat23(lm5p: np.ndarray, size: int) -> np.ndarray:
    mat23 = _get_align_5p_mat23_size_256(lm5p.copy())
    mat23 *= size / 256
    return mat23


def align_given_lm5p(img: np.ndarray, lm5p: np.ndarray, size: int) -> np.ndarray:
    mat23 = get_align_5p_mat23(lm5p, size)
    return cv2.warpAffine(img, mat23, (size, size))


def align_face_5p(img: np.ndarray, landmarks: list[int]) -> np.ndarray:
    return align_given_lm5p(img, np.array(landmarks).reshape((5, 2)), 256)


def _read_landmarks_txt(path: Path) -> dict[str, list[int]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    rows = lines[2:]
    out: dict[str, list[int]] = {}
    for line in rows:
        parts = line.split()
        fname = parts[0]
        lm = list(map(int, parts[1:11]))
        out[fname] = lm
    return out


def _read_landmarks_csv(path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] not in {"image_id", "image"}:
            raise ValueError(f"Unexpected landmark CSV header in {path}")
        for row in reader:
            if not row:
                continue
            fname = row[0]
            lm = list(map(int, row[1:11]))
            out[fname] = lm
    return out


def _default_src_dir(data_dir: Path) -> Path:
    for name in ("data", "img_align_celeba"):
        p = data_dir / name
        if p.is_dir():
            nested = p / p.name
            if nested.is_dir():
                probe = "000001.jpg"
                if (nested / probe).exists() and not (p / probe).exists():
                    return nested
            return p
    raise FileNotFoundError(f"Could not find data/img_align_celeba under {data_dir}")


def _default_landmarks_file(data_dir: Path) -> Path:
    for name in (
        "list_landmarks_celeba.txt",
        "list_landmarks_align_celeba.txt",
        "list_landmarks_align_celeba.csv",
    ):
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find landmarks file under {data_dir}")


def _load_landmarks(path: Path) -> dict[str, list[int]]:
    if path.suffix.lower() == ".csv":
        return _read_landmarks_csv(path)
    return _read_landmarks_txt(path)


def _work(src_dir: Path, out_dir: Path, item: tuple[str, list[int]]) -> int:
    fname, lm = item
    src = src_dir / fname
    dst = out_dir / fname
    img = cv2.imread(str(src))
    if img is None:
        return 0
    aligned = align_face_5p(img, lm)
    cv2.imwrite(str(dst), aligned)
    return 1


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("genegan-preprocess-celeba")
    p.add_argument("--data_dir", required=True, type=str, help="CelebA dataset root")
    p.add_argument("--out_dir", required=True, type=str, help="Output directory")
    p.add_argument("--num_workers", default=0, type=int)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = _default_src_dir(data_dir)
    lm_path = _default_landmarks_file(data_dir)
    landmarks = _load_landmarks(lm_path)

    items = sorted(landmarks.items(), key=lambda kv: kv[0])

    if args.num_workers <= 0:
        for item in tqdm.tqdm(items, desc="align"):
            _work(src_dir, out_dir, item)
        return

    with Pool(args.num_workers) as pool:
        fn = partial(_work, src_dir, out_dir)
        list(tqdm.tqdm(pool.imap(fn, items), total=len(items), desc="align"))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
