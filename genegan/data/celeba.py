from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from genegan.data.transforms import CelebAImageTransform, load_pil_rgb


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _read_attr_txt(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    with path.open("r", encoding="utf-8") as f:
        _n = f.readline()
        header = f.readline().strip().split()
        attrs = header
        rows: dict[str, list[int]] = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            fname = parts[0]
            vals = [int(x) for x in parts[1:]]
            rows[fname] = vals
    return attrs, rows


def _read_attr_csv(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] not in {"image_id", "image"}:
            raise ValueError(f"Unexpected CSV header in {path}")
        attrs = header[1:]
        rows: dict[str, list[int]] = {}
        for row in reader:
            if not row:
                continue
            fname = row[0]
            vals = [int(x) for x in row[1:]]
            rows[fname] = vals
    return attrs, rows


@dataclass(frozen=True)
class CelebAAttributeIndex:
    attrs: list[str]
    by_file: dict[str, list[int]]

    @staticmethod
    def load(path: str | Path) -> "CelebAAttributeIndex":
        path = Path(path)
        if path.suffix.lower() == ".csv":
            attrs, rows = _read_attr_csv(path)
        else:
            attrs, rows = _read_attr_txt(path)
        return CelebAAttributeIndex(attrs=attrs, by_file=rows)

    def attribute_names(self) -> list[str]:
        return list(self.attrs)

    def split(self, attribute: str) -> tuple[list[str], list[str]]:
        if attribute not in self.attrs:
            raise ValueError(
                f"Unknown attribute: {attribute}. Available: {', '.join(self.attrs)}"
            )
        idx = self.attrs.index(attribute)
        with_attr: list[str] = []
        without_attr: list[str] = []
        for fname, vals in self.by_file.items():
            v = vals[idx]
            if v == 1:
                with_attr.append(fname)
            elif v == -1:
                without_attr.append(fname)
        return with_attr, without_attr


class CelebAImageDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        image_dir: str | Path,
        filenames: Iterable[str],
        transform: CelebAImageTransform,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.filenames = list(filenames)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fname = self.filenames[idx]
        path = self.image_dir / fname
        img = load_pil_rgb(path)
        return self.transform(img)


def default_attr_file(data_root: str | Path) -> Path:
    root = Path(data_root)
    for name in ("list_attr_celeba.txt", "list_attr_celeba.csv"):
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find list_attr_celeba.(txt|csv) under {root.resolve()}"
    )


def default_image_dir(data_root: str | Path) -> Path:
    root = Path(data_root)
    for name in ("align_5p", "img_align_celeba", "data"):
        p = root / name
        if p.is_dir():
            return p
    if root.is_dir():
        return root
    raise FileNotFoundError(f"Could not find image directory under {root.resolve()}")


@dataclass(frozen=True)
class CelebASplitDatasets:
    with_attr: CelebAImageDataset
    without_attr: CelebAImageDataset
    attribute: str
    image_dir: Path
    attr_file: Path


def build_celeba_attribute_datasets(
    *,
    data_root: str | Path,
    attribute: str,
    image_dir: str | Path | None = None,
    attr_file: str | Path | None = None,
    img_size: int = 128,
    max_images_per_split: int | None = None,
) -> CelebASplitDatasets:
    root = Path(data_root)
    if attr_file is None:
        attr_path = default_attr_file(root)
    else:
        attr_path = Path(attr_file)
        if not attr_path.is_absolute():
            attr_path = root / attr_path
    attr_path = attr_path.resolve()

    if image_dir is None:
        img_dir = default_image_dir(root)
    else:
        img_dir = Path(image_dir)
        if not img_dir.is_absolute():
            img_dir = root / img_dir
    img_dir = img_dir.resolve()
    # Common CelebA archives have an extra nesting: img_align_celeba/img_align_celeba/*.jpg
    nested = img_dir / img_dir.name
    if nested.is_dir():
        probe = "000001.jpg"
        if (nested / probe).exists() and not (img_dir / probe).exists():
            img_dir = nested.resolve()

    index = CelebAAttributeIndex.load(attr_path)
    with_attr_names, without_attr_names = index.split(attribute)
    if max_images_per_split is not None:
        if max_images_per_split <= 0:
            raise ValueError("max_images_per_split must be positive")
        with_attr_names = with_attr_names[:max_images_per_split]
        without_attr_names = without_attr_names[:max_images_per_split]

    transform = CelebAImageTransform(img_size=img_size)
    ds_with = CelebAImageDataset(img_dir, with_attr_names, transform=transform)
    ds_without = CelebAImageDataset(img_dir, without_attr_names, transform=transform)
    return CelebASplitDatasets(
        with_attr=ds_with,
        without_attr=ds_without,
        attribute=attribute,
        image_dir=img_dir,
        attr_file=attr_path,
    )


def make_dataloader(
    dataset: Dataset[torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool = False,
) -> DataLoader[torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )
