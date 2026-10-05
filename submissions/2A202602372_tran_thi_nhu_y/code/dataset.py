"""Dataset loading, split validation, image transforms, and DataLoaders."""
from __future__ import annotations

import os
import random
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
               "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_TOTAL = 17_509


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the author-provided train/validation/test CSVs without resplitting."""
    if fold not in range(5):
        raise ValueError("fold must be one of 0, 1, 2, 3, 4")
    root = Path(labels_dir)
    frames = tuple(pd.read_csv(root / f"{split}_subset{fold}.csv")
                   for split in ("train", "val", "test"))
    for split, frame in zip(("train", "val", "test"), frames):
        missing = {"Filename", "Label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{split} CSV missing columns: {sorted(missing)}")
        if frame["Filename"].duplicated().any():
            raise ValueError(f"{split} CSV contains duplicate filenames")
        if not frame["Label"].between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"{split} CSV has labels outside [0, 8]")
    species = {int(label): str(name) for label, name in
                pd.read_csv(root / "labels.csv")[["Label", "Species"]].drop_duplicates().itertuples(index=False, name=None)}
    for frame in frames:
        if "Species" not in frame.columns:
            frame["Species"] = frame["Label"].map(species)
    return frames


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict[str, Any]:
    """Validate disjointness, full dataset coverage, class counts, and image files."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    names = {key: set(df["Filename"].astype(str)) for key, df in frames.items()}
    overlap = {"train_val": sorted(names["train"] & names["val"]),
               "train_test": sorted(names["train"] & names["test"]),
               "val_test": sorted(names["val"] & names["test"])}
    if any(overlap.values()):
        raise ValueError(f"Split overlap found: { {k: len(v) for k, v in overlap.items()} }")
    union = set.union(*names.values())
    if len(union) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} unique images across fold, found {len(union)}")
    root = Path(images_dir)
    if root.is_file() and zipfile.is_zipfile(root):
        with zipfile.ZipFile(root) as archive:
            available = {Path(name).name for name in archive.namelist() if not name.endswith("/")}
        missing = sorted(union - available)
    else:
        missing = sorted(name for name in union if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"{len(missing)} image files are missing; examples: {missing[:10]}")
    sizes = {name: len(df) for name, df in frames.items()}
    expected_ratios = {"train": 0.60, "val": 0.20, "test": 0.20}
    bad_ratios = {name: sizes[name] / EXPECTED_TOTAL for name in frames
                  if abs(sizes[name] / EXPECTED_TOTAL - expected_ratios[name]) > 0.01}
    if bad_ratios:
        raise ValueError(f"Split proportions differ from 60/20/20 by more than 1 point: {bad_ratios}")
    per_class = {name: {CLASS_NAMES[int(label)]: int(count)
                        for label, count in df["Label"].value_counts().sort_index().items()}
                 for name, df in frames.items()}
    report = {"n": sizes, "per_class": per_class,
              "overlap": {key: len(value) for key, value in overlap.items()},
              "union": len(union), "missing_images": 0}
    print(report)
    return report


class ImageTransform:
    """Torchvision-independent ImageNet preprocessing with reproducible basic augmentations."""
    def __init__(self, train: bool, img_size: int, aug: str) -> None:
        self.train, self.img_size, self.aug = train, img_size, aug

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.train:
            width, height = image.size
            scale = random.uniform(0.65, 1.0)
            crop_w, crop_h = max(1, int(width * scale)), max(1, int(height * scale))
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
            image = image.crop((left, top, left + crop_w, top + crop_h))
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if self.aug == "color":
                image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            elif self.aug in {"trivial", "randaug"}:
                operations = [
                    lambda img: img.rotate(random.uniform(-25, 25)),
                    lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3)),
                    lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2)),
                    lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
                ]
                for operation in random.sample(operations, 1 if self.aug == "trivial" else 2):
                    image = operation(image)
            if self.aug not in {"basic", "color", "trivial", "randaug"}:
                raise ValueError(f"Unknown augmentation: {self.aug}")
        else:
            # Resize shortest side to 256, then center crop to configured model resolution.
            width, height = image.size
            scale = max(256, self.img_size) / min(width, height)
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
            left = max(0, (image.width - self.img_size) // 2)
            top = max(0, (image.height - self.img_size) // 2)
            image = image.crop((left, top, left + self.img_size, top + self.img_size))
        if image.size != (self.img_size, self.img_size):
            image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
        std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
        return (tensor - mean) / std


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> ImageTransform:
    """Build train augmentation or deterministic center-crop evaluation transform."""
    return ImageTransform(train, img_size, aug)


class DeepWeedsDataset(Dataset):
    """Read image and integer class from a DeepWeeds CSV frame."""
    def __init__(self, df: pd.DataFrame, images_dir: str | Path,
                 transform: Callable[[Image.Image], torch.Tensor] | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self._archive_path = self.images_dir if self.images_dir.is_file() and zipfile.is_zipfile(self.images_dir) else None
        self._archive = None
        self._archive_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archive"] = None
        state["_archive_pid"] = None
        return state

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        if self._archive_path is None:
            with Image.open(self.images_dir / filename) as source:
                image = source.convert("RGB")
        else:
            current_pid = os.getpid()
            if self._archive is None or self._archive_pid != current_pid:
                self._archive = zipfile.ZipFile(self._archive_path)
                self._archive_pid = current_pid
            with self._archive.open(filename) as source:
                image = Image.open(source).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["Label"]), filename


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path,
                transform: Callable[[Image.Image], torch.Tensor], batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Create deterministic-order evaluation or shuffled/balanced training loader."""
    if sampler not in (None, "balanced"):
        raise ValueError("sampler must be None or 'balanced'")
    dataset = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator().manual_seed(torch.initial_seed())
    weights = None
    shuffle = train and sampler is None
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        sample_weights = df["Label"].map(lambda label: 1.0 / counts[label]).to_numpy()
        weights = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                        len(sample_weights), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=weights,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train, worker_init_fn=_seed_worker, generator=generator,
                      persistent_workers=num_workers > 0)
