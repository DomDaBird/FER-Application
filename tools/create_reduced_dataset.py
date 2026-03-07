"""
Reduced dataset builder for the Emotion Recognition project.

This script creates a balanced reduced dataset from the original dataset.

Behavior:
- train split: randomly samples the same number of images per class
- val split: copied unchanged
- test split: copied unchanged

The script is intended for faster and more stable training on a laptop.
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import config as cfg


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TRAIN_SAMPLES_PER_CLASS = 10000
OUTPUT_DIR_NAME = "dataset_reduced"


def collect_images(folder: Path) -> list[Path]:
    """This function collects all valid image files recursively from a folder."""
    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTS
    ]


def ensure_clean_dir(path: Path) -> None:
    """This function recreates an output directory from scratch."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_files(files: list[Path], dst_dir: Path) -> None:
    """This function copies image files into a destination directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dst_dir / src.name)


def build_reduced_train_split(
    src_root: Path,
    dst_root: Path,
    samples_per_class: int,
    seed: int,
) -> None:
    """
    This function creates a balanced reduced training split.

    It samples the same number of images per class.
    """
    rng = random.Random(seed)

    print("\nBuilding reduced train split...")

    for class_name in cfg.ACTIVE_CLASSES:
        src_class_dir = src_root / "train" / class_name
        dst_class_dir = dst_root / "train" / class_name

        images = collect_images(src_class_dir)

        if len(images) < samples_per_class:
            raise ValueError(
                f"Class '{class_name}' has only {len(images)} images, "
                f"but {samples_per_class} were requested."
            )

        rng.shuffle(images)
        selected = images[:samples_per_class]

        copy_files(selected, dst_class_dir)

        print(
            f"train/{class_name:<10} -> copied {len(selected):>5} images "
            f"(from {len(images):>5})"
        )


def copy_split_unchanged(src_root: Path, dst_root: Path, split: str) -> None:
    """This function copies a full split unchanged."""
    print(f"\nCopying {split} split unchanged...")

    for class_name in cfg.ACTIVE_CLASSES:
        src_class_dir = src_root / split / class_name
        dst_class_dir = dst_root / split / class_name

        images = collect_images(src_class_dir)
        copy_files(images, dst_class_dir)

        print(f"{split}/{class_name:<10} -> copied {len(images):>5} images")


def main() -> None:
    """This function creates the reduced balanced dataset."""
    src_root = cfg.PROJECT_ROOT / "dataset"
    dst_root = cfg.PROJECT_ROOT / OUTPUT_DIR_NAME

    if not src_root.exists():
        print(f"Source dataset not found: {src_root}")
        return

    print("Reduced Dataset Builder")
    print(f"Source dataset: {src_root}")
    print(f"Target dataset: {dst_root}")
    print(f"Train samples per class: {DEFAULT_TRAIN_SAMPLES_PER_CLASS}")

    ensure_clean_dir(dst_root)

    build_reduced_train_split(
        src_root=src_root,
        dst_root=dst_root,
        samples_per_class=DEFAULT_TRAIN_SAMPLES_PER_CLASS,
        seed=cfg.SEED,
    )

    copy_split_unchanged(src_root=src_root, dst_root=dst_root, split="val")
    copy_split_unchanged(src_root=src_root, dst_root=dst_root, split="test")

    print("\nReduced dataset finished successfully.")
    print(f"Output folder: {dst_root}")


if __name__ == "__main__":
    main()