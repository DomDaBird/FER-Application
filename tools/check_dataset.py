"""
Dataset validation script for the Emotion Recognition project.

This script checks the dataset structure and reports:

- image counts per split and per class
- image format distribution
- corrupted or unreadable images
- suspiciously small images
- PNG files with ICC profiles (which can trigger warnings)

This script is intended as a local data quality check before training.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List
import sys

from PIL import Image, UnidentifiedImageError


# ------------------------------------------------------------
# Ensure project root is available for imports
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import config as cfg


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIN_WIDTH = 32
MIN_HEIGHT = 32


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def collect_images(folder: Path) -> List[Path]:
    """Collect all image files recursively from a folder."""
    return [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]


def check_image(path: Path) -> Dict[str, object]:
    """
    Validate a single image file.

    Returns metadata including:
    - readable
    - width
    - height
    - format
    - has_icc_profile
    - error message
    """

    result: Dict[str, object] = {
        "readable": False,
        "width": None,
        "height": None,
        "format": None,
        "has_icc_profile": False,
        "error": None,
    }

    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            img = img.convert("RGB")

            width, height = img.size
            result["width"] = width
            result["height"] = height
            result["format"] = img.format
            result["readable"] = True

        with Image.open(path) as img_meta:
            result["has_icc_profile"] = img_meta.info.get("icc_profile") is not None

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        result["error"] = str(exc)

    return result


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------
# Main dataset validation
# ------------------------------------------------------------

def main() -> None:
    """Run the dataset validation."""

    data_root = cfg.DATA_DIR

    if not data_root.exists():
        print(f"Dataset folder not found: {data_root}")
        return

    expected_splits = ["train", "val", "test"]

    counts_by_split_class: Dict[str, Dict[str, int]] = defaultdict(dict)
    format_counter: Counter[str] = Counter()

    broken_files: List[Path] = []
    tiny_files: List[Path] = []
    png_icc_files: List[Path] = []

    total_images = 0

    print_section("DATASET STRUCTURE CHECK")
    print(f"Dataset root: {data_root}")

    for split in expected_splits:

        split_dir = data_root / split

        if not split_dir.exists():
            print(f"[Missing split folder] {split_dir}")
            continue

        print_section(f"SPLIT: {split}")

        for class_name in cfg.ACTIVE_CLASSES:

            class_dir = split_dir / class_name

            if not class_dir.exists():
                print(f"[Missing class folder] {class_dir}")
                counts_by_split_class[split][class_name] = 0
                continue

            image_paths = collect_images(class_dir)

            counts_by_split_class[split][class_name] = len(image_paths)

            print(f"{class_name:<10} -> {len(image_paths):>6} images")

            for path in image_paths:

                total_images += 1

                info = check_image(path)

                if not info["readable"]:
                    broken_files.append(path)
                    continue

                fmt = str(info["format"]).upper()
                format_counter[fmt] += 1

                width = int(info["width"])
                height = int(info["height"])

                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    tiny_files.append(path)

                if path.suffix.lower() == ".png" and info["has_icc_profile"]:
                    png_icc_files.append(path)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_section("SUMMARY")

    print(f"Total images checked: {total_images}")

    print("\nCounts by split and class:")

    for split in expected_splits:

        if split not in counts_by_split_class:
            continue

        print(f"\n{split}:")

        split_total = 0

        for class_name in cfg.ACTIVE_CLASSES:
            n = counts_by_split_class[split].get(class_name, 0)
            split_total += n
            print(f"  {class_name:<10} {n:>6}")

        print(f"  {'total':<10} {split_total:>6}")

    # --------------------------------------------------------
    # Format distribution
    # --------------------------------------------------------

    print_section("FORMAT DISTRIBUTION")

    for fmt, count in format_counter.most_common():
        print(f"{fmt:<10} {count:>6}")

    # --------------------------------------------------------
    # Broken files
    # --------------------------------------------------------

    print_section("BROKEN OR UNREADABLE FILES")

    if broken_files:

        print(f"Found {len(broken_files)} broken files:")

        for p in broken_files[:50]:
            print(f" - {p}")

        if len(broken_files) > 50:
            print(f"... and {len(broken_files) - 50} more")

    else:
        print("No broken files found.")

    # --------------------------------------------------------
    # Tiny images
    # --------------------------------------------------------

    print_section("SUSPICIOUSLY SMALL IMAGES")

    if tiny_files:

        print(f"Found {len(tiny_files)} very small images:")

        for p in tiny_files[:50]:
            print(f" - {p}")

        if len(tiny_files) > 50:
            print(f"... and {len(tiny_files) - 50} more")

    else:
        print("No suspiciously small images found.")

    # --------------------------------------------------------
    # PNG ICC profiles
    # --------------------------------------------------------

    print_section("PNG FILES WITH ICC PROFILE")

    if png_icc_files:

        print(f"Found {len(png_icc_files)} PNG files with ICC metadata:")

        for p in png_icc_files[:50]:
            print(f" - {p}")

        if len(png_icc_files) > 50:
            print(f"... and {len(png_icc_files) - 50} more")

    else:
        print("No PNG ICC profile issues detected.")

    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    print_section("CLASS BALANCE OVERVIEW")

    for split in expected_splits:

        if split not in counts_by_split_class:
            continue

        values = [
            counts_by_split_class[split].get(cls, 0)
            for cls in cfg.ACTIVE_CLASSES
        ]

        if min(values) == 0:
            print(f"{split}: imbalance check not possible (missing class)")
            continue

        max_count = max(values)
        min_count = min(values)

        ratio = max_count / min_count

        print(
            f"{split}: min={min_count}, max={max_count}, "
            f"max/min ratio={ratio:.2f}"
        )

    print("\nDataset check finished.")


if __name__ == "__main__":
    main()