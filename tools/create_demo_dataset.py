"""
Demo dataset builder for the Emotion Recognition project.

This script creates a very small dataset used for demonstration
and testing in the Streamlit application.

The script randomly selects a few images from the test split.
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import config as cfg


SAMPLES_PER_CLASS = 10
OUTPUT_FOLDER = "demo_data"


def collect_images(folder: Path) -> list[Path]:
    """This function collects all image files from a folder."""
    return [p for p in folder.glob("*") if p.is_file()]


def main() -> None:
    """This function builds the demo dataset."""
    src_root = cfg.DATA_DIR / "test"
    dst_root = cfg.PROJECT_ROOT / OUTPUT_FOLDER

    if dst_root.exists():
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True)

    print("Creating demo dataset")

    for cls in cfg.ACTIVE_CLASSES:
        src_class = src_root / cls
        dst_class = dst_root / cls
        dst_class.mkdir()

        images = collect_images(src_class)

        selected = random.sample(images, SAMPLES_PER_CLASS)

        for img in selected:
            shutil.copy2(img, dst_class / img.name)

        print(f"{cls}: {len(selected)} images")

    print("\nDemo dataset created:")
    print(dst_root)


if __name__ == "__main__":
    main()