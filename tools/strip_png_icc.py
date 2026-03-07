"""
PNG metadata cleanup script for the Emotion Recognition project.

This script searches the dataset for PNG files with ICC profile metadata
and rewrites them without the ICC profile. This can reduce noisy libpng
warnings during training and evaluation.

The image content is preserved as closely as possible.
"""

from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import config as cfg


VALID_EXTS = {".png"}


def find_pngs(root: Path) -> list[Path]:
    """This function collects all PNG files recursively."""
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTS
    ]


def has_icc_profile(path: Path) -> bool:
    """This function checks whether a PNG file contains ICC profile metadata."""
    try:
        with Image.open(path) as img:
            return img.info.get("icc_profile") is not None
    except Exception:
        return False


def rewrite_png_without_icc(path: Path) -> bool:
    """
    This function rewrites a PNG file without ICC metadata.

    It returns True if the file was rewritten successfully.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.save(path, format="PNG")
        return True
    except Exception as exc:
        print(f"[ERROR] Could not rewrite: {path}")
        print(f"        {exc}")
        return False


def main() -> None:
    """This function removes ICC profile metadata from PNG files in the dataset."""
    data_root = cfg.DATA_DIR

    if not data_root.exists():
        print(f"Dataset folder not found: {data_root}")
        return

    png_files = find_pngs(data_root)

    if not png_files:
        print("No PNG files found.")
        return

    files_with_icc = [path for path in png_files if has_icc_profile(path)]

    print(f"Found {len(png_files)} PNG files in total.")
    print(f"Found {len(files_with_icc)} PNG files with ICC profile metadata.")

    if not files_with_icc:
        print("No cleanup needed.")
        return

    rewritten = 0
    failed = 0

    for path in files_with_icc:
        ok = rewrite_png_without_icc(path)
        if ok:
            rewritten += 1
        else:
            failed += 1

    print("\nCleanup finished.")
    print(f"Rewritten PNG files: {rewritten}")
    print(f"Failed files:        {failed}")


if __name__ == "__main__":
    main()