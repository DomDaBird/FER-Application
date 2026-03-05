"""
Data pipeline for the Facial Emotion Recognition project.

This module builds the TensorFlow tf.data pipeline used for
training, validation and testing.

Expected dataset structure:

dataset/
    train/
        angry/
        fear/
        happy/
        sad/
        surprise/
    val/
        angry/
        fear/
        happy/
        sad/
        surprise/
    test/
        angry/
        fear/
        happy/
        sad/
        surprise/

Each class is stored inside its own folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
import numpy as np
import config as cfg

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# Helper: load directory as dataset
# ============================================================

def _image_ds_from_dir(
    root: Path,
    class_names: List[str],
    img_size: Tuple[int, int],
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    This function creates an unbatched tf.data.Dataset from a directory.

    The directory must contain one subfolder per class.

    Parameters
    ----------
    root : Path
        Path to the directory containing class folders.
    class_names : List[str]
        Ordered list of class names.
    img_size : Tuple[int, int]
        Image size (H, W) used for resizing.
    shuffle : bool
        Whether the dataset should be shuffled.
    seed : int
        Random seed for deterministic behaviour.

    Returns
    -------
    tf.data.Dataset
        Dataset of (image, one_hot_label) pairs.
    """

    return tf.keras.utils.image_dataset_from_directory(
        directory=str(root),
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=img_size,
        shuffle=shuffle,
        seed=seed,
        batch_size=None,  # batching happens later
    )


# ============================================================
# Balanced training dataset
# ============================================================

def _balanced_train_ds_exact(train_root: Path, class_names: List[str]) -> tf.data.Dataset:
    """
    This function creates a balanced training dataset.

    The method loads the entire training set once and then creates
    filtered datasets for each class. These datasets are sampled with
    equal probability to avoid class imbalance.

    This prevents majority classes from dominating the training batches.
    """

    num_classes = len(class_names)

    base_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(train_root),
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=cfg.IMG_SIZE,
        shuffle=True,
        seed=cfg.SEED,
        batch_size=None,
    )

    per_class_datasets: List[tf.data.Dataset] = []

    for class_index in range(num_classes):

        def _predicate(x, y, ci=class_index):
            return tf.equal(tf.argmax(y, axis=-1), ci)

        ds_ci = base_ds.filter(_predicate)
        per_class_datasets.append(ds_ci)

    balanced = tf.data.Dataset.sample_from_datasets(
        per_class_datasets,
        weights=[1.0 / num_classes] * num_classes,
        seed=cfg.SEED,
    )

    return balanced


# ============================================================
# Main dataset builder
# ============================================================

def make_datasets(data_root: Path):
    """
    This function creates the train, validation and test datasets.

    Balancing is applied only to the training dataset depending on
    the configuration parameter cfg.BALANCE_MODE.

    Returns
    -------
    train_ds : tf.data.Dataset
    val_ds : tf.data.Dataset
    test_ds : tf.data.Dataset
    class_names : List[str]
    """

    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "val"
    test_dir = Path(data_root) / "test"

    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

    expected = list(cfg.ACTIVE_CLASSES)

    if set(class_names) != set(expected):
        raise ValueError(
            f"Class folders do not match cfg.ACTIVE_CLASSES\n"
            f"Found:    {class_names}\n"
            f"Expected: {expected}"
        )

    # enforce exact class order from config
    class_names = expected

    # ------------------------------------------------
    # TRAIN
    # ------------------------------------------------

    if cfg.BALANCE_MODE == "roundrobin":
        train_raw = _balanced_train_ds_exact(train_dir, class_names)
    else:
        train_raw = _image_ds_from_dir(
            root=train_dir,
            class_names=class_names,
            img_size=cfg.IMG_SIZE,
            shuffle=True,
            seed=cfg.SEED,
        )

    train_raw = train_raw.shuffle(
        buffer_size=cfg.SHUFFLE_BUFFER_SIZE,
        seed=cfg.SEED,
        reshuffle_each_iteration=True,
    )

    # ------------------------------------------------
    # VALIDATION
    # ------------------------------------------------

    val_raw = _image_ds_from_dir(
        root=val_dir,
        class_names=class_names,
        img_size=cfg.IMG_SIZE,
        shuffle=False,
        seed=cfg.SEED,
    )

    # ------------------------------------------------
    # TEST
    # ------------------------------------------------

    test_raw = _image_ds_from_dir(
        root=test_dir,
        class_names=class_names,
        img_size=cfg.IMG_SIZE,
        shuffle=False,
        seed=cfg.SEED,
    )

    # ------------------------------------------------
    # Pipeline (batch + prefetch)
    # ------------------------------------------------

    train_ds = (
        train_raw
        .batch(cfg.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_raw
        .batch(cfg.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        test_raw
        .batch(cfg.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, class_names