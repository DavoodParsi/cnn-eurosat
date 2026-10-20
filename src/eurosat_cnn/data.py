"""Dataset discovery, loading, class-imbalance analysis and sanity checks."""

from pathlib import Path

import tensorflow as tf

from eurosat_cnn.config import ProjectPaths

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


# --------------------------------------------
# Class Names & Dataset Structure
# --------------------------------------------

def _list_class_dirs(split_dir: Path) -> list[str]:
    """Return the sorted names of all class subdirectories."""
    return sorted(d.name for d in split_dir.iterdir() if d.is_dir())


def get_class_names(paths: ProjectPaths, num_classes: int) -> list[str]:
    """Validate the dataset structure and return the class names.

    Checks that train/val/test exist, that val/test contain exactly the
    same classes as train, and that the class count matches num_classes.
    Fails fast so a bad dataset can never silently corrupt training.

    Args:
        paths: Project paths for the current model.
        num_classes: Expected number of classes.

    Returns:
        Sorted list of class names (the index is the label id).
    """
    if not paths.dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {paths.dataset_dir}"
        )

    splits = {
        "train": paths.train_dir,
        "validation": paths.val_dir,
        "test": paths.test_dir,
    }

    for split_name, split_dir in splits.items():
        if not split_dir.exists():
            raise FileNotFoundError(
                f"{split_name.capitalize()} directory not found: {split_dir}"
            )

    class_names = _list_class_dirs(paths.train_dir)

    if not class_names:
        raise ValueError(
            f"No class directories found in training directory: "
            f"{paths.train_dir}"
        )

    for split_name in ("validation", "test"):
        split_classes = _list_class_dirs(splits[split_name])

        if split_classes != class_names:
            missing = set(class_names) - set(split_classes)
            extra = set(split_classes) - set(class_names)
            raise ValueError(
                f"Class mismatch in {split_name} directory "
                f"({splits[split_name]}).\n"
                f"Missing: {sorted(missing) if missing else 'None'}\n"
                f"Extra   : {sorted(extra) if extra else 'None'}"
            )

    if len(class_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} classes, "
            f"but found {len(class_names)} classes in {paths.train_dir}"
        )

    return class_names


# --------------------------------------------
# Dataset Loading
# --------------------------------------------

def load_dataset(
    directory: Path,
    image_size: tuple[int, int],
    batch_size: int,
    class_names: list[str],
    shuffle: bool,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Load an image dataset from a directory structure.

    Args:
        directory: Path to the split directory (train/val/test),
            containing one subdirectory per class.
        image_size: Target (height, width) for image resizing.
        batch_size: Number of samples per batch.
        class_names: Explicit ordered list of class names. Ensures
            consistent label indices across train/val/test.
        shuffle: Whether to shuffle. Should be True only for training.
        seed: Random seed used when shuffling is enabled.

    Returns:
        A batched tf.data.Dataset of images and one-hot labels.
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def build_datasets(
    paths: ProjectPaths,
    class_names: list[str],
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train/val/test datasets with prefetching enabled.

    The dataset is pre-split on disk, so no splitting happens here.

    Note:
        The test set is loaded for pipeline consistency, but must NOT
        be evaluated until the final official test evaluation. Do not
        use it for training, tuning, or model selection.

    Returns:
        (train_ds, val_ds, test_ds)
    """
    train_ds = load_dataset(
        paths.train_dir, image_size, batch_size, class_names,
        shuffle=True, seed=seed,
    )
    val_ds = load_dataset(
        paths.val_dir, image_size, batch_size, class_names, shuffle=False,
    )
    test_ds = load_dataset(
        paths.test_dir, image_size, batch_size, class_names, shuffle=False,
    )

    autotune = tf.data.AUTOTUNE

    return (
        train_ds.prefetch(buffer_size=autotune),
        val_ds.prefetch(buffer_size=autotune),
        test_ds.prefetch(buffer_size=autotune),
    )


# --------------------------------------------
# Class Imbalance
# --------------------------------------------

def count_samples_per_class(
    directory: Path,
    class_names: list[str],
) -> dict[str, int]:
    """Count image files for each class in a dataset split."""
    class_counts = {}

    for class_name in class_names:
        class_dir = directory / class_name
        class_counts[class_name] = sum(
            1
            for file_path in class_dir.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )

    return class_counts


def analyze_class_imbalance(
    class_counts: dict[str, int],
    class_names: list[str],
    threshold: float = 1.50,
) -> dict:
    """Measure class imbalance and pre-compute class weights.

    Class weights use total_samples / (num_classes * class_count), keyed
    by class index as expected by `model.fit(..., class_weight=...)`.
    They are computed for later phases and are not applied automatically.

    Args:
        class_counts: Samples per class name (training split).
        class_names: Ordered class names.
        threshold: Max acceptable largest/smallest class ratio.

    Returns:
        Dict with min_count, max_count, imbalance_ratio, strategy
        ("none" or "review_required") and class_weights.
    """
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())

    if min_count == 0:
        raise ValueError("At least one class contains zero training samples.")

    ratio = max_count / min_count
    total_samples = sum(class_counts.values())
    num_classes = len(class_names)

    return {
        "min_count": min_count,
        "max_count": max_count,
        "imbalance_ratio": ratio,
        "threshold": threshold,
        "strategy": "review_required" if ratio > threshold else "none",
        "class_weights": {
            index: total_samples / (num_classes * class_counts[name])
            for index, name in enumerate(class_names)
        },
    }


# --------------------------------------------
# Dataset Sanity Checks
# --------------------------------------------

def validate_dataset_batch(
    dataset: tf.data.Dataset,
    dataset_name: str,
    image_size: tuple[int, int],
    image_channels: int,
    num_classes: int,
) -> dict:
    """Run sanity checks on the first batch of a dataset.

    Returns:
        Dict with the checked batch's shapes, dtype and value range.
    """
    expected_image_shape = (*image_size, image_channels)

    for images, labels in dataset.take(1):

        if images.shape.rank != 4:
            raise ValueError(
                f"{dataset_name}: Expected image tensor with rank 4, "
                f"got shape {images.shape}."
            )

        if tuple(images.shape[1:]) != expected_image_shape:
            raise ValueError(
                f"{dataset_name}: Expected image shape "
                f"(batch, {expected_image_shape[0]}, "
                f"{expected_image_shape[1]}, {expected_image_shape[2]}), "
                f"got {images.shape}."
            )

        if labels.shape.rank != 2:
            raise ValueError(
                f"{dataset_name}: Expected one-hot labels with rank 2, "
                f"got shape {labels.shape}."
            )

        if labels.shape[-1] != num_classes:
            raise ValueError(
                f"{dataset_name}: Expected {num_classes} label columns, "
                f"got {labels.shape[-1]}."
            )

        # image_dataset_from_directory returns float32 in the raw
        # [0, 255] range; normalization happens inside the model.
        if images.dtype != tf.float32:
            raise TypeError(
                f"{dataset_name}: Expected images with dtype float32 "
                f"(raw [0, 255] range), got {images.dtype}."
            )

        image_min = tf.reduce_min(images).numpy()
        image_max = tf.reduce_max(images).numpy()

        if image_min < 0 or image_max > 255:
            raise ValueError(
                f"{dataset_name}: Image values must be in [0, 255], "
                f"got range [{image_min}, {image_max}]."
            )

        if not tf.reduce_all(tf.math.is_finite(images)):
            raise ValueError(f"{dataset_name}: Images contain NaN or Inf values.")

        if not tf.reduce_all(tf.math.is_finite(labels)):
            raise ValueError(f"{dataset_name}: Labels contain NaN or Inf values.")

        label_sums = tf.reduce_sum(labels, axis=1)

        if not tf.reduce_all(tf.equal(label_sums, 1.0)):
            raise ValueError(f"{dataset_name}: Labels are not valid one-hot vectors.")

        if not tf.reduce_all(
            tf.logical_or(tf.equal(labels, 0.0), tf.equal(labels, 1.0))
        ):
            raise ValueError(
                f"{dataset_name}: One-hot labels must contain only 0 or 1."
            )

        return {
            "name": dataset_name,
            "images_shape": tuple(images.shape),
            "labels_shape": tuple(labels.shape),
            "dtype": images.dtype.name,
            "range": (float(image_min), float(image_max)),
        }

    raise ValueError(f"{dataset_name}: Dataset is empty.")


def validate_datasets(
    datasets: dict[str, tf.data.Dataset],
    image_size: tuple[int, int],
    image_channels: int,
    num_classes: int,
) -> list[dict]:
    """Validate several named datasets (batch checks + cardinality).

    Args:
        datasets: Mapping of display name -> dataset.

    Returns:
        One summary dict per dataset (see validate_dataset_batch).
    """
    summaries = [
        validate_dataset_batch(
            dataset, name, image_size, image_channels, num_classes
        )
        for name, dataset in datasets.items()
    ]

    for name, dataset in datasets.items():
        if dataset.cardinality().numpy() <= 0:
            raise ValueError(f"{name}: Dataset contains no batches.")

    return summaries