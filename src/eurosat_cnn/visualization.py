"""Plotting and reporting utilities.

Covers prediction extraction, training curves, confusion matrices,
classification reports, error-sample grids and Grad-CAM figures.

Every plotting function takes `model_name` and `save_dir` explicitly
(keyword-only). Output filenames follow the pattern
`{model_name}_{phase}_{experiment_type}_..._{what}.png`, so different
experiments under the same phase never overwrite each other.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras


# --------------------------------------------
# Internal Helpers
# --------------------------------------------

def _output_path(save_dir: str | Path, filename: str) -> Path:
    """Return save_dir / filename, creating save_dir if needed."""
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def _save_and_close(fig: plt.Figure, save_path: Path, show: bool) -> str:
    """Save a figure at 300 dpi, optionally show it, then close it."""
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return str(save_path)


# --------------------------------------------
# Prediction Extraction
# --------------------------------------------

def extract_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract true labels, predicted labels, and probabilities.

    Runs a single batched model.predict() call over the full dataset
    instead of looping batch-by-batch.

    Warning:
        The dataset is iterated twice (predictions, then labels), so it
        must NOT be shuffled (use the validation/test datasets).

    Args:
        model: A trained Keras model.
        dataset: A batched, non-shuffled tf.data.Dataset of
            (images, labels).

    Returns:
        Tuple of (y_true, y_pred, y_prob): ground-truth class indices,
        predicted class indices, and predicted class probabilities.
    """
    y_prob = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    y_true = []

    for _, labels in dataset:
        if labels.ndim > 1:
            labels = np.argmax(labels.numpy(), axis=1)
        else:
            labels = labels.numpy()

        y_true.extend(labels)

    y_true = np.asarray(y_true)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Mismatch between ground-truth labels and predictions."
        )

    return y_true, y_pred, y_prob


# --------------------------------------------
# Training History
# --------------------------------------------

def plot_training_history(
    history: keras.callbacks.History,
    *,
    model_name: str,
    save_dir: str | Path,
    phase: str = "phase-1",
    experiment_type: str = "baseline",
    show: bool = True,
) -> str:
    """Plot and save training/validation accuracy and loss.

    Args:
        history: History object returned by model.fit().
        model_name: Identifier used in the plot title and filename.
        save_dir: Directory to save the plot into.
        phase: Pipeline phase (e.g. "phase-1", "phase-2").
        experiment_type: Experiment type (e.g. "baseline", "retrain",
            "hpt"), included in the filename.
        show: Whether to display the figure before closing it.

    Returns:
        Path to the saved PNG plot.
    """
    acc = history.history.get("accuracy")
    val_acc = history.history.get("val_accuracy")
    loss = history.history.get("loss")
    val_loss = history.history.get("val_loss")

    if acc is None or val_acc is None:
        raise ValueError("Accuracy history was not found.")

    if loss is None or val_loss is None:
        raise ValueError("Loss history was not found.")

    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axes[0].plot(epochs, acc, label="Training Accuracy")
    axes[0].plot(epochs, val_acc, label="Validation Accuracy")
    axes[0].set_title(f"{model_name} - {phase} - {experiment_type} Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(epochs, loss, label="Training Loss")
    axes[1].plot(epochs, val_loss, label="Validation Loss")
    axes[1].set_title(f"{model_name} - {phase} - {experiment_type} Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    save_path = _output_path(
        save_dir,
        f"{model_name}_{phase}_{experiment_type}_training_history.png",
    )

    return _save_and_close(fig, save_path, show)


# --------------------------------------------
# Confusion Matrix
# --------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    *,
    model_name: str,
    save_dir: str | Path,
    phase: str = "phase-1",
    experiment_type: str = "baseline",
    dataset_name: str = "validation",
    show: bool = True,
) -> str:
    """Plot and save a confusion matrix.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Display names for each class index.
        model_name: Identifier used in the plot title and filename.
        save_dir: Directory to save the plot into.
        phase: Pipeline phase this matrix corresponds to.
        experiment_type: Experiment type, included in the filename.
        dataset_name: Evaluated split (e.g. "validation", "test").
        show: Whether to display the figure before closing it.

    Returns:
        Path to the saved PNG plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    disp.plot(cmap="Blues", values_format="d", ax=ax)

    ax.set_title(
        f"{model_name} - {phase} - {experiment_type} - "
        f"{dataset_name} Confusion Matrix"
    )

    fig.tight_layout()

    save_path = _output_path(
        save_dir,
        f"{model_name}_{phase}_{experiment_type}_"
        f"{dataset_name}_confusion_matrix.png",
    )

    return _save_and_close(fig, save_path, show)


# --------------------------------------------
# Classification Report
# --------------------------------------------

def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    *,
    model_name: str,
    save_dir: str | Path,
    phase: str = "phase-1",
    experiment_type: str = "baseline",
    dataset_name: str = "validation",
    verbose: bool = True,
) -> tuple[str, dict]:
    """Generate, optionally print, and save a classification report.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Display names for each class index.
        model_name: Identifier used in the printed header and filename.
        save_dir: Directory to save the report into.
        phase: Pipeline phase this report corresponds to.
        experiment_type: Experiment type, included in the filename.
        dataset_name: Evaluated split (e.g. "validation", "test").
        verbose: Whether to print the report.

    Returns:
        Tuple of (save_path, report_dict). report_dict is the report as
        a dict (output_dict=True), for cross-trial metric comparison.
    """
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )

    if verbose:
        print(
            f"{model_name} - {phase} - {experiment_type} - "
            f"{dataset_name} Classification Report:\n"
        )
        print(report_str)

    save_path = _output_path(
        save_dir,
        f"{model_name}_{phase}_{experiment_type}_"
        f"{dataset_name}_classification_report.txt",
    )

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(report_str)

    return str(save_path), report_dict


# --------------------------------------------
# Plot Error Samples
# --------------------------------------------

def plot_error_samples(
    file_paths: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    *,
    model_name: str,
    save_dir: str | Path,
    phase: str = "phase-1",
    experiment_type: str = "baseline",
    max_samples: int = 16,
    show: bool = True,
) -> str:
    """Plot and save incorrectly classified samples.

    Samples are sorted by confidence in descending order, so the most
    confident errors are displayed first.

    Args:
        file_paths: Path to each sample's source image file.
        true_labels: Ground-truth class index for each sample.
        predicted_labels: Predicted class index for each sample.
        confidences: Model confidence (max predicted probability).
        class_names: Display names for each class index.
        model_name: Identifier used in the plot title and filename.
        save_dir: Directory to save the plot into.
        phase: Pipeline phase this plot corresponds to.
        experiment_type: Experiment type, included in the filename.
        max_samples: Maximum number of samples to plot.
        show: Whether to display the figure before closing it.

    Returns:
        Path to the saved PNG plot.
    """
    if not (
        len(file_paths)
        == len(true_labels)
        == len(predicted_labels)
        == len(confidences)
    ):
        raise ValueError("Error sample arrays have inconsistent lengths.")

    if len(file_paths) == 0:
        raise ValueError("No error samples were provided.")

    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero.")

    num_samples = min(max_samples, len(file_paths))

    sorted_indices = np.argsort(confidences)[::-1]
    selected_indices = sorted_indices[:num_samples]

    num_columns = min(4, num_samples)
    num_rows = int(np.ceil(num_samples / num_columns))

    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(4 * num_columns, 4 * num_rows),
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()

    for axis, index in zip(axes, selected_indices):
        image_path = file_paths[index]

        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Error image was not found: {image_path}")

        image = plt.imread(image_path)

        true_label = class_names[true_labels[index]]
        predicted_label = class_names[predicted_labels[index]]
        confidence = confidences[index]

        axis.imshow(image)
        axis.set_title(
            f"True: {true_label}\n"
            f"Pred: {predicted_label}\n"
            f"Confidence: {confidence:.4f}"
        )
        axis.axis("off")

    for axis in axes[num_samples:]:
        axis.axis("off")

    fig.suptitle(
        f"{model_name} - {phase} - {experiment_type} Error Analysis",
        fontsize=14,
    )

    save_path = _output_path(
        save_dir,
        f"{model_name}_{phase}_{experiment_type}_error_analysis.png",
    )

    return _save_and_close(fig, save_path, show)


# --------------------------------------------
# Plot Grad-CAM
# --------------------------------------------

def plot_grad_cam(
    image: np.ndarray,
    true_heatmap: np.ndarray,
    predicted_heatmap: np.ndarray,
    true_label: str,
    predicted_label: str,
    confidence: float,
    *,
    model_name: str,
    save_dir: str | Path,
    phase: str = "phase-1",
    experiment_type: str = "baseline",
    sample_index: int | None = None,
    show: bool = True,
) -> str:
    """Plot and save a Grad-CAM visualization.

    Shows the original image next to the Grad-CAM overlay for the true
    class and for the predicted class: where the model should have
    looked versus where it actually looked. Both heatmaps are resized
    to the original image's dimensions.

    Args:
        image: Original image in raw pixel values, shape (H, W, C).
        true_heatmap: Grad-CAM heatmap for the true class
            (from generate_grad_cam).
        predicted_heatmap: Grad-CAM heatmap for the predicted class
            (from generate_grad_cam).
        true_label: Ground-truth class name.
        predicted_label: Predicted class name.
        confidence: Model confidence, within [0, 1].
        model_name: Identifier used in the plot title and filename.
        save_dir: Directory to save the plot into.
        phase: Pipeline phase this plot corresponds to.
        experiment_type: Experiment type, included in the filename.
        sample_index: Optional sample identifier, appended to the title
            and filename to tell multiple plots apart.
        show: Whether to display the figure before closing it.

    Returns:
        Path to the saved PNG plot.
    """
    if image.ndim != 3:
        raise ValueError("Image must have shape (H, W, C).")

    if true_heatmap.ndim != 2 or predicted_heatmap.ndim != 2:
        raise ValueError("Grad-CAM heatmaps must have shape (H, W).")

    if true_heatmap.shape != predicted_heatmap.shape:
        raise ValueError("Grad-CAM heatmaps must have the same shape.")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be within the [0, 1] range.")

    display_image = np.clip(image, 0, 255).astype(np.uint8)

    # Both heatmaps come from the same target layer for the same
    # image, so one resize target covers both.
    target_size = display_image.shape[:2]

    resized_true_heatmap = tf.image.resize(
        true_heatmap[..., np.newaxis],
        size=target_size,
        method="bilinear",
    ).numpy()[..., 0]

    resized_predicted_heatmap = tf.image.resize(
        predicted_heatmap[..., np.newaxis],
        size=target_size,
        method="bilinear",
    ).numpy()[..., 0]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        constrained_layout=True,
    )

    axes[0].imshow(display_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(display_image)
    axes[1].imshow(resized_true_heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title(f"Grad-CAM: True\n({true_label})")
    axes[1].axis("off")

    axes[2].imshow(display_image)
    axes[2].imshow(resized_predicted_heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title(
        f"Grad-CAM: Predicted\n({predicted_label})\n"
        f"Confidence: {confidence:.4f}"
    )
    axes[2].axis("off")

    title = f"{model_name} - {phase} - {experiment_type} - Grad-CAM"

    if sample_index is not None:
        title += f" - Sample {sample_index}"

    fig.suptitle(title, fontsize=14)

    sample_suffix = (
        f"_sample_{sample_index}" if sample_index is not None else ""
    )

    save_path = _output_path(
        save_dir,
        f"{model_name}_{phase}_{experiment_type}_grad_cam{sample_suffix}.png",
    )

    return _save_and_close(fig, save_path, show)