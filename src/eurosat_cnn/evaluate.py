"""Evaluation metrics: calculation and persistence."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# --------------------------------------------
# Calculate Classification Metrics
# --------------------------------------------

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> dict:
    """Calculate precision, recall, and F1-score with an explicit
    averaging strategy.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        average: Averaging strategy. Supported values:
            "macro" or "weighted".

    Returns:
        Dict with precision, recall, and f1_score.
    """
    if average not in {"macro", "weighted"}:
        raise ValueError("average must be either 'macro' or 'weighted'.")

    return {
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        ),
    }


# --------------------------------------------
# Calculate All Classification Metrics
# --------------------------------------------

def calculate_all_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate accuracy plus macro and weighted precision/recall/F1.

    Reuses calculate_classification_metrics() as the single source of
    truth for precision/recall/f1, rather than recomputing them.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.

    Returns:
        Dict with "accuracy", "macro", and "weighted" keys.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": calculate_classification_metrics(
            y_true, y_pred, average="macro"
        ),
        "weighted": calculate_classification_metrics(
            y_true, y_pred, average="weighted"
        ),
    }


# --------------------------------------------
# Save Evaluation Metrics
# --------------------------------------------

def save_evaluation_metrics(
    metrics: dict,
    *,
    model_name: str,
    save_dir: str | Path,
    phase: str = "phase-1",
    experiment_type: str = "baseline",
    dataset_name: str = "validation",
) -> str:
    """Save evaluation metrics as a JSON file.

    Args:
        metrics: Dict of evaluation metrics to save.
        model_name: Identifier used in the filename.
        save_dir: Directory to save the file into (created if missing).
        phase: Pipeline phase (e.g. "phase-1", "phase-2").
        experiment_type: Experiment type (e.g. "baseline", "retrain",
            "hpt"). Included in the filename so different experiment
            types under the same phase never overwrite each other.
        dataset_name: Evaluated split (e.g. "validation", "test").

    Returns:
        Path to the saved JSON file.
    """
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)

    save_path = directory / (
        f"{model_name}_{phase}_{experiment_type}_{dataset_name}_metrics.json"
    )

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    return str(save_path)