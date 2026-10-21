"""Error analysis: prediction confidence, error extraction and statistics."""

import numpy as np
import pandas as pd


# --------------------------------------------
# Calculate Prediction Confidence
# --------------------------------------------

def calculate_prediction_confidence(y_prob: np.ndarray) -> np.ndarray:
    """Calculate the confidence of each prediction.

    Confidence is defined as the maximum predicted class probability.

    Args:
        y_prob: Predicted class probabilities, shape (N, num_classes).

    Returns:
        Maximum class probability for each sample, shape (N,).
    """
    return np.max(y_prob, axis=1)


# --------------------------------------------
# Extract Error Samples
# --------------------------------------------

def extract_error_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract incorrect predictions and their confidence scores.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        y_prob: Predicted class probabilities.

    Returns:
        Tuple of (error_indices, true_labels, predicted_labels,
        confidences) for the incorrectly classified samples.
    """
    error_indices = np.where(y_true != y_pred)[0]

    true_labels = y_true[error_indices]
    predicted_labels = y_pred[error_indices]
    confidences = calculate_prediction_confidence(y_prob[error_indices])

    return error_indices, true_labels, predicted_labels, confidences


# --------------------------------------------
# Extract High-Confidence Errors
# --------------------------------------------

def extract_high_confidence_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract errors made with confidence >= threshold.

    Builds on extract_error_samples() and applies a confidence filter.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        y_prob: Predicted class probabilities.
        threshold: Minimum confidence for an error to be kept.

    Returns:
        Tuple of (indices, true_labels, predicted_labels, confidences)
        for the high-confidence errors.
    """
    (
        error_indices,
        true_labels,
        predicted_labels,
        confidences,
    ) = extract_error_samples(y_true, y_pred, y_prob)

    high_conf_mask = confidences >= threshold

    return (
        error_indices[high_conf_mask],
        true_labels[high_conf_mask],
        predicted_labels[high_conf_mask],
        confidences[high_conf_mask],
    )


# --------------------------------------------
# Calculate Error Statistics
# --------------------------------------------

def calculate_error_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict:
    """Calculate overall and per-class error statistics.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        y_prob: Predicted class probabilities.
        class_names: Class names indexed by class ID.

    Returns:
        Dict with overall statistics and a "per_class" sub-dict.
    """
    confidences = calculate_prediction_confidence(y_prob)

    total_samples = len(y_true)
    error_mask = y_true != y_pred
    total_errors = int(np.sum(error_mask))

    overall_error_rate = (
        total_errors / total_samples if total_samples > 0 else 0.0
    )

    mean_confidence = (
        float(np.mean(confidences)) if total_samples > 0 else 0.0
    )

    mean_error_confidence = (
        float(np.mean(confidences[error_mask])) if total_errors > 0 else 0.0
    )

    statistics = {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "overall_error_rate": float(overall_error_rate),
        "mean_confidence": mean_confidence,
        "mean_error_confidence": mean_error_confidence,
        "per_class": {},
    }

    for class_id, class_name in enumerate(class_names):
        class_mask = y_true == class_id
        class_total = int(np.sum(class_mask))
        class_errors = int(np.sum(class_mask & error_mask))

        class_error_rate = (
            class_errors / class_total if class_total > 0 else 0.0
        )

        statistics["per_class"][class_name] = {
            "total_samples": class_total,
            "errors": class_errors,
            "error_rate": float(class_error_rate),
        }

    return statistics


# --------------------------------------------
# Calculate High-Confidence Error Distribution
# --------------------------------------------

def calculate_high_confidence_error_distribution(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Group high-confidence errors by True Class -> Predicted Class.

    Assumes the input arrays already contain only high-confidence
    errors. No confidence thresholding or error filtering happens here.

    Args:
        true_labels: Ground-truth class indices of the errors.
        predicted_labels: Predicted class indices of the errors.
        confidences: Prediction confidence of the errors.
        class_names: Class names indexed by class ID.

    Returns:
        DataFrame with columns true_class, predicted_class,
        error_count, mean_confidence, max_confidence, min_confidence,
        sorted by error_count then mean_confidence (descending).
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    confidences = np.asarray(confidences)

    if not (len(true_labels) == len(predicted_labels) == len(confidences)):
        raise ValueError(
            "true_labels, predicted_labels, and "
            "confidences must have the same length."
        )

    if len(class_names) == 0:
        raise ValueError("class_names must contain at least one class.")

    if len(true_labels) == 0:
        return pd.DataFrame(
            columns=[
                "true_class",
                "predicted_class",
                "error_count",
                "mean_confidence",
                "max_confidence",
                "min_confidence",
            ]
        )

    class_name_array = np.asarray(class_names, dtype=object)

    error_data = pd.DataFrame(
        {
            "true_class": class_name_array[true_labels],
            "predicted_class": class_name_array[predicted_labels],
            "confidence": confidences,
        }
    )

    distribution = error_data.groupby(
        ["true_class", "predicted_class"],
        as_index=False,
    ).agg(
        error_count=("confidence", "size"),
        mean_confidence=("confidence", "mean"),
        max_confidence=("confidence", "max"),
        min_confidence=("confidence", "min"),
    )

    return distribution.sort_values(
        by=["error_count", "mean_confidence"],
        ascending=[False, False],
    ).reset_index(drop=True)