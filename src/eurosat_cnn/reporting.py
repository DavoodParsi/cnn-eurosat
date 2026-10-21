"""Model reporting: parameter statistics, configuration and JSON reports."""

import json
from pathlib import Path

import numpy as np
from tensorflow import keras


# --------------------------------------------
# Model Parameter Statistics
# --------------------------------------------

def get_model_parameter_stats(model: keras.Model) -> dict:
    """Calculate model parameter statistics.

    Args:
        model: A built (or compiled) Keras model.

    Returns:
        Dict with total, trainable, and non-trainable parameter counts.
    """
    trainable_params = int(
        sum(np.prod(w.shape) for w in model.trainable_weights)
    )

    non_trainable_params = int(
        sum(np.prod(w.shape) for w in model.non_trainable_weights)
    )

    return {
        "total_params": trainable_params + non_trainable_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


# --------------------------------------------
# Model Configuration
# --------------------------------------------

def get_model_config(model: keras.Model) -> dict:
    """Extract basic model configuration information.

    Args:
        model: A built (or compiled) Keras model.

    Returns:
        Dict with model name, input/output shape, and layer count.
    """
    return {
        "name": model.name,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "num_layers": len(model.layers),
    }


# --------------------------------------------
# Build Model Report
# --------------------------------------------

def build_model_report(
    model: keras.Model,
    *,
    model_name: str,
    phase: str = "phase1",
) -> dict:
    """Build a structured model report.

    Args:
        model: A built (or compiled) Keras model.
        model_name: Identifier used in the saved report.
        phase: Pipeline phase this report corresponds to
            (e.g. "phase1", "phase2").

    Returns:
        Dict combining model configuration and parameter statistics.
    """
    return {
        "model_name": model_name,
        "phase": phase,
        "model": get_model_config(model),
        "parameters": get_model_parameter_stats(model),
    }


# --------------------------------------------
# Save Model Report
# --------------------------------------------

def save_model_report(
    report: dict,
    *,
    save_dir: str | Path,
) -> str:
    """Save a structured model report as JSON.

    model_name and phase are read from the report itself to keep the
    saved filename consistent with the report.

    Args:
        report: Dict produced by `build_model_report`.
        save_dir: Directory to save the report into (created if missing).

    Returns:
        Path to the saved JSON report.
    """
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)

    save_path = directory / (
        f"{report['model_name']}_{report['phase']}_model_report.json"
    )

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return str(save_path)