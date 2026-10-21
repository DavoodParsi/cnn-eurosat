"""Experiment tracking utilities for Weights & Biases (W&B).

Authentication is never handled in code. Log in once with `wandb login`
or set the WANDB_API_KEY environment variable.

Settings are resolved at call time, in this order:
    explicit argument > environment variable > default
    - WANDB_ENTITY: W&B username/team (default: the logged-in account)
    - WANDB_MODE:   "online" (default) or "offline"
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from tensorflow import keras

# Main W&B project for the CNN experiments. High-FLOPs CNN, Low-FLOPs
# CNN and MobileNetV2 experiments are organized inside this project.
DEFAULT_WANDB_PROJECT = "cnn-classification"


# --------------------------------------------
# Settings
# --------------------------------------------

def get_wandb_entity(entity: str | None = None) -> str | None:
    """Resolve the W&B entity: argument > WANDB_ENTITY env var > None.

    None lets W&B use the entity of the logged-in account.
    """
    return entity or os.getenv("WANDB_ENTITY") or None


def get_wandb_mode(mode: str | None = None) -> str:
    """Resolve the W&B mode: argument > WANDB_MODE env var > "online".

    "online" logs to W&B; "offline" stores logs locally for later sync.
    """
    return mode or os.getenv("WANDB_MODE", "online")


# --------------------------------------------
# Shared Internal Helpers
# --------------------------------------------

def _require_active_run(run: "wandb.Run", action: str) -> None:
    """Raise ValueError if no W&B Run was provided.

    Args:
        run: Run object to validate.
        action: Short description of the attempted action, used in the
            error message (e.g. "log metrics").
    """
    if run is None:
        raise ValueError(
            f"A valid, active W&B Run is required to {action}."
        )


def _require_nonempty_name(name: str, param_name: str) -> str:
    """Strip and validate a non-empty string parameter."""
    name = name.strip()

    if not name:
        raise ValueError(f"{param_name} must be a non-empty string.")

    return name


def _filter_numeric_metrics(
    metrics: dict[str, Any],
    *,
    namespace: str | None = None,
) -> dict[str, float]:
    """Filter a metrics dict down to scalar numeric values.

    Non-numeric values are silently skipped.

    Args:
        metrics: Metric names and values.
        namespace: Optional prefix, e.g. namespace="test" turns
            "accuracy" into "test/accuracy".

    Returns:
        Numeric values cast to float, with keys optionally namespaced.
    """
    filtered: dict[str, float] = {}

    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            metric_key = f"{namespace}/{key}" if namespace else str(key)
            filtered[metric_key] = float(value)

    return filtered


def _flatten_nested_metrics(
    metrics: dict[str, Any],
    *,
    namespace: str | None = None,
) -> dict[str, float]:
    """Flatten nested metric dicts into scalar W&B metrics.

    Example: {"accuracy": 0.95, "macro": {"f1_score": 0.95}} becomes
    {"accuracy": 0.95, "macro/f1_score": 0.95}. Only numeric leaf
    values are kept.

    Args:
        metrics: Metric dict that may contain nested dicts.
        namespace: Optional top-level prefix for every key.
    """
    flat: dict[str, Any] = {}

    def _flatten(current: dict[str, Any], prefix: str = "") -> None:
        for key, value in current.items():
            current_key = f"{prefix}/{key}" if prefix else str(key)

            if isinstance(value, dict):
                _flatten(value, prefix=current_key)
            else:
                flat[current_key] = value

    _flatten(metrics)

    return _filter_numeric_metrics(flat, namespace=namespace)


def _split_train_val_metrics(
    logs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a Keras logs dict into (train_metrics, val_metrics).

    Validation metrics are prefixed with "val_" in Keras; the prefix is
    stripped from the returned validation keys.
    """
    train_metrics = {
        key: value for key, value in logs.items()
        if not key.startswith("val_")
    }

    val_metrics = {
        key[4:]: value for key, value in logs.items()
        if key.startswith("val_")
    }

    return train_metrics, val_metrics


def _build_epoch_payload(logs: dict[str, Any], step: int) -> dict[str, float]:
    """Build the W&B payload for one epoch (train/, val/ and epoch)."""
    train_logs, val_logs = _split_train_val_metrics(logs)

    return {
        **_filter_numeric_metrics(train_logs, namespace="train"),
        **_filter_numeric_metrics(val_logs, namespace="val"),
        "epoch": step,
    }


def _normalize_experiment_identity(
    model_type: str,
    phase: str,
    experiment_type: str,
) -> tuple[str, str, str]:
    """Normalize (strip + lowercase) and validate the experiment identity.

    Raises:
        ValueError: If any field is empty after stripping.
    """
    model_type = model_type.strip().lower()
    phase = phase.strip().lower()
    experiment_type = experiment_type.strip().lower()

    if not model_type or not phase or not experiment_type:
        raise ValueError(
            "model_type, phase and experiment_type must be non-empty."
        )

    return model_type, phase, experiment_type


def _normalize_split(split: str) -> str:
    """Normalize and validate a dataset split name ("test", ...)."""
    split = split.strip().lower()

    if not split:
        raise ValueError("split must be a non-empty string.")

    return split


def _require_artifact(artifact: "wandb.Artifact") -> None:
    """Raise ValueError if no W&B Artifact was provided."""
    if artifact is None:
        raise ValueError("A valid W&B artifact is required.")


# --------------------------------------------
# Run Management
# --------------------------------------------

def build_wandb_run_name(
    model_type: str,
    experiment_type: str,
    run_index: int | None = None,
) -> str:
    """Build a consistent W&B run name, e.g. "high-baseline".

    Args:
        model_type: Short model identifier ("high", "low", ...).
        experiment_type: "baseline", "retrain", "hpt", "final", ...
        run_index: Optional index for repeated HPT/sweep runs,
            producing e.g. "high-hpt-007".
    """
    model_type = model_type.strip().lower()
    experiment_type = experiment_type.strip().lower()

    if not model_type or not experiment_type:
        raise ValueError("model_type and experiment_type must be non-empty.")

    if run_index is not None:
        return f"{model_type}-{experiment_type}-{run_index:03d}"

    return f"{model_type}-{experiment_type}"


def initialize_wandb_run(
    *,
    model_type: str,
    phase: str,
    experiment_type: str,
    config: dict[str, Any] | None = None,
    run_name: str | None = None,
    run_index: int | None = None,
    extra_tags: Sequence[str] | None = None,
    notes: str | None = None,
    run_id: str | None = None,
    project: str | None = None,
    entity: str | None = None,
    mode: str | None = None,
) -> "wandb.Run":
    """Initialize a W&B run, or resume an existing one.

    Centralizes run creation so all experiments follow one structure.

    Args:
        model_type: Short model identifier (see build_wandb_run_name).
        phase: Project phase ("phase-1", "phase-2", "final").
        experiment_type: Type of experiment.
        config: Hyperparameters, typically from build_wandb_config().
            Ignored when run_id is given.
        run_name: Custom run name; generated if omitted. Ignored when
            run_id is given.
        run_index: Numeric run index, mainly for HPT runs.
        extra_tags: Additional tags. Ignored when run_id is given.
        notes: Human-readable description. Ignored when run_id is given.
        run_id: Existing W&B run ID. When provided, that run is resumed
            instead of creating a new one, so later logging
            (evaluation, error analysis) attaches to the original run
            after an interruption such as a Colab runtime reset.
        project: W&B project (default: DEFAULT_WANDB_PROJECT).
        entity: W&B entity (default: see get_wandb_entity).
        mode: "online" or "offline" (default: see get_wandb_mode).

    Returns:
        The active W&B Run.

    Notes:
        reinit=True lets a notebook re-run the initialization cell
        without manually finishing the previous run. It does not apply
        to the resume path, which always reconnects to run_id.
    """
    project = project or DEFAULT_WANDB_PROJECT
    entity = get_wandb_entity(entity)
    mode = get_wandb_mode(mode)

    if run_id is not None:
        return wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            resume="must",
            mode=mode,
        )

    if run_name is None:
        run_name = build_wandb_run_name(
            model_type=model_type,
            experiment_type=experiment_type,
            run_index=run_index,
        )

    tags = build_wandb_tags(
        model_type=model_type,
        phase=phase,
        experiment_type=experiment_type,
        extra_tags=extra_tags,
    )

    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,
        notes=notes,
        config=config,
        mode=mode,
        reinit=True,
    )


def finish_wandb_run(run: "wandb.Run | None" = None) -> None:
    """Finish the W&B run cleanly.

    Passing the run explicitly keeps this usable in both notebook and
    script workflows; without it the active global run is finished.
    """
    if run is not None:
        run.finish()
    else:
        wandb.finish()


def save_wandb_run_state(run: "wandb.Run", path: str | Path) -> None:
    """Save the run's identity (id/name/project/entity) as local JSON.

    Only the identity is saved, not config or metrics: those already
    live on W&B and are reloaded when the run is resumed via its id.

    Args:
        run: Active W&B Run.
        path: Destination JSON file (parent folder is created).
    """
    _require_active_run(run, "save W&B run state")

    state = {
        "run_id": run.id,
        "run_name": run.name,
        "project": run.project,
        "entity": run.entity,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4)


def load_wandb_run_state(path: str | Path) -> dict[str, str]:
    """Load a previously saved W&B run identity from disk.

    Returns:
        The stored identity, containing at least "run_id".

    Raises:
        FileNotFoundError: If the state file does not exist.
        ValueError: If the file has no valid run_id.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"W&B run state file was not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        state = json.load(file)

    if not state.get("run_id"):
        raise ValueError("W&B run state does not contain a valid run_id.")

    return state


# --------------------------------------------
# Experiment Configuration
# --------------------------------------------

def build_wandb_config(
    *,
    model_name: str,
    model_type: str,
    image_size: tuple[int, int],
    image_channels: int,
    num_classes: int,
    batch_size: int,
    epochs: int,
    optimizer_name: str,
    learning_rate: float,
    loss_name: str,
    augmentation_config: dict[str, Any] | None = None,
    class_imbalance_config: dict[str, Any] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standardized configuration dictionary for W&B.

    Args:
        model_name: Human-readable name, e.g. "High-FLOPs Custom CNN".
        model_type: Short identifier ("high", "low", "mobilenetv2").
        image_size: Input spatial size, e.g. (64, 64).
        image_channels: Number of input channels.
        num_classes: Number of target classes.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        optimizer_name: e.g. "Adam".
        learning_rate: Initial learning rate.
        loss_name: Loss function name.
        augmentation_config: Data augmentation settings.
        class_imbalance_config: Class imbalance handling settings.
        extra_config: Additional experiment-specific settings.
    """
    config: dict[str, Any] = {
        "model_name": model_name,
        "model_type": model_type,
        "image_size": list(image_size),
        "image_channels": image_channels,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "loss_name": loss_name,
    }

    if augmentation_config is not None:
        config["augmentation"] = augmentation_config

    if class_imbalance_config is not None:
        config["class_imbalance"] = class_imbalance_config

    if extra_config is not None:
        config.update(extra_config)

    return config


def update_wandb_config(
    config: dict[str, Any],
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of config with updates applied.

    The original dict is not modified. The update is shallow: nested
    values are replaced as a whole, not recursively merged.
    """
    updated_config = config.copy()

    if updates is not None:
        updated_config.update(updates)

    return updated_config


# --------------------------------------------
# Tags & Run Metadata
# --------------------------------------------

def build_wandb_tags(
    model_type: str,
    phase: str,
    experiment_type: str,
    extra_tags: Sequence[str] | None = None,
) -> list[str]:
    """Build a standardized, deduplicated list of W&B tags.

    Core tags (model type, phase, experiment type) come first, followed
    by optional custom tags. All tags are stripped and lowercased.

    Raises:
        ValueError: If a core field is empty after stripping.
    """
    model_type, phase, experiment_type = _normalize_experiment_identity(
        model_type, phase, experiment_type
    )

    tags = [model_type, phase, experiment_type]

    if extra_tags is not None:
        tags.extend(
            tag.strip().lower()
            for tag in extra_tags
            if isinstance(tag, str) and tag.strip()
        )

    # Preserve insertion order while removing duplicates.
    return list(dict.fromkeys(tags))


def build_wandb_run_metadata(
    *,
    model_type: str,
    phase: str,
    experiment_type: str,
    run_index: int | None = None,
) -> dict[str, Any]:
    """Build standardized metadata that identifies a W&B run.

    These values describe the identity/context of the experiment and
    are kept separate from performance metrics.

    Raises:
        ValueError: If a core field is empty after stripping.
    """
    model_type, phase, experiment_type = _normalize_experiment_identity(
        model_type, phase, experiment_type
    )

    metadata: dict[str, Any] = {
        "model_type": model_type,
        "experiment_phase": phase,
        "experiment_type": experiment_type,
    }

    if run_index is not None:
        metadata["run_index"] = int(run_index)

    return metadata


# --------------------------------------------
# Training Metric Logging
# --------------------------------------------

class WandBEpochLogger(keras.callbacks.Callback):
    """Keras callback that logs train/val metrics to W&B every epoch.

    This is the primary metric-logging path (as opposed to logging only
    after model.fit() completes), so that:
      - a crash/disconnect mid-training does not lose the metrics
        logged so far, and
      - W&B Sweep early termination (Hyperband, in Phase 2) can observe
        metrics epoch by epoch.

    Args:
        run: Active W&B Run to log into.
    """

    def __init__(self, run: "wandb.Run"):
        super().__init__()
        _require_active_run(run, "attach WandBEpochLogger")
        self.run = run

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        if not logs:
            return

        step = epoch + 1

        self.run.log(_build_epoch_payload(logs, step), step=step)


def log_wandb_metrics(
    run: "wandb.Run",
    metrics: dict[str, Any],
    *,
    step: int | None = None,
    namespace: str | None = None,
) -> None:
    """Log arbitrary scalar metrics to W&B.

    Non-numeric values are silently skipped.

    Args:
        run: Active W&B Run.
        metrics: Metric names and values.
        step: Optional explicit W&B step.
        namespace: Optional prefix, e.g. "test" -> "test/accuracy".
    """
    _require_active_run(run, "log metrics")

    logged_metrics = _filter_numeric_metrics(metrics, namespace=namespace)

    if not logged_metrics:
        return

    run.log(logged_metrics, step=step)


def get_training_history_dict(history) -> dict[str, list]:
    """Convert a Keras History object into a plain dict of lists."""
    if history is None:
        raise ValueError("A valid Keras History object is required.")

    return {key: list(values) for key, values in history.history.items()}


def log_wandb_training_history(run: "wandb.Run", history) -> None:
    """Log a complete Keras training history to W&B, epoch by epoch.

    FALLBACK PATH ONLY. The primary mechanism is WandBEpochLogger,
    attached as a callback during model.fit(), which logs in real time
    and survives mid-training crashes. Use this function only when
    training ran without that callback and only a finished `history`
    is available (e.g. an offline script or a debugging session).
    """
    _require_active_run(run, "log training history")

    history_dict = get_training_history_dict(history)

    if not history_dict:
        return

    num_epochs = len(next(iter(history_dict.values())))

    for epoch in range(num_epochs):
        epoch_values = {
            key: values[epoch]
            for key, values in history_dict.items()
            if epoch < len(values)
        }

        step = epoch + 1

        run.log(_build_epoch_payload(epoch_values, step), step=step)


def log_wandb_training_summary(run: "wandb.Run", history) -> None:
    """Log final and best training metrics to the W&B Run Summary.

    For each tracked metric, records final_<metric>. For performance
    metrics (everything except learning_rate/lr) also records
    best_<metric>: the minimum for metrics containing "loss", the
    maximum otherwise.

    This assumes higher-is-better for every non-loss performance metric
    (accuracy, precision, recall, F1). learning_rate/lr are excluded
    from "best_" because they are a changing hyperparameter, not a
    performance metric (with ReduceLROnPlateau the rate only stays flat
    or decreases, so "max" would trivially be the initial value).
    """
    _require_active_run(run, "log training summary")

    history_dict = get_training_history_dict(history)

    if not history_dict:
        return

    non_performance_metrics = {"learning_rate", "lr"}

    for metric_name, values in history_dict.items():
        numeric_values = [
            float(v)
            for v in values
            if isinstance(v, (int, float, np.integer, np.floating))
        ]

        if not numeric_values:
            continue

        if metric_name.startswith("val_"):
            metric_key = metric_name[4:]
            namespace = "val"
        else:
            metric_key = metric_name
            namespace = "train"

        run.summary[f"{namespace}/final_{metric_key}"] = numeric_values[-1]

        if metric_key.lower() in non_performance_metrics:
            continue

        best_value = (
            min(numeric_values)
            if "loss" in metric_key.lower()
            else max(numeric_values)
        )

        run.summary[f"{namespace}/best_{metric_key}"] = best_value


# --------------------------------------------
# Run Summary Management
# --------------------------------------------

def log_wandb_parameter_summary(
    run: "wandb.Run",
    parameter_stats: dict[str, Any],
) -> None:
    """Log parameter statistics to the W&B Run Summary.

    Args:
        run: Active W&B Run.
        parameter_stats: Output of reporting.get_model_parameter_stats.
    """
    _require_active_run(run, "log parameter statistics")

    summary_mapping = {
        "model/total_params": "total_params",
        "model/trainable_params": "trainable_params",
        "model/non_trainable_params": "non_trainable_params",
    }

    for wandb_key, stats_key in summary_mapping.items():
        if stats_key in parameter_stats:
            run.summary[wandb_key] = int(parameter_stats[stats_key])


def log_wandb_complexity_summary(
    run: "wandb.Run",
    complexity_stats: dict[str, Any],
) -> None:
    """Log model complexity statistics (e.g. FLOPs) to the Run Summary.

    Only scalar numeric values are added.
    """
    _require_active_run(run, "log complexity statistics")

    for key, value in _filter_numeric_metrics(complexity_stats).items():
        run.summary[f"model/{key}"] = value


# --------------------------------------------
# Model Metadata Logging
# --------------------------------------------

def build_wandb_model_metadata(
    model: keras.Model,
    *,
    model_name: str,
    model_type: str,
    input_shape: tuple | None = None,
    num_classes: int | None = None,
    parameter_stats: dict[str, Any] | None = None,
    complexity_stats: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build standardized model metadata for W&B.

    Args:
        model: Built Keras model.
        model_name: Human-readable model name.
        model_type: Short identifier ("high", "low", "mobilenetv2").
        input_shape: Model input shape, e.g. (64, 64, 3).
        num_classes: Number of output classes.
        parameter_stats: From reporting.get_model_parameter_stats.
        complexity_stats: Model complexity information (e.g. FLOPs).
        extra_metadata: Additional model metadata.
    """
    metadata: dict[str, Any] = {
        "model/name": model_name,
        "model/type": model_type,
        "model/architecture": model.__class__.__name__,
        "model/layers": len(model.layers),
    }

    if input_shape is not None:
        metadata["model/input_shape"] = list(input_shape)

    if num_classes is not None:
        metadata["model/num_classes"] = int(num_classes)

    if parameter_stats is not None:
        for key in (
            "total_params",
            "trainable_params",
            "non_trainable_params",
        ):
            if key in parameter_stats:
                metadata[f"model/{key}"] = int(parameter_stats[key])

    if complexity_stats is not None:
        for key, value in complexity_stats.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metadata[f"model/{key}"] = float(value)
            else:
                metadata[f"model/{key}"] = value

    if extra_metadata is not None:
        metadata.update(extra_metadata)

    return metadata


def log_wandb_model_metadata(
    run: "wandb.Run",
    metadata: dict[str, Any],
) -> None:
    """Log model metadata (from build_wandb_model_metadata) to the run config."""
    _require_active_run(run, "log model metadata")
    run.config.update(metadata)


# --------------------------------------------
# Evaluation Logging
# --------------------------------------------

def log_wandb_evaluation_metrics(
    run: "wandb.Run",
    metrics: dict[str, Any],
    *,
    split: str = "test",
    step: int | None = None,
) -> None:
    """Log evaluation metrics to W&B, namespaced by dataset split.

    Nested metric dicts are flattened, e.g. with split="validation":
    {"accuracy": 0.95, "macro": {"f1_score": 0.95}} becomes
    "validation/accuracy" and "validation/macro/f1_score".

    Args:
        run: Active W&B Run.
        metrics: Output of evaluate.calculate_all_classification_metrics.
        split: Dataset split ("train", "validation", "test").
        step: Optional explicit W&B step.
    """
    _require_active_run(run, "log evaluation metrics")

    split = _normalize_split(split)

    logged_metrics = _flatten_nested_metrics(metrics, namespace=split)

    if not logged_metrics:
        return

    run.log(logged_metrics, step=step)


def log_wandb_evaluation_summary(
    run: "wandb.Run",
    metrics: dict[str, Any],
    *,
    split: str = "test",
) -> None:
    """Log final evaluation metrics to the W&B Run Summary.

    Same flattening and "<split>/..." naming as
    log_wandb_evaluation_metrics.
    """
    _require_active_run(run, "log evaluation summary")

    split = _normalize_split(split)

    for key, value in _flatten_nested_metrics(
        metrics, namespace=split
    ).items():
        run.summary[key] = value


# --------------------------------------------
# Error Analysis Logging
# --------------------------------------------

def log_wandb_error_analysis_metrics(
    run: "wandb.Run",
    metrics: dict[str, Any],
) -> None:
    """Log error analysis metrics to W&B under "error_analysis/".

    Nested dicts (e.g. the per-class breakdown) are flattened, so
    {"per_class": {"Forest": {"errors": 12}}} becomes
    "error_analysis/per_class/Forest/errors".

    Args:
        run: Active W&B Run.
        metrics: Output of error_analysis.calculate_error_statistics.
    """
    _require_active_run(run, "log error analysis metrics")

    logged_metrics = _flatten_nested_metrics(
        metrics, namespace="error_analysis"
    )

    if not logged_metrics:
        return

    run.log(logged_metrics)


def log_wandb_error_analysis_summary(
    run: "wandb.Run",
    metrics: dict[str, Any],
) -> None:
    """Log error analysis metrics to the Run Summary ("error_analysis/...")."""
    _require_active_run(run, "log error analysis summary")

    for key, value in _flatten_nested_metrics(
        metrics, namespace="error_analysis"
    ).items():
        run.summary[key] = value


# --------------------------------------------
# Tables & Media Logging
# --------------------------------------------

def log_wandb_table(
    run: "wandb.Run",
    table_name: str,
    data: Any | None = None,
    *,
    columns: Sequence[str] | None = None,
    rows: Sequence[Sequence[Any]] | None = None,
) -> None:
    """Log a W&B Table to the active run.

    Supports an existing wandb.Table (pass it as `data`) or creation of
    a new table from `columns` and `rows`.

    Raises:
        ValueError: If table_name is empty, or neither a wandb.Table
            nor columns were provided.
    """
    _require_active_run(run, "log W&B table")

    table_name = _require_nonempty_name(table_name, "table_name")

    if isinstance(data, wandb.Table):
        table = data

    elif columns is not None:
        table = wandb.Table(columns=list(columns))

        if rows is not None:
            for row in rows:
                table.add_data(*row)

    else:
        raise ValueError(
            "Provide an existing wandb.Table or "
            "columns to create a new table."
        )

    run.log({table_name: table})


def log_wandb_media(
    run: "wandb.Run",
    media_name: str,
    media: Any,
    *,
    step: int | None = None,
) -> None:
    """Log a W&B media object (Image, Video, Audio, Html, ...).

    Raises:
        ValueError: If media_name is empty.
    """
    _require_active_run(run, "log W&B media")

    media_name = _require_nonempty_name(media_name, "media_name")

    run.log({media_name: media}, step=step)


def log_wandb_media_collection(
    run: "wandb.Run",
    media_name: str,
    media_items: Sequence[Any],
    *,
    step: int | None = None,
) -> None:
    """Log a collection of W&B media objects under one name.

    Raises:
        ValueError: If media_name is empty or media_items is empty.
    """
    _require_active_run(run, "log W&B media collection")

    media_name = _require_nonempty_name(media_name, "media_name")

    if not media_items:
        raise ValueError("media_items must contain at least one item.")

    run.log({media_name: list(media_items)}, step=step)


# --------------------------------------------
# Reports
# --------------------------------------------

def log_wandb_report_link(
    run: "wandb.Run",
    report_name: str,
    report_url: str,
) -> None:
    """Store a W&B report link in the Run Summary ("reports/<name>").

    Raises:
        ValueError: If report_name or report_url is empty.
    """
    _require_active_run(run, "log W&B report link")

    report_name = _require_nonempty_name(report_name, "report_name")
    report_url = _require_nonempty_name(report_url, "report_url")

    run.summary[f"reports/{report_name}"] = report_url


# --------------------------------------------
# Artifact Management
# --------------------------------------------

def build_wandb_artifact(
    artifact_name: str,
    *,
    artifact_type: str,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> "wandb.Artifact":
    """Build a W&B Artifact object.

    Args:
        artifact_name: Name identifying the artifact.
        artifact_type: "model", "dataset", "checkpoint", "config", ...
        description: Human-readable description.
        metadata: Additional artifact metadata.
    """
    artifact_name = _require_nonempty_name(artifact_name, "artifact_name")
    artifact_type = _require_nonempty_name(artifact_type, "artifact_type")

    return wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description,
        metadata=metadata,
    )


def add_wandb_artifact_file(
    artifact: "wandb.Artifact",
    file_path: str | Path,
    *,
    name: str | None = None,
) -> None:
    """Add a local file to a W&B Artifact.

    Args:
        artifact: Artifact from build_wandb_artifact().
        file_path: Local path to the file.
        name: Optional filename inside the artifact.
    """
    _require_artifact(artifact)

    file_path = _require_nonempty_name(str(file_path), "file_path")

    artifact.add_file(local_path=file_path, name=name)


def add_wandb_artifact_directory(
    artifact: "wandb.Artifact",
    directory_path: str | Path,
    *,
    name: str | None = None,
) -> None:
    """Add a local directory to a W&B Artifact.

    Args:
        artifact: Artifact from build_wandb_artifact().
        directory_path: Local directory path.
        name: Optional directory name inside the artifact.
    """
    _require_artifact(artifact)

    directory_path = _require_nonempty_name(
        str(directory_path), "directory_path"
    )

    artifact.add_dir(local_path=directory_path, name=name)


def log_wandb_artifact(
    run: "wandb.Run",
    artifact: "wandb.Artifact",
    *,
    aliases: Sequence[str] | None = None,
) -> None:
    """Log a W&B Artifact to the active run.

    Args:
        run: Active W&B Run.
        artifact: Artifact to log.
        aliases: Optional version aliases, e.g. ["baseline", "best"].
    """
    _require_active_run(run, "log W&B artifact")
    _require_artifact(artifact)

    run.log_artifact(
        artifact,
        aliases=list(aliases) if aliases is not None else None,
    )


# --------------------------------------------
# Model Handoff
# --------------------------------------------

def log_wandb_model_artifact(
    run: "wandb.Run",
    model_path: str | Path,
    *,
    artifact_name: str,
    artifact_type: str = "model",
    aliases: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Package a saved model as a W&B Artifact and log it to the run.

    This is the handoff path from a local saved model to a versioned
    W&B model artifact.

    Args:
        run: Active W&B Run.
        model_path: Local path to the saved model file or directory.
        artifact_name: Name of the model artifact.
        artifact_type: W&B artifact type (default "model").
        aliases: Optional version aliases ("latest", "baseline", ...).
        metadata: Additional model artifact metadata.

    Raises:
        ValueError: If the run, model path, name or type is invalid.
    """
    _require_active_run(run, "log model artifact")

    model_path = Path(_require_nonempty_name(str(model_path), "model_path"))

    if not model_path.exists():
        raise ValueError(f"model_path does not exist: {model_path}")

    artifact = build_wandb_artifact(
        artifact_name=artifact_name,
        artifact_type=artifact_type,
        metadata=metadata,
    )

    if model_path.is_dir():
        add_wandb_artifact_directory(artifact, model_path)
    else:
        add_wandb_artifact_file(artifact, model_path)

    log_wandb_artifact(run, artifact, aliases=aliases)


# --------------------------------------------
# Sweep / HPT Support
# --------------------------------------------

def build_wandb_sweep_config(
    *,
    method: str,
    metric_name: str,
    metric_goal: str,
    parameters: dict[str, Any],
    project: str | None = None,
) -> dict[str, Any]:
    """Build a standardized W&B Sweep configuration.

    Args:
        method: Search method: "grid", "random" or "bayes".
        metric_name: Metric monitored by the sweep, e.g. "val/loss".
        metric_goal: "minimize" or "maximize".
        parameters: W&B Sweep parameter definitions.
        project: W&B project (default: DEFAULT_WANDB_PROJECT).

    Raises:
        ValueError: If a required value is empty or invalid.
    """
    method = _require_nonempty_name(method, "method").lower()
    metric_name = _require_nonempty_name(metric_name, "metric_name")
    metric_goal = _require_nonempty_name(metric_goal, "metric_goal").lower()

    if method not in {"grid", "random", "bayes"}:
        raise ValueError("method must be one of: grid, random, bayes.")

    if metric_goal not in {"minimize", "maximize"}:
        raise ValueError(
            "metric_goal must be either 'minimize' or 'maximize'."
        )

    if not parameters:
        raise ValueError(
            "parameters must contain at least one sweep parameter."
        )

    sweep_project = (
        DEFAULT_WANDB_PROJECT
        if project is None
        else _require_nonempty_name(project, "project")
    )

    return {
        "method": method,
        "metric": {"name": metric_name, "goal": metric_goal},
        "parameters": parameters,
        "project": sweep_project,
    }


def create_wandb_sweep(
    sweep_config: dict[str, Any],
    *,
    project: str | None = None,
    entity: str | None = None,
) -> str:
    """Create a W&B Sweep and return its Sweep ID.

    Args:
        sweep_config: Config from build_wandb_sweep_config().
        project: W&B project (default: the one in sweep_config, else
            DEFAULT_WANDB_PROJECT).
        entity: W&B entity (default: see get_wandb_entity).

    Raises:
        ValueError: If sweep_config is empty.
    """
    if not sweep_config:
        raise ValueError("sweep_config must not be empty.")

    sweep_project = (
        project
        if project is not None
        else sweep_config.get("project", DEFAULT_WANDB_PROJECT)
    )
    sweep_project = _require_nonempty_name(sweep_project, "project")

    sweep_config = sweep_config.copy()
    sweep_config.pop("project", None)

    return wandb.sweep(
        sweep=sweep_config,
        entity=get_wandb_entity(entity),
        project=sweep_project,
    )