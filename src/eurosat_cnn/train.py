"""Training utilities: callbacks and learning-rate schedules.

Learning-rate strategies (pick ONE per training run)
----------------------------------------------------
`build_reduce_lr_on_plateau` and `build_cosine_lr_schedule` are
ALTERNATIVE learning-rate strategies, not complementary ones. Do not
use both in the same training run.

ReduceLROnPlateau mutates the optimizer's learning rate in place, which
only works when learning_rate is a plain scalar/variable. If the
optimizer's learning_rate is a LearningRateSchedule (e.g. CosineDecay),
the callback has no effect (or errors, depending on the TF version),
because the schedule computes the rate internally from the step count.

    - ReduceLROnPlateau -> pass a plain float as learning_rate
    - CosineDecay       -> pass the schedule as learning_rate, and do
                           NOT add ReduceLROnPlateau to the callbacks
"""

from pathlib import Path

from tensorflow import keras


# --------------------------------------------
# Early Stopping
# --------------------------------------------

def build_early_stopping(
    monitor: str = "val_loss",
    min_delta: float = 1e-4,
    patience: int = 5,
    mode: str = "min",
    restore_best_weights: bool = True,
) -> keras.callbacks.EarlyStopping:
    """Create an EarlyStopping callback.

    Args:
        monitor: Metric to monitor.
        min_delta: Minimum absolute change that counts as an improvement.
        patience: Epochs with no improvement before stopping.
        mode: "min" or "max", direction of improvement for `monitor`.
        restore_best_weights: Whether to restore the weights from the
            epoch with the best value of `monitor`.

    Returns:
        A configured EarlyStopping callback.
    """
    return keras.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        mode=mode,
        restore_best_weights=restore_best_weights,
        verbose=1,
    )


# --------------------------------------------
# Model Checkpoint
# --------------------------------------------

def build_model_checkpoint(
    filepath: str | Path,
    monitor: str = "val_loss",
    mode: str = "min",
    save_best_only: bool = True,
    save_weights_only: bool = False,
) -> keras.callbacks.ModelCheckpoint:
    """Create a ModelCheckpoint callback.

    Args:
        filepath: Path to save the checkpoint to. The parent directory
            is created automatically if it doesn't exist. With Keras 3
            the name must end in ".keras" (full model) or
            ".weights.h5" (weights only).
        monitor: Metric to monitor.
        mode: "min" or "max", direction of improvement for `monitor`.
        save_best_only: Whether to keep only the best checkpoint
            according to `monitor`.
        save_weights_only: Whether to save only the model weights
            instead of the full model.

    Returns:
        A configured ModelCheckpoint callback.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    return keras.callbacks.ModelCheckpoint(
        filepath=str(filepath),
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        verbose=1,
    )


# --------------------------------------------
# Reduce Learning Rate on Plateau
# --------------------------------------------

def build_reduce_lr_on_plateau(
    monitor: str = "val_loss",
    factor: float = 0.5,
    patience: int = 2,
    min_lr: float = 1e-6,
    mode: str = "min",
    min_delta: float = 1e-4,
) -> keras.callbacks.ReduceLROnPlateau:
    """Create a ReduceLROnPlateau callback.

    Requires the optimizer's learning_rate to be a plain scalar, not a
    LearningRateSchedule (see the module docstring).

    Args:
        monitor: Metric to monitor.
        factor: Factor by which the learning rate is reduced
            (new_lr = lr * factor).
        patience: Epochs with no improvement before reducing the rate.
        min_lr: Lower bound on the learning rate.
        mode: "min" or "max", direction of improvement for `monitor`.
        min_delta: Minimum change that counts as an improvement.

    Returns:
        A configured ReduceLROnPlateau callback.
    """
    return keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        mode=mode,
        min_delta=min_delta,
        verbose=1,
    )


# --------------------------------------------
# Cosine Learning Rate Schedule
# --------------------------------------------

def build_cosine_lr_schedule(
    initial_learning_rate: float,
    decay_steps: int,
    alpha: float = 0.0,
) -> keras.optimizers.schedules.CosineDecay:
    """Create a CosineDecay learning-rate schedule.

    Pass the returned schedule directly as the optimizer's
    learning_rate. Do not combine with ReduceLROnPlateau (see the
    module docstring).

    Args:
        initial_learning_rate: Initial learning rate.
        decay_steps: Number of optimizer steps over which the learning
            rate decays.
        alpha: Minimum learning-rate ratio relative to the initial
            learning rate.

    Returns:
        A CosineDecay learning-rate schedule.
    """
    if decay_steps <= 0:
        raise ValueError("decay_steps must be greater than zero.")

    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha,
    )