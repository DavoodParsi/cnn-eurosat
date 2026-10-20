"""Reproducibility and hardware/environment setup."""

import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision


def set_seed(seed: int, full_determinism: bool = False) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Seed value for Python, NumPy and TensorFlow.
        full_determinism: If True, enforce bit-exact deterministic ops.
            This can slow down training, especially on GPU.
    """
    keras.utils.set_random_seed(seed)

    if full_determinism:
        tf.config.experimental.enable_op_determinism()


def configure_hardware(
    enable_mixed_precision: bool = True,
    mixed_precision_policy: str = "mixed_float16",
) -> dict:
    """Detect GPUs and set the global mixed-precision policy.

    Mixed precision is used only when a GPU is available. On CPU the
    policy falls back to float32, because mixed_float16 is not
    beneficial there.

    Args:
        enable_mixed_precision: Whether to allow mixed precision.
        mixed_precision_policy: Policy name used when enabled.

    Returns:
        Dictionary describing the detected hardware and active policy.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if enable_mixed_precision and gpus:
        mixed_precision.set_global_policy(mixed_precision_policy)
    else:
        mixed_precision.set_global_policy("float32")

    return {
        "gpus": [str(gpu) for gpu in gpus],
        "gpu_count": len(gpus),
        "mixed_precision_enabled": enable_mixed_precision,
        "mixed_precision_policy": mixed_precision.global_policy().name,
    }


def print_environment_summary(
    seed: int,
    full_determinism: bool,
    hardware: dict,
) -> None:
    """Print a snapshot of the software and hardware environment."""
    print("GPU Information:")
    if hardware["gpu_count"]:
        print(f"Number of GPUs: {hardware['gpu_count']}")
        for gpu in hardware["gpus"]:
            print(f" - {gpu}")
    else:
        print("No GPU detected. Training will use CPU.")

    print("\nEnvironment Snapshot:")
    print(f"Python Version         : {sys.version.split()[0]}")
    print(f"TensorFlow Version     : {tf.__version__}")
    print(f"Keras Version          : {keras.__version__}")
    print(f"NumPy Version          : {np.__version__}")
    print(f"Random Seed            : {seed}")
    print(f"Full Determinism       : {full_determinism}")
    print(f"GPU Available          : {hardware['gpu_count'] > 0}")
    print(f"GPU Count              : {hardware['gpu_count']}")
    print(f"Mixed Precision        : {hardware['mixed_precision_enabled']}")
    print(f"Mixed Precision Policy : {hardware['mixed_precision_policy']}")