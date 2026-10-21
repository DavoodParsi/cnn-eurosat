"""Model building blocks and architectures."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------
# Data Augmentation
# --------------------------------------------

def build_augmentation(
    flip_mode: str = "horizontal_and_vertical",
    rotation_factor: float = 1.0,
    zoom_factor: float = 0.2,
) -> keras.Sequential:
    """Create the training-time data augmentation pipeline.

    Used as the first layers of the model, together with Rescaling.
    Keras augmentation layers are active only during training and
    inactive during evaluation/inference, so no manual filtering is
    needed.

    The defaults are intentionally aggressive because satellite imagery
    has no canonical orientation: any rotation or flip direction still
    produces a realistic sample.

    Args:
        flip_mode: Mode of the RandomFlip layer.
        rotation_factor: Factor of the RandomRotation layer.
        zoom_factor: Factor of the RandomZoom layer.

    Returns:
        A Sequential model named "data_augmentation".
    """
    return keras.Sequential(
        [
            layers.RandomFlip(flip_mode),
            layers.RandomRotation(rotation_factor),
            layers.RandomZoom(zoom_factor),
        ],
        name="data_augmentation",
    )


# --------------------------------------------
# Residual Block
# --------------------------------------------

def residual_block(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    name: str | None = None,
) -> tf.Tensor:
    """Apply a standard two-layer residual block (ResNet v1 style).

    Main path: Conv -> BN -> ReLU -> Conv -> BN
    Shortcut path: identity, or a 1x1 projection Conv -> BN when the
    spatial resolution changes (stride != 1) or the channel count
    changes (shortcut channels != filters).
    The two paths are summed and passed through a final ReLU.

    Args:
        x: Input feature map.
        filters: Number of output filters for both conv layers.
        stride: Stride applied to the first conv (and the projection
            shortcut, if used). Use stride=2 to downsample.
        name: Optional prefix for naming sub-layers; if None, layers
            get auto-generated Keras names.

    Returns:
        Output feature map after the residual addition and final ReLU.
    """
    shortcut = x

    # Main path
    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv1",
    )(x)

    x = layers.BatchNormalization(
        name=None if name is None else f"{name}_bn1",
    )(x)

    x = layers.ReLU(
        name=None if name is None else f"{name}_relu1",
    )(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv2",
    )(x)

    x = layers.BatchNormalization(
        name=None if name is None else f"{name}_bn2",
    )(x)

    # Projection shortcut for shape matching.
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=None if name is None else f"{name}_projection_conv",
        )(shortcut)

        shortcut = layers.BatchNormalization(
            name=None if name is None else f"{name}_projection_bn",
        )(shortcut)

    # Residual connection
    x = layers.Add(
        name=None if name is None else f"{name}_add",
    )([x, shortcut])

    x = layers.ReLU(
        name=None if name is None else f"{name}_output_relu",
    )(x)

    return x


# --------------------------------------------
# Main Architecture
# --------------------------------------------

def build_custom_cnn_high_flops(
    input_shape: tuple[int, int, int],
    num_classes: int,
    augmentation: keras.layers.Layer,
    dropout_rate: float = 0.4,
) -> keras.Model:
    """Build the high-FLOPs custom residual CNN.

    Architecture: stem conv -> 3 residual stages (64/128/256 filters,
    2 blocks each) with strided-conv downsampling between stages ->
    global average pooling -> dense classifier head.

    Args:
        input_shape: Shape of a single input image, e.g. (64, 64, 3).
        num_classes: Number of output classes.
        augmentation: A Keras layer (or Sequential of layers) applying
            training-time data augmentation; should be a no-op at
            inference time (e.g. built from `layers.RandomFlip`, etc.).
        dropout_rate: Dropout rate used in the classifier head.

    Returns:
        An uncompiled `keras.Model` mapping raw images in [0, 255]
        to class probabilities.
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # Apply training-time data augmentation.
    x = augmentation(inputs)
    # Normalize input pixel values to the [0, 1] range.
    x = layers.Rescaling(
        scale=1.0 / 255,
        name="rescaling",
    )(x)

    # Initial feature extraction
    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Residual Stage 1
    x = residual_block(x, filters=64, name="stage1_block1")
    x = residual_block(x, filters=64, name="stage1_block2")

    # Downsample to 32x32 and increase channel capacity.
    x = layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="downsample1_conv",
    )(x)
    x = layers.BatchNormalization(name="downsample1_bn")(x)
    x = layers.ReLU(name="downsample1_relu")(x)

    # Residual Stage 2
    x = residual_block(x, filters=128, name="stage2_block1")
    x = residual_block(x, filters=128, name="stage2_block2")

    # Downsample to 16x16 and increase channel capacity.
    x = layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="downsample2_conv",
    )(x)
    x = layers.BatchNormalization(name="downsample2_bn")(x)
    x = layers.ReLU(name="downsample2_relu")(x)

    # Residual Stage 3
    x = residual_block(x, filters=256, name="stage3_block1")
    x = residual_block(x, filters=256, name="stage3_block2")

    # Global feature aggregation
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    # Classification head
    x = layers.Dense(
        256,
        activation="relu",
        name="classifier_dense",
    )(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)

    # Keep classifier outputs in float32 under mixed precision.
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="custom_cnn_high_flops",
    )