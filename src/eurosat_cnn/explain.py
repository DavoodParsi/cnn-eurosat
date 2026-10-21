"""Model interpretability: Grad-CAM."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def generate_grad_cam(
    model: keras.Model,
    image: np.ndarray,
    target_class: int,
    layer_name: str,
    output_layer_name: str = "predictions",
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a target class.

    Gradients are computed with respect to the pre-softmax logit of the
    target class, which avoids the class coupling introduced by softmax
    normalization.

    Compatible with mixed-precision models: the classifier input is
    explicitly cast to the dtype of the final Dense layer kernel before
    the manual logits computation.

    Args:
        model: Trained classification model.
        image: Input image in raw pixel-value format, shape (H, W, C)
            (or (B, H, W, C)).
        target_class: Class index used as the Grad-CAM target.
        layer_name: Name of the convolutional layer used for Grad-CAM.
        output_layer_name: Name of the final Dense classification layer.

    Returns:
        Normalized Grad-CAM heatmap with values in [0, 1].
    """
    target_layer = model.get_layer(layer_name)

    if len(target_layer.output.shape) != 4:
        raise ValueError(
            f"Target layer '{layer_name}' must produce a 4D feature map."
        )

    output_layer = model.get_layer(output_layer_name)

    if not isinstance(output_layer, layers.Dense):
        raise ValueError(
            f"The final '{output_layer_name}' layer must be a Dense layer."
        )

    if not 0 <= target_class < output_layer.units:
        raise ValueError(
            f"target_class must be in the range "
            f"[0, {output_layer.units - 1}]."
        )

    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, output_layer.input],
    )

    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    if len(image_tensor.shape) == 3:
        image_tensor = tf.expand_dims(image_tensor, axis=0)

    if len(image_tensor.shape) != 4:
        raise ValueError("Image must have shape (H, W, C) or (B, H, W, C).")

    with tf.GradientTape() as tape:
        conv_outputs, classifier_input = grad_model(
            image_tensor,
            training=False,
        )

        # Mixed-precision compatibility:
        # under mixed_float16, classifier_input is float16 while
        # output_layer.kernel is float32, and MatMul needs compatible
        # dtypes. This cast does NOT modify the model weights or the
        # model's mixed-precision configuration.
        classifier_input = tf.cast(
            classifier_input,
            output_layer.kernel.dtype,
        )

        logits = tf.matmul(classifier_input, output_layer.kernel)

        if output_layer.bias is not None:
            logits = logits + tf.cast(output_layer.bias, logits.dtype)

        class_score = logits[:, target_class]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise ValueError(
            f"Gradients could not be computed for target layer "
            f"'{layer_name}'."
        )

    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        tf.zeros_like(heatmap),
    )

    return heatmap.numpy()