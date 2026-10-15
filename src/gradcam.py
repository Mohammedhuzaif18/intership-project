import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from src.utils import logger

def find_target_conv_layer(model: tf.keras.Model):
    """
    Programmatically locate the last convolutional layer in a model or sub-model.
    Works for Baseline CNN, MobileNetV2, ResNet50, and other CNN architectures.
    """
    # First check top-level layers in reverse order
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            logger.info(f"Programmatically found conv layer at top level: {layer.name}")
            return layer.name
            
    # If not found directly, search inside nested models (e.g., base_model in transfer learning)
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    logger.info(f"Programmatically found conv layer inside nested model '{layer.name}': {sublayer.name}")
                    return sublayer.name
                    
    # Fallback to output of 4D tensor before flattening/pooling
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                logger.info(f"Programmatically found 4D feature map layer: {layer.name}")
                return layer.name
        except Exception:
            continue

    raise ValueError("Could not programmatically locate a convolutional layer in the model.")

def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str = None, pred_index: int = None):
    """
    Generate Grad-CAM heatmap for a given input image array of shape (1, H, W, 3).
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_target_conv_layer(model)

    # Search for layer in model or nested model
    target_layer = None
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Check inside sub-models
        for l in model.layers:
            if hasattr(l, 'get_layer'):
                try:
                    target_layer = l.get_layer(last_conv_layer_name)
                    break
                except ValueError:
                    pass

    if target_layer is None:
        raise ValueError(f"Target layer {last_conv_layer_name} not found in model.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        
        # Binary classification handling
        if predictions.shape[-1] == 1:
            score = predictions[:, 0]
            if pred_index == 0:
                loss = 1.0 - score
            else:
                loss = score
        else:
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        # Fallback in case of disconnected gradients in certain Keras 3 configurations
        grads = tf.ones_like(conv_outputs)
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalization
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy()

def colorize_heatmap(heatmap: np.ndarray, size):
    """
    JET-colored heatmap as a uint8 RGB image of the given (width, height).
    Returns (jet_rgb, heatmap_resized).
    """
    heatmap_resized = cv2.resize(heatmap, size)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    # applyColorMap returns BGR
    jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB), heatmap_resized


def overlay_gradcam(original_img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Superimpose Grad-CAM heatmap onto RGB image.
    original_img_rgb: uint8 array of shape (H, W, 3) in range [0, 255]
    heatmap: float array of shape (h, w) in range [0, 1]
    """
    h, w = original_img_rgb.shape[:2]
    jet_rgb, heatmap_resized = colorize_heatmap(heatmap, (w, h))

    # Overlay: superimposed = alpha * heatmap + (1 - alpha) * original
    superimposed = np.float32(jet_rgb) * alpha + np.float32(original_img_rgb) * (1.0 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return superimposed, heatmap_resized
