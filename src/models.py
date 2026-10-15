"""Model definitions. Every model takes 224x224x3 float RGB in [0, 255] and normalizes internally."""
import tensorflow as tf

from src.config import IMAGE_SIZE
from src.data.loaders import get_augmentation

L = tf.keras.layers
INPUT_SHAPE = (*IMAGE_SIZE, 3)


def _head(x, num_classes: int):
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dense(128, activation="relu", name="fc1")(x)
    x = L.Dropout(0.4, name="dropout")(x)
    if num_classes <= 2:
        return L.Dense(1, activation="sigmoid", name="predictions")(x)
    return L.Dense(num_classes, activation="softmax", name="predictions")(x)


def build_baseline_cnn(num_classes: int = 2) -> tf.keras.Model:
    """Four conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool), 32 to 256 filters."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")
    x = get_augmentation()(inputs)
    x = L.Rescaling(1.0 / 255.0, name="rescale")(x)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = L.Conv2D(filters, 3, padding="same", use_bias=False, name=f"block{i}_conv")(x)
        x = L.BatchNormalization(name=f"block{i}_bn")(x)
        x = L.Activation("relu", name=f"block{i}_out")(x)
        x = L.MaxPooling2D(2, name=f"block{i}_pool")(x)
    return tf.keras.Model(inputs, _head(x, num_classes), name="Baseline_CNN")


def build_transfer_model(backbone: str = "resnet50", num_classes: int = 2):
    """ImageNet backbone (frozen) + new head. Returns (model, backbone_model)."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")
    x = get_augmentation()(inputs)
    if backbone == "mobilenet_v2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        base = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_tensor=x)
        name = "MobileNetV2_transfer"
    elif backbone == "resnet50":
        x = tf.keras.applications.resnet50.preprocess_input(x)
        base = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=x)
        name = "ResNet50_transfer"
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    base.trainable = False
    return tf.keras.Model(inputs, _head(base.output, num_classes), name=name), base


def unfreeze_top(base: tf.keras.Model, n_layers: int = 20) -> None:
    """Make the last n backbone layers trainable. BatchNorm stays frozen (inference mode),
    because re-estimating ImageNet statistics from small batches tends to hurt fine-tuning."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, L.BatchNormalization):
            layer.trainable = False


def compile_model(model: tf.keras.Model, num_classes: int, learning_rate: float) -> tf.keras.Model:
    binary = num_classes <= 2
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy" if binary else "categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=False)],
    )
    return model
