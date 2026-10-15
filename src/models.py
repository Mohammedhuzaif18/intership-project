import tensorflow as tf
from src.dataset import get_augmentation_layer


def _prediction_head(feat, num_classes: int):
    feat = tf.keras.layers.GlobalAveragePooling2D(name="gap")(feat)
    feat = tf.keras.layers.Dense(128, activation="relu", name="fc1")(feat)
    feat = tf.keras.layers.Dropout(0.4, name="dropout")(feat)
    if num_classes <= 2:
        return tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(feat)
    return tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(feat)


def compile_model(model, num_classes: int, learning_rate: float = 1e-3):
    loss = "binary_crossentropy" if num_classes <= 2 else "categorical_crossentropy"
    metrics = ["accuracy"]
    if num_classes <= 2:
        metrics.append(tf.keras.metrics.AUC(name="auc"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model


def build_baseline_cnn(input_shape=(224, 224, 3), include_augmentation=True, num_classes=2):
    """Custom baseline CNN with 4 convolutional blocks."""
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    x = inputs
    if include_augmentation:
        x = get_augmentation_layer()(x)

    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_out")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    outputs = _prediction_head(x, num_classes)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Baseline_CNN")
    return model


def build_transfer_model(
    model_name="resnet50",
    input_shape=(224, 224, 3),
    include_augmentation=True,
    num_classes=2,
):
    """
    Transfer learning with ImageNet weights.
    Default backbone is ResNet50 as required by the internship spec.
    MobileNetV2 remains available as a lighter CPU option.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    x = inputs
    if include_augmentation:
        x = get_augmentation_layer()(x)

    name = model_name.lower()
    if name == "mobilenet_v2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_tensor=x,
        )
    else:
        x = tf.keras.applications.resnet50.preprocess_input(x)
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=x,
        )

    base_model.trainable = False
    outputs = _prediction_head(base_model.output, num_classes)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"Transfer_{model_name}")
    return model, base_model
