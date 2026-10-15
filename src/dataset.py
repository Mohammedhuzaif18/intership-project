import json
import os
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from src.utils import logger
from src.config import PREFERRED_CLASSES, IMAGE_SIZE, BATCH_SIZE

def get_augmentation_layer():
    """Augmentation: rotation ±15°, shift ±10%, zoom ±10%, horizontal flip."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=(-15 / 360, 15 / 360)),
        tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1)),
        tf.keras.layers.RandomFlip("horizontal"),
    ], name="data_augmentation")


def get_default_data_dir():
    candidates = [
        "./data/preprocessed_224",
        "./data/processed_multiclass",
        "./data/processed",
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "train")):
            return path
    return "./data/processed"


def discover_classes(train_dir: str):
    found = [
        name for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name)) and not name.startswith(".")
    ]
    ordered = [c for c in PREFERRED_CLASSES if c in found]
    extras = sorted(c for c in found if c not in PREFERRED_CLASSES)
    classes = ordered + extras
    if not classes:
        raise FileNotFoundError(f"No class folders found in {train_dir}")
    logger.info(f"Discovered classes: {classes}")
    return classes


def save_class_indices(class_names):
    os.makedirs("./models", exist_ok=True)
    mapping = {"classes": class_names, "num_classes": len(class_names)}
    for i, name in enumerate(class_names):
        mapping[str(i)] = name
    with open("./models/class_indices.json", "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved models/class_indices.json")


def create_datasets(data_dir: str = None, batch_size: int = BATCH_SIZE):
    if data_dir is None:
        data_dir = get_default_data_dir()
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    class_names = discover_classes(train_dir)
    num_classes = len(class_names)
    label_mode = "binary" if num_classes <= 2 else "categorical"

    logger.info(f"Loading datasets from disk ({data_dir}) with label_mode={label_mode}...")

    common = dict(
        labels="inferred",
        label_mode=label_mode,
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, seed=42, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, shuffle=False, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, shuffle=False, **common)

    save_class_indices(class_names)
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_dir: str = None, class_names=None):
    if train_dir is None:
        train_dir = os.path.join(get_default_data_dir(), "train")
    if class_names is None:
        class_names = discover_classes(train_dir)

    labels = []
    counts = {}
    for idx, name in enumerate(class_names):
        folder = os.path.join(train_dir, name)
        n = len([f for f in os.listdir(folder) if not f.startswith(".")]) if os.path.isdir(folder) else 0
        counts[name] = n
        labels.extend([idx] * n)

    y = np.array(labels)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=y)
    class_weights = {int(i): float(w) for i, w in enumerate(weights)}
    logger.info(f"Class counts in train: {counts}")
    logger.info(f"Computed class weights: {class_weights}")
    return class_weights
