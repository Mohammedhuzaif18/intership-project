"""tf.data pipelines built from the split files, augmentation, class weights."""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.config import BATCH_SIZE, IMAGE_SIZE, ROOT, SEED, SPLITS_DIR
from src.utils import logger


def get_augmentation(seed: int = SEED) -> tf.keras.Sequential:
    """
    Training-time augmentation. These Keras layers are only active with training=True,
    so validation, test and API inference always see the unaugmented image.
    Constant (black) fill avoids mirrored anatomy at the borders.
    """
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.RandomRotation(15 / 360, fill_mode="constant", seed=seed),
        L.RandomTranslation(0.1, 0.1, fill_mode="constant", seed=seed),
        L.RandomZoom(0.1, fill_mode="constant", seed=seed),
        L.RandomFlip("horizontal", seed=seed),
        L.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
        L.RandomContrast(0.1, seed=seed),
    ], name="train_augmentation")


def load_split(task: str, split: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS_DIR / task / f"{split}.csv")


def encode_labels(df: pd.DataFrame, class_names) -> np.ndarray:
    index = {c: i for i, c in enumerate(class_names)}
    unknown = set(df["label"]) - set(index)
    if unknown:
        raise ValueError(f"Labels not in class_names: {unknown}")
    return df["label"].map(index).to_numpy()


def make_dataset(df: pd.DataFrame, class_names, training: bool, batch_size: int = BATCH_SIZE,
                 seed: int = SEED) -> tf.data.Dataset:
    """(image float32 RGB in [0, 255], label) batches; labels are (n, 1) floats for 2 classes, one-hot otherwise."""
    paths = [str(ROOT / p) for p in df["preprocessed_path"]]
    idx = encode_labels(df, class_names)
    if len(class_names) == 2:
        labels = idx.astype("float32")[:, None]
    else:
        labels = np.eye(len(class_names), dtype="float32")[idx]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)  # grayscale PNG -> 3 identical channels
        img = tf.ensure_shape(img, (*IMAGE_SIZE, 3))
        return tf.cast(img, tf.float32), label

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(df: pd.DataFrame, class_names) -> dict:
    """'balanced' weights: n_samples / (n_classes * n_class_samples)."""
    y = encode_labels(df, class_names)
    w = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=y)
    weights = {int(i): float(v) for i, v in enumerate(w)}
    logger.info(f"Class weights: { {class_names[i]: round(v, 3) for i, v in weights.items()} }")
    return weights
