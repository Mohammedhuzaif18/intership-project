"""
Lung segmenter for the lung-box preprocessing ablation (XRAY_PREPROCESS=lungbox).

A small U-Net on 256x256 grayscale inputs, trained on the COVID-QU-Ex lung masks
(Lung Segmentation Data/{Train,Val,Test}/*/lung masks). Train fits, Val picks the epoch, Test reports Dice.
Two groups of images are kept out of training and validation, so the segmenter has never seen the images
the ablation evaluates on:
  - COVID-QU-Ex copies of Kermany val/test images (the near-duplicates matched in src.data.multiclass);
  - the 2,400 COVID-QU-Ex images of the cross-dataset test (splits/cross_dataset).
Copies of Kermany *train* images stay in, so the segmenter sees the pediatric Kermany style.

  python -m src.lung_segmenter            # train -> models/lung_segmenter/, Dice -> results/ablation_lungbox/segmenter

At inference, a native-size X-ray is resized to 256x256 (whole image, as COVID-QU-Ex did with its copies of
Kermany), segmented, and the lung bounding box is mapped back to native pixels (lung_bbox).
"""
import argparse
import threading

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import LUNG_SEGMENTER_PATH, RAW_DIR, ROOT, SEED, variant_subdir
from src.utils import absolute, logger, save_json, set_seed

SEG_SIZE = 256
SEG_ROOT = RAW_DIR / "covidqu" / "Lung Segmentation Data" / "Lung Segmentation Data"
OUT = ROOT / "results" / variant_subdir("lungbox") / "segmenter"
# Sized for CPU training: 16 base filters took 2.2 s per batch of 16 here (2.6 h for 6 epochs); a lung bounding box
# needs only a coarse mask
FILTERS = (8, 16, 32, 64, 128)  # per U-Net level; the last is the bottleneck
EPOCHS = 5
STEPS_PER_EPOCH = 400           # x BATCH = 6,400 random training images per epoch
BATCH = 16
THRESHOLD = 0.5
MIN_COMPONENT = 0.10   # keep connected components with at least 10% of the largest one's area
MIN_LUNG_AREA = 0.01   # a mask under 1% of the image counts as empty (caller falls back)

L = tf.keras.layers


# ---------- Data ----------

def list_pairs() -> pd.DataFrame:
    rows = []
    for split in ("Train", "Val", "Test"):
        for img in sorted((SEG_ROOT / split).glob("*/images/*.png")):
            # relative_to, not utils.rel: Path.resolve() per file takes minutes for 34k files on OneDrive
            rows.append({"split": split, "group": img.parent.parent.name, "image": img.relative_to(ROOT).as_posix(),
                         "mask": (img.parent.parent / "lung masks" / img.name).relative_to(ROOT).as_posix()})
    return pd.DataFrame(rows)


def held_out_images() -> dict:
    """COVID-QU-Ex image -> why it is kept out of segmenter training (Kermany val/test copy, cross-dataset test)."""
    out = {}
    removed = pd.read_csv(ROOT / "splits" / "multiclass" / "removed.csv")
    kermany = removed["remove_reason"].str.extract(r"^near-duplicate .* of (data/raw/extracted/chest_xray/.+)$")[0]
    kermany_split = pd.concat([pd.read_csv(ROOT / "splits" / "binary" / f"{s}.csv").assign(split=s)
                               for s in ("train", "val", "test")]).set_index("path")["split"]
    for path, k in zip(removed["path"], kermany):
        if isinstance(k, str) and kermany_split.get(k) in ("val", "test"):
            out[path] = f"kermany_{kermany_split[k]}_copy"
    for path in pd.read_csv(ROOT / "splits" / "cross_dataset" / "covidqu_normal_pneumonia.csv")["path"]:
        out.setdefault(path, "cross_dataset_test")
    return out


def _load(image_path, mask_path):
    img = tf.image.decode_png(tf.io.read_file(image_path), channels=1)
    mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)
    img = tf.image.resize(tf.cast(img, tf.float32) / 255.0, (SEG_SIZE, SEG_SIZE), antialias=True)
    mask = tf.cast(tf.image.resize(mask, (SEG_SIZE, SEG_SIZE), method="nearest") > 127, tf.float32)
    return img, mask


def augmentation() -> tf.keras.Sequential:
    """Geometric augmentation applied to image and mask together (stacked as two channels)."""
    return tf.keras.Sequential([
        L.RandomRotation(10 / 360, fill_mode="constant", seed=SEED),
        L.RandomTranslation(0.1, 0.1, fill_mode="constant", seed=SEED),
        L.RandomZoom((-0.15, 0.25), fill_mode="constant", seed=SEED),
        L.RandomFlip("horizontal", seed=SEED),
    ], name="seg_augmentation")


def make_dataset(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
    paths = [str(absolute(p)) for p in df["image"]], [str(absolute(p)) for p in df["mask"]]
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True).repeat()
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)
    if training:
        aug = augmentation()

        def augment(img, mask):
            both = aug(tf.concat([img, mask], axis=-1), training=True)
            img, mask = both[..., :1], tf.cast(both[..., 1:] > 0.5, tf.float32)
            # Brightness / contrast on the image only
            img = tf.image.random_contrast(tf.image.random_brightness(img, 0.1, seed=SEED), 0.8, 1.2, seed=SEED)
            return tf.clip_by_value(img, 0.0, 1.0), mask

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------- Model ----------

def _block(x, filters: int, name: str):
    for i in (1, 2):
        x = L.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{i}")(x)
        x = L.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = L.Activation("relu", name=f"{name}_relu{i}")(x)
    return x


def build_unet(filters=FILTERS) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(SEG_SIZE, SEG_SIZE, 1), name="xray")
    x, skips = inputs, []
    for i, f in enumerate(filters[:-1]):
        x = _block(x, f, f"down{i}")
        skips.append(x)
        x = L.MaxPooling2D(2, name=f"pool{i}")(x)
    x = _block(x, filters[-1], "bottleneck")
    for i, f in reversed(list(enumerate(filters[:-1]))):
        x = L.Conv2DTranspose(f, 2, strides=2, padding="same", name=f"up{i}")(x)
        x = _block(L.Concatenate(name=f"skip{i}")([x, skips[i]]), f, f"dec{i}")
    outputs = L.Conv2D(1, 1, activation="sigmoid", name="lung_prob")(x)
    return tf.keras.Model(inputs, outputs, name="Lung_UNet")


def dice_loss(y_true, y_pred, eps: float = 1.0):
    inter = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    total = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1.0 - tf.reduce_mean((2.0 * inter + eps) / (total + eps))


def bce_dice_loss(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred)) + dice_loss(y_true, y_pred)


def hard_dice(y_true, y_pred):
    """Mean per-image Dice of the thresholded prediction (both empty -> 1)."""
    pred = tf.cast(y_pred >= THRESHOLD, tf.float32)
    inter = tf.reduce_sum(y_true * pred, axis=(1, 2, 3))
    total = tf.reduce_sum(y_true + pred, axis=(1, 2, 3))
    return tf.reduce_mean(tf.where(total > 0, 2.0 * inter / tf.maximum(total, 1.0), 1.0))


# ---------- Inference ----------

class _Segmenter:
    """Loads the trained U-Net once; calls are serialised so threaded preprocessing stays deterministic."""
    _model = None
    _lock = threading.Lock()

    @classmethod
    def predict(cls, gray: np.ndarray) -> np.ndarray:
        """Native-size uint8 grayscale -> lung probability map, SEG_SIZE x SEG_SIZE float32."""
        x = cv2.resize(gray, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        with cls._lock:
            if cls._model is None:
                if not LUNG_SEGMENTER_PATH.exists():
                    raise FileNotFoundError(f"{LUNG_SEGMENTER_PATH} not found: run python -m src.lung_segmenter")
                cls._model = tf.keras.models.load_model(LUNG_SEGMENTER_PATH, compile=False)
            return cls._model(x[None, ..., None], training=False).numpy()[0, ..., 0]


def clean_mask(prob: np.ndarray) -> np.ndarray:
    """Threshold, then drop connected components smaller than MIN_COMPONENT of the largest (stray blobs)."""
    binary = (prob >= THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= MIN_COMPONENT * areas.max())
    return np.isin(labels, keep)


def predict_mask(gray: np.ndarray) -> np.ndarray:
    """Cleaned lung mask at SEG_SIZE x SEG_SIZE (bool) for a native-size uint8 grayscale X-ray."""
    return clean_mask(_Segmenter.predict(gray))


def mask_bbox(mask: np.ndarray, width: int, height: int):
    """(x0, y0, x1, y1) of a SEG_SIZE mask's lungs in native pixels (floats), or None if the mask is (nearly) empty."""
    if mask.mean() < MIN_LUNG_AREA:
        return None
    ys, xs = np.nonzero(mask)
    sx, sy = width / mask.shape[1], height / mask.shape[0]
    return xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy


def lung_bbox(gray: np.ndarray):
    """Predicted lung bounding box (x0, y0, x1, y1) in native pixels, or None if no lungs were found."""
    return mask_bbox(predict_mask(gray), gray.shape[1], gray.shape[0])


# ---------- Training and evaluation ----------

def _dice_np(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / total) if total else 1.0


def _box_iou(a, b) -> float:
    if a is None or b is None:
        return np.nan
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union)


def evaluate(model: tf.keras.Model, df: pd.DataFrame) -> pd.DataFrame:
    """Per-image Dice (raw and cleaned mask) and lung bounding-box IoU against the ground-truth mask."""
    rows = []
    for start in range(0, len(df), 64):
        chunk = df.iloc[start:start + 64]
        loaded = [_load(str(absolute(r.image)), str(absolute(r.mask))) for r in chunk.itertuples()]
        x = np.stack([i.numpy() for i, _ in loaded])
        probs = model.predict_on_batch(x)[..., 0]
        for r, (_, m), prob in zip(chunk.itertuples(), loaded, probs):
            truth = m.numpy()[..., 0] > 0.5
            cleaned = clean_mask(prob)
            rows.append({"image": r.image, "split": r.split, "group": r.group, "held_out": r.held_out,
                         "dice": _dice_np(prob >= THRESHOLD, truth), "dice_cleaned": _dice_np(cleaned, truth),
                         "bbox_iou": _box_iou(mask_bbox(cleaned, SEG_SIZE, SEG_SIZE),
                                              mask_bbox(truth, SEG_SIZE, SEG_SIZE)),
                         "empty_prediction": mask_bbox(cleaned, SEG_SIZE, SEG_SIZE) is None})
    return pd.DataFrame(rows)


def summarize(per_image: pd.DataFrame) -> pd.DataFrame:
    def row(name, d):
        return {"Images": name, "n": len(d), "Dice (mean)": d["dice"].mean(), "Dice (median)": d["dice"].median(),
                "Dice after cleaning": d["dice_cleaned"].mean(), "Dice < 0.80": int((d["dice_cleaned"] < 0.8).sum()),
                "Lung bounding-box IoU": d["bbox_iou"].mean(), "Empty predictions": int(d["empty_prediction"].sum())}

    test = per_image[per_image["split"] == "Test"]
    rows = [row("Test folder, all", test)]
    rows += [row(f"Test folder, {g}", d) for g, d in test.groupby("group")]
    rows += [row(f"Held out from training: {h.replace('_', ' ')} (any folder)", d)
             for h, d in per_image[per_image["held_out"] != ""].groupby("held_out")]
    return pd.DataFrame(rows)


def train(epochs: int = EPOCHS, steps: int = STEPS_PER_EPOCH):
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    LUNG_SEGMENTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs = list_pairs()
    held = held_out_images()
    pairs["held_out"] = pairs["image"].map(held).fillna("")
    assert pairs["mask"].map(lambda p: absolute(p).exists()).all(), "missing lung masks"
    fit = pairs[(pairs["split"] == "Train") & (pairs["held_out"] == "")]
    val = pairs[(pairs["split"] == "Val") & (pairs["held_out"] == "")]
    counts = pairs.groupby(["split", "held_out"]).size().rename("images").reset_index()
    logger.info(f"Lung segmentation pairs:\n{counts.to_string(index=False)}")
    logger.info(f"Training on {len(fit)} Train images, selecting on {len(val)} Val images; "
                f"{epochs} epochs x {steps} steps x batch {BATCH}")

    model = build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=bce_dice_loss, metrics=[hard_dice])
    logger.info(f"U-Net parameters: {model.count_params():,}")
    history = model.fit(
        make_dataset(fit, training=True), steps_per_epoch=steps, epochs=epochs,
        validation_data=make_dataset(val, training=False), verbose=2,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(LUNG_SEGMENTER_PATH, monitor="val_hard_dice", mode="max",
                                                      save_best_only=True, verbose=1),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor="val_hard_dice", mode="max", factor=0.3,
                                                        patience=1, verbose=1),
                   tf.keras.callbacks.LambdaCallback(on_train_batch_end=lambda b, logs: (b + 1) % 100 or logger.info(
                       f"  step {b + 1}: loss {logs['loss']:.4f}, Dice {logs['hard_dice']:.4f}"))])
    pd.DataFrame(history.history).rename_axis("epoch").to_csv(OUT / "history.csv")

    best = tf.keras.models.load_model(LUNG_SEGMENTER_PATH, compile=False)
    report(best, pairs, counts, len(fit), len(val), epochs, steps)


def report(model, pairs, counts, n_fit, n_val, epochs, steps):
    scored = pairs[(pairs["split"] == "Test") | (pairs["held_out"] != "")]
    logger.info(f"Scoring {len(scored)} images (Test folder + held-out images in Train/Val) ...")
    per_image = evaluate(model, scored)
    per_image.to_csv(OUT / "dice_per_image.csv", index=False)
    summary = summarize(per_image)
    with open(OUT / "dice.md", "w", encoding="utf-8") as f:
        f.write("# Lung segmenter (U-Net, 256x256) on COVID-QU-Ex lung masks\n\n"
                + summary.to_markdown(index=False, floatfmt=".4f") + "\n")
    logger.info(f"Dice:\n{summary.to_string(index=False)}")
    save_json({"model": str(LUNG_SEGMENTER_PATH.relative_to(ROOT).as_posix()), "parameters": model.count_params(),
               "filters": FILTERS, "input": f"{SEG_SIZE}x{SEG_SIZE} grayscale / 255",
               "training": {"images": n_fit, "val_images": n_val, "epochs": epochs, "steps_per_epoch": steps,
                            "batch": BATCH, "loss": "BCE + soft Dice", "optimizer": "Adam 1e-3",
                            "selection": "best validation Dice"},
               "threshold": THRESHOLD, "min_component_fraction": MIN_COMPONENT, "min_lung_area": MIN_LUNG_AREA,
               "pairs": counts.to_dict(orient="records"), "dice": summary.to_dict(orient="records")},
              OUT / "segmenter.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPOCH)
    parser.add_argument("--evaluate-only", action="store_true", help="score the saved model without training")
    args = parser.parse_args()
    if args.evaluate_only:
        pairs = list_pairs()
        pairs["held_out"] = pairs["image"].map(held_out_images()).fillna("")
        counts = pairs.groupby(["split", "held_out"]).size().rename("images").reset_index()
        report(tf.keras.models.load_model(LUNG_SEGMENTER_PATH, compile=False), pairs, counts,
               int(((pairs["split"] == "Train") & (pairs["held_out"] == "")).sum()),
               int(((pairs["split"] == "Val") & (pairs["held_out"] == "")).sum()), args.epochs, args.steps)
    else:
        train(args.epochs, args.steps)


if __name__ == "__main__":
    main()
