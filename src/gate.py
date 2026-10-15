"""
Non-chest-image gate for the app.

The served classifier answers "which disease?" for any image, so on its own it cannot tell a face
photo from a chest X-ray. This gate compares each upload with chest X-rays in the classifier's own
feature space (the global-average-pooled ResNet50 features): cosine similarity to the k nearest of a
reference set of training X-rays. Images below a threshold are reported as not chest X-rays and get
no disease result.

  python -m src.gate                        # fit for the served model -> models/chest_gate.npz
  python -m src.gate --negatives FOLDER     # also report how many images in FOLDER are rejected

The threshold is the 1st percentile of similarities for validation X-rays, so about 99% of real chest
X-rays pass. The test split is used only to report how often real X-rays are rejected.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.config import BEST_MODEL_PATH, MODELS_DIR, SEED, SPLITS_DIR
from src.data.preprocessing import load_image, preprocess_image
from src.utils import absolute, logger, save_json, sha256_file

GATE_PATH = MODELS_DIR / "chest_gate.npz"
GATE_REPORT = MODELS_DIR / "chest_gate.json"
FEATURE_LAYER = "gap"  # global average pool of the served ResNet50, 2048-d
K = 5                  # neighbours averaged per image
REF_SIZE = 400         # training X-rays kept as the reference set
CAL_SIZE = 300         # validation X-rays used to set the threshold
TEST_SIZE = 300        # test X-rays used only to report false rejections
PERCENTILE = 1.0       # threshold = this percentile of validation similarities


def feature_model(model: tf.keras.Model) -> tf.keras.Model:
    return tf.keras.Model(model.input, model.get_layer(FEATURE_LAYER).output)


def embed(fmodel: tf.keras.Model, images: np.ndarray, batch: int = 32) -> np.ndarray:
    """images: float32 (n, 224, 224, 3) in [0, 255] -> L2-normalised features (n, d)."""
    feats = np.concatenate([fmodel(images[i:i + batch], training=False).numpy()
                            for i in range(0, len(images), batch)])
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def knn_similarity(feats: np.ndarray, reference: np.ndarray, k: int = K) -> np.ndarray:
    """Mean cosine similarity to the k most similar reference X-rays."""
    sims = feats @ reference.T
    return np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)


def _rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=-1).astype("float32")


def load_stored(paths) -> np.ndarray:
    """Training-time inputs: the stored CLAHE PNGs, exactly as the classifier saw them."""
    return np.stack([_rgb(np.asarray(Image.open(absolute(p)).convert("L"))) for p in paths])


def load_uploads(paths) -> np.ndarray:
    """App-time inputs: the same preprocessing the API applies to an upload."""
    return np.stack([preprocess_image(load_image(p)).astype("float32") for p in paths])


class ChestGate:
    def __init__(self, fmodel: tf.keras.Model, reference: np.ndarray, threshold: float, model_sha: str):
        self.fmodel = fmodel
        self.reference = reference
        self.threshold = float(threshold)
        self.model_sha = model_sha

    @classmethod
    def load(cls, model: tf.keras.Model, model_path=BEST_MODEL_PATH, path=GATE_PATH):
        """Returns None (gate off) if the file is missing or was fitted for a different model file."""
        path = Path(path)
        if not path.exists():
            return None
        with np.load(path) as data:
            if str(data["model_sha"]) != sha256_file(model_path):
                logger.warning(f"{path} was fitted for another model; run python -m src.gate. Gate disabled.")
                return None
            return cls(feature_model(model), data["reference"].astype("float32"), float(data["threshold"]),
                       str(data["model_sha"]))

    def similarity(self, model_input: np.ndarray) -> float:
        """model_input: the 224x224x3 uint8 array the API builds for one upload."""
        feats = embed(self.fmodel, model_input[None].astype("float32"))
        return float(knn_similarity(feats, self.reference)[0])

    def is_chest_xray(self, model_input: np.ndarray):
        score = self.similarity(model_input)
        return score >= self.threshold, score

    def save(self, path=GATE_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, reference=self.reference.astype("float16"), threshold=self.threshold,
                 model_sha=self.model_sha)


def fit(model: tf.keras.Model, model_path=BEST_MODEL_PATH, seed: int = SEED):
    fmodel = feature_model(model)
    split = {s: pd.read_csv(SPLITS_DIR / "multiclass" / f"{s}.csv") for s in ("train", "val", "test")}
    ref_df = split["train"].sample(REF_SIZE, random_state=seed)
    cal_df = split["val"].sample(CAL_SIZE, random_state=seed)
    test_df = split["test"].sample(TEST_SIZE, random_state=seed)

    reference = embed(fmodel, load_stored(ref_df["preprocessed_path"]))
    cal = knn_similarity(embed(fmodel, load_stored(cal_df["preprocessed_path"])), reference)
    test = knn_similarity(embed(fmodel, load_stored(test_df["preprocessed_path"])), reference)
    threshold = float(np.percentile(cal, PERCENTILE))
    report = {
        "model_file": str(model_path), "model_sha256": sha256_file(model_path),
        "feature_layer": FEATURE_LAYER, "k": K, "reference_size": REF_SIZE,
        "threshold": threshold, "percentile": PERCENTILE,
        "validation_similarity": {"min": float(cal.min()), "median": float(np.median(cal))},
        "test_real_xrays_rejected": int((test < threshold).sum()), "test_n": int(len(test)),
        "test_similarity_median": float(np.median(test)),
    }
    return ChestGate(fmodel, reference, threshold, report["model_sha256"]), report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives", type=Path, default=None,
                        help="folder of non-chest photos; reports how many would be rejected")
    args = parser.parse_args()

    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    gate, report = fit(model, BEST_MODEL_PATH)
    gate.save()
    logger.info(f"Chest gate: threshold {gate.threshold:.4f}; real X-rays rejected on test "
                f"{report['test_real_xrays_rejected']}/{report['test_n']}")

    if args.negatives:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        paths = sorted(p for p in args.negatives.rglob("*") if p.suffix.lower() in exts)
        sims = [gate.similarity(x) for x in load_uploads(paths)]
        rejected = [p.name for p, s in zip(paths, sims) if s < gate.threshold]
        report["negatives"] = {"folder": str(args.negatives), "n": len(paths), "rejected": len(rejected),
                               "similarity_median": float(np.median(sims)),
                               "accepted_examples": [p.name for p, s in zip(paths, sims) if s >= gate.threshold][:20]}
        logger.info(f"Non-chest folder: {len(rejected)}/{len(paths)} rejected")

    save_json(report, GATE_REPORT)
    logger.info(f"Saved {GATE_PATH} and {GATE_REPORT}")


if __name__ == "__main__":
    main()
