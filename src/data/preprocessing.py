"""
Image preprocessing shared by training, Grad-CAM and the API.

grayscale -> resize to 224x224 (INTER_AREA) -> CLAHE -> 3-channel RGB, uint8.
Normalization to the model's input range happens inside each model
(Rescaling(1/255) for the baseline, resnet50.preprocess_input for ResNet50),
so every caller feeds the same uint8 RGB array.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, IMAGE_SIZE
from src.utils import logger


def to_gray_uint8(img) -> np.ndarray:
    """PIL image or array (any common mode, 8- or 16-bit) -> 2-D uint8 array."""
    if isinstance(img, Image.Image):
        if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.asarray(img, dtype=np.float32)
        else:
            return np.asarray(img.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=-1) if arr.shape[-1] >= 3 else arr[..., 0]
        if arr.dtype == np.uint8:
            return arr
        arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    scaled = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
    return scaled.astype(np.uint8)


def resize_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)


def preprocess_gray(img) -> np.ndarray:
    """Preprocessed single-channel 224x224 uint8 image."""
    return apply_clahe(resize_gray(to_gray_uint8(img)))


def preprocess_image(img) -> np.ndarray:
    """Model input: 224x224x3 uint8 RGB (the three channels are identical)."""
    return np.repeat(preprocess_gray(img)[..., None], 3, axis=-1)


def load_image(path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return im.copy()


def _preprocess_one(args):
    src, dst = args
    try:
        gray = preprocess_gray(load_image(src))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        # Stored single-channel; the loader decodes it to 3-channel RGB
        Image.fromarray(gray).save(dst, format="PNG")
        return None
    except Exception as e:
        return f"{src}: {type(e).__name__}: {e}"


def preprocess_files(pairs, workers: int = 8) -> None:
    """pairs: iterable of (source path, destination .png path)."""
    pairs = list(pairs)
    logger.info(f"Preprocessing {len(pairs)} images (CLAHE clip={CLAHE_CLIP_LIMIT}, tiles={CLAHE_TILE_GRID}, "
                f"size={IMAGE_SIZE}) with {workers} threads ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        errors = [e for e in pool.map(_preprocess_one, pairs) if e]
    if errors:
        raise RuntimeError(f"{len(errors)} images failed preprocessing, e.g. {errors[:3]}")
    logger.info("Preprocessing finished.")
