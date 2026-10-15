"""
Image preprocessing shared by training, Grad-CAM and the API.

grayscale -> fit to 224x224 -> CLAHE -> 3-channel RGB, uint8.
The fit step depends on the variant (src.config.PREPROCESS_VARIANT):
  squash (default, all reported results): resize the whole image (INTER_AREA); the aspect ratio is not kept
  crop (ablation): take the central square, then resize (INTER_AREA); no aspect-ratio distortion
  lungbox (ablation): a U-Net (src.lung_segmenter) finds the lungs; their bounding box plus a 5% margin is cut
    out (black where it leaves the image), centred on a black square, then resized (INTER_AREA). Nothing outside
    the box is kept, so the lungs fill the same part of every input. Falls back to the central square if no
    lungs are found.
Normalization to the model's input range happens inside each model
(Rescaling(1/255) for the baseline, resnet50.preprocess_input for ResNet50),
so every caller feeds the same uint8 RGB array.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import (CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, IMAGE_SIZE, LUNGBOX_MARGIN, PREPROCESS_VARIANT,
                        PREPROCESS_VARIANTS)
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


def center_square(width: int, height: int) -> tuple[int, int, int]:
    """(left, top, side) of the largest centred square inside a width x height image."""
    side = min(width, height)
    return (width - side) // 2, (height - side) // 2, side


def lung_rect(box) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of a lung box plus LUNGBOX_MARGIN of its width / height on each side (may leave the image)."""
    x0, y0, x1, y1 = box
    mx, my = LUNGBOX_MARGIN * (x1 - x0), LUNGBOX_MARGIN * (y1 - y0)
    return round(x0 - mx), round(y0 - my), round(x1 + mx), round(y1 + my)


def predicted_lung_rect(gray: np.ndarray):
    """lung_rect of the U-Net's lung box, or None if no lungs were found."""
    from src.lung_segmenter import lung_bbox  # loads TensorFlow; only the lungbox variant needs it

    box = lung_bbox(gray)
    return None if box is None else lung_rect(box)


def input_rect(gray: np.ndarray, variant: str = PREPROCESS_VARIANT):
    """(x0, y0, x1, y1) of the native-image region a variant fits to 224x224; None for squash (whole image)."""
    h, w = gray.shape[:2]
    left, top, side = center_square(w, h)
    centre = (left, top, left + side, top + side)
    if variant == "crop":
        return centre
    if variant == "lungbox":
        return predicted_lung_rect(gray) or centre
    return None


def crop_rect(arr: np.ndarray, rect) -> np.ndarray:
    """The (x0, y0, x1, y1) region of arr; parts outside arr are filled with 0 (black)."""
    x0, y0, x1, y1 = rect
    h, w = arr.shape[:2]
    if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
        return arr[y0:y1, x0:x1]
    out = np.zeros((y1 - y0, x1 - x0) + arr.shape[2:], dtype=arr.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def pad_to_square(arr: np.ndarray) -> np.ndarray:
    """Centre arr on a black square whose side is its longer edge (no-op for a square)."""
    h, w = arr.shape[:2]
    if h == w:
        return arr
    side = max(h, w)
    out = np.zeros((side, side) + arr.shape[2:], dtype=arr.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    out[top:top + h, left:left + w] = arr
    return out


def fit_to_input(gray: np.ndarray, variant: str = PREPROCESS_VARIANT, rect=None) -> np.ndarray:
    """Geometric step: native-size grayscale -> 224x224 for the given preprocessing variant.
    rect: the (x0, y0, x1, y1) to use instead of input_rect(gray, variant), e.g. to map a lung mask
    through the same crop as its X-ray."""
    if variant not in PREPROCESS_VARIANTS:
        raise ValueError(f"Unknown preprocessing variant {variant!r}")
    if variant != "squash":
        gray = pad_to_square(crop_rect(gray, rect or input_rect(gray, variant)))
    return resize_gray(gray)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)


def preprocess_gray(img, variant: str = PREPROCESS_VARIANT) -> np.ndarray:
    """Preprocessed single-channel 224x224 uint8 image."""
    return apply_clahe(fit_to_input(to_gray_uint8(img), variant))


def preprocess_image(img, variant: str = PREPROCESS_VARIANT) -> np.ndarray:
    """Model input: 224x224x3 uint8 RGB (the three channels are identical)."""
    return np.repeat(preprocess_gray(img, variant)[..., None], 3, axis=-1)


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
    logger.info(f"Preprocessing {len(pairs)} images (variant={PREPROCESS_VARIANT}, CLAHE clip={CLAHE_CLIP_LIMIT}, "
                f"tiles={CLAHE_TILE_GRID}, size={IMAGE_SIZE}) with {workers} threads ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        errors = [e for e in pool.map(_preprocess_one, pairs) if e]
    if errors:
        raise RuntimeError(f"{len(errors)} images failed preprocessing, e.g. {errors[:3]}")
    logger.info("Preprocessing finished.")
