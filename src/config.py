"""Shared paths and constants for the respiratory X-ray pipeline."""
import os
from pathlib import Path

# XRAY_ROOT redirects data/splits/results/models to another folder (used for isolated pipeline tests)
ROOT = Path(os.environ.get("XRAY_ROOT") or Path(__file__).resolve().parent.parent).resolve()

# Preprocessing ablation. XRAY_PREPROCESS=crop centre-crops every image to a square before resizing, so the
# aspect ratio is no longer squashed. All reported results use the default "squash"; any other variant reads
# the same raw data but writes its preprocessed images, splits, results and models to separate ablation_<variant>
# folders, so an ablation run can never overwrite the default pipeline's outputs or the model the app serves.
PREPROCESS_VARIANTS = ("squash", "crop")
PREPROCESS_VARIANT = os.environ.get("XRAY_PREPROCESS", "squash")
if PREPROCESS_VARIANT not in PREPROCESS_VARIANTS:
    raise ValueError(f"XRAY_PREPROCESS must be one of {PREPROCESS_VARIANTS}, got {PREPROCESS_VARIANT!r}")
_ABLATION = "" if PREPROCESS_VARIANT == "squash" else f"ablation_{PREPROCESS_VARIANT}"  # "" adds no path level

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
KERMANY_DIR = RAW_DIR / "extracted" / "chest_xray"
PREPROCESSED_DIR = DATA_DIR / "preprocessed" / _ABLATION
SPLITS_DIR = ROOT / "splits" / _ABLATION
RESULTS_DIR = ROOT / "results" / _ABLATION
MODELS_DIR = ROOT / "models" / _ABLATION

CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
BINARY_THRESHOLD = 0.5

# CLAHE runs after resizing to 224x224, so the same settings mean the same thing for
# every source regardless of native resolution
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

BINARY_CLASSES = ["NORMAL", "PNEUMONIA"]
MULTICLASS_ORDER = ["NORMAL", "PNEUMONIA", "COVID", "TB"]
DISPLAY_NAMES = {
    "NORMAL": "Normal",
    "PNEUMONIA": "Pneumonia",
    "COVID": "COVID-19",
    "TB": "Tuberculosis",
}
# In the multi-class task the PNEUMONIA class also holds COVID-QU-Ex "Non-COVID infection"
MULTICLASS_DISPLAY_NAMES = {**DISPLAY_NAMES, "PNEUMONIA": "Pneumonia / Non-COVID infection"}

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
