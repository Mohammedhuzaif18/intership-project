import hashlib
import json
import logging
import os
import random
import shutil
import stat
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("RespiratoryDL")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow for repeatable runs."""
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Fixed random seed set to: {seed}")


def detect_device() -> str:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU detected: {len(gpus)} device(s) - {[gpu.name for gpu in gpus]}")
        return "GPU"
    logger.info("No GPU detected. Running on CPU.")
    return "CPU"


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _clear_readonly_and_retry(func, path, _exc):
    # OneDrive marks synced folders read-only, and Windows refuses to rmdir those
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path) -> None:
    """shutil.rmtree that also works on read-only (e.g. OneDrive) folders on Windows."""
    if os.path.isdir(path):
        shutil.rmtree(path, onexc=_clear_readonly_and_retry)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rel(path) -> str:
    """Project-relative POSIX path, used in split files so they work on any machine."""
    from src.config import ROOT

    return Path(path).resolve().relative_to(ROOT).as_posix()


def absolute(rel_path) -> Path:
    from src.config import ROOT

    return ROOT / rel_path
