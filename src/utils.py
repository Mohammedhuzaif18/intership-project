import os
import random
import shutil
import stat
import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RespiratoryDL")

def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Fixed random seed set to: {seed}")

def _clear_readonly_and_retry(func, path, _exc):
    # OneDrive marks synced folders read-only, and Windows refuses to rmdir those
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path: str) -> None:
    """shutil.rmtree that also works on read-only (e.g. OneDrive) folders on Windows."""
    if os.path.isdir(path):
        shutil.rmtree(path, onexc=_clear_readonly_and_retry)


def detect_device() -> str:
    """Detect GPU availability in TensorFlow."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU detected: {len(gpus)} device(s) - {[gpu.name for gpu in gpus]}")
        return "GPU"
    else:
        logger.info("No GPU detected. Running on CPU.")
        return "CPU"
