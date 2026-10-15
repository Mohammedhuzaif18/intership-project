"""Dataset downloads: Kermany via Hugging Face, extension datasets via the Kaggle API."""
import os
import zipfile
from collections import defaultdict
from pathlib import Path

from src.config import IMAGE_EXTS, KERMANY_DIR, RAW_DIR
from src.utils import logger


def download_kermany() -> Path:
    """Fetch chest_xray.zip from the Hugging Face mirror and extract it under data/raw/extracted."""
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(repo_id="AiresPucrs/chest-xray", filename="chest_xray.zip", repo_type="dataset")
    logger.info(f"Kermany zip: {zip_path}")
    if not all((KERMANY_DIR / s).is_dir() for s in ("train", "val", "test")):
        logger.info(f"Extracting to {KERMANY_DIR.parent} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(KERMANY_DIR.parent)
    else:
        logger.info("Kermany already extracted.")
    return KERMANY_DIR


def print_tree_with_counts(root: Path) -> dict:
    """Log the folder tree with image counts; returns {relative folder: count}."""
    counts = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        n = sum(1 for f in filenames if f.lower().endswith(IMAGE_EXTS) and not f.startswith("._"))
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        name = root.name if depth == 0 else rel.parts[-1]
        logger.info(f"{'    ' * depth}{name}/" + (f"  ({n} images)" if n else ""))
        if n:
            counts[rel.as_posix()] = n
    per_class = defaultdict(int)
    for folder, n in counts.items():
        per_class[Path(folder).name.upper()] += n
    logger.info(f"Images per class: {dict(per_class)}  total={sum(per_class.values())}")
    return counts


def kaggle_credentials_available() -> bool:
    """Credentials are read only from KAGGLE_USERNAME / KAGGLE_KEY; their values are never logged."""
    return bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))


def download_kaggle_dataset(slug: str, dest_name: str):
    """Download and unzip a Kaggle dataset into data/raw/<dest_name>. Returns the folder or None."""
    if not kaggle_credentials_available():
        logger.warning(f"KAGGLE_USERNAME / KAGGLE_KEY not set; cannot download {slug}.")
        return None
    dest = RAW_DIR / dest_name
    if dest.is_dir() and any(dest.iterdir()):
        logger.info(f"Using existing download: {dest}")
        return dest
    try:
        # Imported lazily: the kaggle package authenticates on import
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading Kaggle dataset {slug} -> {dest}")
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
        return dest
    except Exception as e:
        logger.warning(f"Kaggle download of {slug} failed: {type(e).__name__}: {e}")
        return None
