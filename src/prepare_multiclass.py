"""
Collect COVID-19 and TB chest X-rays and merge them with the Kermany
NORMAL / PNEUMONIA set into a stratified 4-class dataset.

Sources (no Kaggle required):
  - COVID: ieee8023/covid-chestxray-dataset (GitHub)
  - TB: NLM Montgomery County set (NIH Open-i)
Optional extra sources if KAGGLE_USERNAME / KAGGLE_KEY are set.
Local drop-in folders are also accepted: data/raw/extra/COVID and data/raw/extra/TB
"""
import csv
import os
import shutil
import sys
import zipfile
from collections import defaultdict
from urllib.request import urlretrieve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prepare_data import extract_and_prepare
from src.splits import kermany_groups, stratified_group_split, assert_no_file_overlap
from src.utils import logger, set_seed, remove_tree
from src.config import SEED, PREFERRED_CLASSES

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS) and not name.startswith("._")


def _one_group_per_file(paths, source: str):
    """Fallback when a source has no patient ids: each image is its own patient."""
    if paths:
        logger.warning(
            f"No patient ids for {len(paths)} {source} images; each is treated as its own patient, "
            "so repeat scans of one patient could still land in different splits."
        )
    return [f"file:{p}" for p in paths]


def download_zip(url: str, dest_zip: str):
    os.makedirs(os.path.dirname(dest_zip), exist_ok=True)
    if os.path.exists(dest_zip) and os.path.getsize(dest_zip) > 1024:
        logger.info(f"Using cached zip: {dest_zip}")
        return dest_zip
    logger.info(f"Downloading {url}")
    urlretrieve(url, dest_zip)
    return dest_zip


def extract_zip(zip_path: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    if os.listdir(dest_dir):
        logger.info(f"Already extracted: {dest_dir}")
        return dest_dir
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir


def collect_from_folder(folder: str):
    paths = []
    if not os.path.isdir(folder):
        return paths
    for root, _, files in os.walk(folder):
        for f in files:
            if _is_image(f):
                paths.append(os.path.join(root, f))
    return paths


def collect_covid_images(raw_dir: str):
    """Returns (paths, patient_groups)."""
    extra = os.path.join(raw_dir, "extra", "COVID")
    extra_files = collect_from_folder(extra)
    if extra_files:
        logger.info(f"Using {len(extra_files)} local COVID images from {extra}")
        return extra_files, _one_group_per_file(extra_files, "local COVID")

    url = "https://github.com/ieee8023/covid-chestxray-dataset/archive/refs/heads/master.zip"
    zip_path = os.path.join(raw_dir, "covid_chestxray.zip")
    extract_dir = os.path.join(raw_dir, "covid_chestxray")
    try:
        download_zip(url, zip_path)
        extract_zip(zip_path, extract_dir)
    except Exception as e:
        logger.warning(f"COVID GitHub download failed: {e}")
        return [], []

    metadata_path = None
    images_dir = None
    for root, dirs, files in os.walk(extract_dir):
        if "metadata.csv" in files:
            metadata_path = os.path.join(root, "metadata.csv")
        if os.path.basename(root).lower() == "images":
            images_dir = root

    selected, groups = [], []
    if metadata_path and images_dir:
        with open(metadata_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                finding = (row.get("finding") or "").upper()
                view = (row.get("view") or "").upper()
                filename = row.get("filename") or ""
                if "COVID" not in finding:
                    continue
                if view and view not in {"PA", "AP", "AP SUPERVISOR", "AP_HORIZONTAL"}:
                    continue
                src = os.path.join(images_dir, filename)
                if os.path.exists(src) and _is_image(filename):
                    selected.append(src)
                    # The metadata has several images for many patients
                    groups.append(f"covid-{row.get('patientid') or filename}")
    else:
        selected = collect_from_folder(extract_dir)
        groups = _one_group_per_file(selected, "COVID (no metadata.csv)")

    logger.info(f"Collected {len(selected)} COVID frontal X-rays from {len(set(groups))} patients")
    return selected, groups


def collect_tb_images(raw_dir: str):
    """Returns (paths, patient_groups)."""
    extra = os.path.join(raw_dir, "extra", "TB")
    extra_files = collect_from_folder(extra)
    if extra_files:
        logger.info(f"Using {len(extra_files)} local TB images from {extra}")
        return extra_files, _one_group_per_file(extra_files, "local TB")

    url = "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
    zip_path = os.path.join(raw_dir, "montgomery_tb.zip")
    extract_dir = os.path.join(raw_dir, "montgomery_tb")
    try:
        download_zip(url, zip_path)
        extract_zip(zip_path, extract_dir)
    except Exception as e:
        logger.warning(f"TB Montgomery download failed: {e}")
        return [], []

    selected = []
    for path in collect_from_folder(extract_dir):
        stem = os.path.splitext(os.path.basename(path))[0]
        # Montgomery: MCUCXR_xxxx_0.png = normal, _1.png = tuberculosis
        if stem.endswith("_1"):
            selected.append(path)
    if not selected:
        selected = collect_from_folder(extract_dir)
    # One X-ray per patient: MCUCXR_<patient>_<label>.png
    groups = [f"montgomery-{os.path.splitext(os.path.basename(p))[0].rsplit('_', 1)[0]}" for p in selected]
    logger.info(f"Collected {len(selected)} TB X-rays")
    return selected, groups


def try_kaggle_extras(raw_dir: str):
    """Optional COVID-QU-Ex / TB Kaggle sets if credentials exist."""
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        logger.info("Kaggle credentials not set; skipping Kaggle COVID-QU-Ex / TB datasets.")
        return [], []
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        covid_dir = os.path.join(raw_dir, "kaggle_covid")
        tb_dir = os.path.join(raw_dir, "kaggle_tb")
        os.makedirs(covid_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        api.dataset_download_files("tawsifurrahman/covid19-radiography-database", path=covid_dir, unzip=True)
        api.dataset_download_files("tawsifurrahman/tuberculosis-tb-chest-xray-dataset", path=tb_dir, unzip=True)
        return collect_from_folder(covid_dir), collect_from_folder(tb_dir)
    except Exception as e:
        logger.warning(f"Kaggle extra download failed: {e}")
        return [], []


def gather_binary_images(processed_dir: str):
    images, labels = [], []
    for split in ("train", "val", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            folder = os.path.join(processed_dir, split, cls)
            for path in collect_from_folder(folder):
                images.append(path)
                labels.append(cls)
    return images, labels


def stratified_copy(all_images, all_labels, groups, dest_root: str, seed: int):
    splits = stratified_group_split(all_images, all_labels, groups, seed=seed)
    assert_no_file_overlap(splits)

    if os.path.isdir(dest_root):
        remove_tree(dest_root)

    for split_name, (imgs, lbls) in splits.items():
        counts = defaultdict(int)
        for src, label in zip(imgs, lbls):
            out_dir = os.path.join(dest_root, split_name, label)
            os.makedirs(out_dir, exist_ok=True)
            dest_name = f"{split_name}_{counts[label]:05d}_{os.path.basename(src)}"
            shutil.copy2(src, os.path.join(out_dir, dest_name))
            counts[label] += 1
        logger.info(f"Split [{split_name.upper()}]: {dict(counts)} total={len(imgs)}")
    return dest_root


def prepare_multiclass(seed: int = SEED):
    set_seed(seed)
    data_dir = os.path.abspath("./data")
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    processed_binary = extract_and_prepare(seed=seed)
    images, labels = gather_binary_images(processed_binary)
    groups = kermany_groups(images)

    covid_imgs, covid_groups = collect_covid_images(raw_dir)
    tb_imgs, tb_groups = collect_tb_images(raw_dir)
    kaggle_covid, kaggle_tb = try_kaggle_extras(raw_dir)
    covid_imgs = covid_imgs + kaggle_covid
    covid_groups = covid_groups + _one_group_per_file(kaggle_covid, "Kaggle COVID")
    tb_imgs = tb_imgs + kaggle_tb
    tb_groups = tb_groups + _one_group_per_file(kaggle_tb, "Kaggle TB")

    for label, paths, path_groups in (("COVID", covid_imgs, covid_groups), ("TB", tb_imgs, tb_groups)):
        images.extend(paths)
        labels.extend([label] * len(paths))
        groups.extend(path_groups)

    present = sorted(set(labels))
    logger.info(f"Multi-class inventory: { {c: labels.count(c) for c in present} }")

    if "COVID" not in present or "TB" not in present:
        logger.warning(
            "COVID and/or TB images were not downloaded. "
            "Place files in data/raw/extra/COVID and data/raw/extra/TB, then re-run. "
            "Training will use the classes that are available."
        )

    dest = os.path.join(data_dir, "processed_multiclass")
    if len(set(labels)) < 2:
        raise RuntimeError("Need at least two classes to build a dataset.")
    stratified_copy(images, labels, groups, dest, seed)
    logger.info(f"Preferred class order: {PREFERRED_CLASSES}")
    return dest


if __name__ == "__main__":
    prepare_multiclass()
