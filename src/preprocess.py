import os
import glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from src.utils import logger, remove_tree
from src.config import IMAGE_SIZE

TARGET_SIZE = IMAGE_SIZE


def process_single_image(args):
    src_path, dst_path = args
    try:
        with Image.open(src_path) as img:
            rgb_img = img.convert("RGB")
            resized = rgb_img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            resized.save(dst_path, "JPEG", quality=95)
    except Exception as e:
        logger.error(f"Error processing {src_path}: {e}")


def _class_folders(split_dir: str):
    if not os.path.isdir(split_dir):
        return []
    return [
        name for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    ]


def preprocess_all(data_dir: str = None, out_dir: str = "./data/preprocessed_224"):
    if data_dir is None:
        if os.path.isdir("./data/processed_multiclass"):
            data_dir = "./data/processed_multiclass"
        else:
            data_dir = "./data/processed"

    if os.path.abspath(out_dir) == os.path.abspath(data_dir):
        raise ValueError("out_dir must differ from data_dir; out_dir is deleted before resizing.")
    # Files left from an earlier split would sit next to the new ones (different
    # names) and put the same X-ray in more than one split.
    if os.path.isdir(out_dir):
        logger.info(f"Removing previous {out_dir} ...")
        remove_tree(out_dir)

    logger.info(f"Starting multi-threaded 224x224 RGB resize from {data_dir} ...")
    tasks = []

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        for cls_name in _class_folders(split_dir):
            src_folder = os.path.join(split_dir, cls_name)
            dst_folder = os.path.join(out_dir, split, cls_name)
            os.makedirs(dst_folder, exist_ok=True)
            files = glob.glob(os.path.join(src_folder, "*.*"))
            for f in files:
                fname = os.path.basename(f)
                out_name = os.path.splitext(fname)[0] + ".jpg"
                tasks.append((f, os.path.join(dst_folder, out_name)))

    logger.info(f"Total images to resize: {len(tasks)}")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_single_image, tasks))

    logger.info("Pre-resizing completed successfully.")


if __name__ == "__main__":
    preprocess_all()
