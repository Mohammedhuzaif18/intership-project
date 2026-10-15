import os
import shutil
import zipfile
import glob
from collections import defaultdict
from huggingface_hub import hf_hub_download
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import set_seed, logger, remove_tree
from src.config import SEED
from src.splits import kermany_groups, stratified_group_split, assert_no_file_overlap

def extract_and_prepare(seed: int = SEED):
    set_seed(seed)
    
    data_dir = os.path.abspath("./data")
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Check if zip exists in cache or download
    logger.info("Locating or downloading chest_xray.zip...")
    zip_path = hf_hub_download(repo_id="AiresPucrs/chest-xray", filename="chest_xray.zip", repo_type="dataset")
    logger.info(f"Dataset zip located at: {zip_path}")
    
    # Extract
    extract_target = os.path.join(raw_dir, "extracted")
    if not os.path.exists(extract_target) or len(os.listdir(extract_target)) == 0:
        logger.info(f"Extracting zip to {extract_target}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_target)
        logger.info("Extraction complete.")
    else:
        logger.info("Extracted data already present.")
        
    # Print extracted folder tree and initial counts
    logger.info("--- Extracted Folder Tree and Raw Image Counts ---")
    raw_counts = defaultdict(int)
    for root, dirs, files in os.walk(extract_target):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            rel_path = os.path.relpath(root, extract_target)
            logger.info(f"  Folder: {rel_path} -> {len(image_files)} images")
            category = os.path.basename(root).upper()
            if "NORMAL" in category:
                raw_counts["NORMAL"] += len(image_files)
            elif "PNEUMONIA" in category:
                raw_counts["PNEUMONIA"] += len(image_files)
                
    logger.info(f"Total raw images detected: {raw_counts}")

    # Gather all images by class across train, val, and test folders
    all_images = []
    all_labels = []
    
    for root, dirs, files in os.walk(extract_target):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._'):
                full_path = os.path.join(root, f)
                category = os.path.basename(root).upper()
                if "NORMAL" in category:
                    all_images.append(full_path)
                    all_labels.append("NORMAL")
                elif "PNEUMONIA" in category:
                    all_images.append(full_path)
                    all_labels.append("PNEUMONIA")

    logger.info(f"Total valid collected images: {len(all_images)} (Normal: {all_labels.count('NORMAL')}, Pneumonia: {all_labels.count('PNEUMONIA')})")

    # Stratified 80 / 10 / 10 split by patient: a patient's X-rays all go to one split
    groups = kermany_groups(all_images)
    logger.info(f"Unique patients: {len(set(groups))}")
    splits = stratified_group_split(all_images, all_labels, groups, seed=seed)
    assert_no_file_overlap(splits)

    # Files from an earlier split would otherwise remain next to the new ones
    if os.path.isdir(processed_dir):
        logger.info(f"Removing previous split at {processed_dir} ...")
        remove_tree(processed_dir)

    logger.info("Copying images to stratified destination folders...")
    for split_name, (imgs, lbls) in splits.items():
        split_dir = os.path.join(processed_dir, split_name)
        for cls_name in ["NORMAL", "PNEUMONIA"]:
            os.makedirs(os.path.join(split_dir, cls_name), exist_ok=True)
            
        cls_count = defaultdict(int)
        for src_path, label in zip(imgs, lbls):
            fname = os.path.basename(src_path)
            # Prefix with original parent folder to prevent any filename collision
            parent = os.path.basename(os.path.dirname(os.path.dirname(src_path)))
            dest_name = f"{parent}_{fname}" if parent else fname
            dest_path = os.path.join(split_dir, label, dest_name)
            shutil.copy2(src_path, dest_path)
            cls_count[label] += 1
            
        logger.info(f"Split [{split_name.upper()}]: Total = {len(imgs)} | NORMAL = {cls_count['NORMAL']} | PNEUMONIA = {cls_count['PNEUMONIA']}")

    return processed_dir

if __name__ == "__main__":
    extract_and_prepare()
