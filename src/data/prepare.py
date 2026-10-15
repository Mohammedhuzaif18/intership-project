"""
Core dataset preparation (Review-I, steps 2-4):
download -> folder tree and counts -> corrupt / duplicate removal -> EDA figures
-> patient-level stratified 80/10/10 split -> CLAHE preprocessing -> split files.

Run: python -m src.data.prepare
"""
import pandas as pd

from src.config import BINARY_CLASSES, IMAGE_EXTS, KERMANY_DIR, PREPROCESSED_DIR, RESULTS_DIR, SEED, SPLITS_DIR
from src.data import quality
from src.data.download import download_kermany, print_tree_with_counts
from src.data.preprocessing import preprocess_files
from src.data.splits import assert_no_overlap, assign_group_splits, kermany_patient_id, save_split_files
from src.utils import absolute, logger, rel, remove_tree, save_json

EDA_DIR = RESULTS_DIR / "eda"
SPLIT_COLUMNS = ["path", "preprocessed_path", "label", "subtype", "patient_id", "source", "orig_folder",
                 "width", "height", "sha256"]


def collect_kermany() -> pd.DataFrame:
    """All images from the provided train/val/test folders, merged into one table."""
    rows = []
    for p in sorted(KERMANY_DIR.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")):
            continue
        label = p.parent.name.upper()
        if label not in BINARY_CLASSES:
            continue
        name = p.name.lower()
        if label == "NORMAL":
            subtype = "normal"
        else:
            subtype = "bacteria" if "bacteria" in name else "virus" if "virus" in name else "unknown"
        rows.append({
            "path": rel(p), "label": label, "subtype": subtype, "orig_folder": p.parent.parent.name,
            "source": "kermany", "patient_id": kermany_patient_id(p),
        })
    return pd.DataFrame(rows)


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download and inspect
    download_kermany()
    folder_counts = print_tree_with_counts(KERMANY_DIR)
    pd.Series(folder_counts, name="images").rename_axis("folder").to_csv(EDA_DIR / "raw_folder_counts.csv")

    df = collect_kermany()
    logger.info(f"Collected {len(df)} images from {sorted(df['orig_folder'].unique())} "
                f"(merged): {df['label'].value_counts().to_dict()}")

    # 2. Quality checks: corrupt files and exact duplicates by SHA-256
    df = quality.mark_duplicates(quality.scan_images(df))
    removed = df[df["remove_reason"] != ""]
    clean = df[df["remove_reason"] == ""].copy()
    SPLITS_DIR.joinpath("binary").mkdir(parents=True, exist_ok=True)
    removed[["path", "label", "sha256", "remove_reason"]].to_csv(SPLITS_DIR / "binary" / "removed.csv", index=False)
    qc = {
        "images_found": len(df),
        "corrupt_removed": int(df["corrupt"].sum()),
        "exact_duplicates_removed": int(removed["remove_reason"].str.startswith("exact duplicate").sum()),
        "conflicting_duplicates_removed": int((removed["remove_reason"] == "duplicate with conflicting labels").sum()),
        "images_kept": len(clean),
        "image_modes": clean["mode"].value_counts().to_dict(),
    }
    logger.info(f"Quality check: {qc}")

    # 3. Pneumonia subtypes from filenames
    subtype_counts = clean.groupby(["label", "subtype"]).size()
    logger.info(f"Subtypes: {subtype_counts.to_dict()}")

    # 4. Patient-level split, stratified by subtype so bacterial/viral shares match across splits
    unparsed = clean["patient_id"].isna().sum()
    clean["group"] = clean["patient_id"].fillna("file:" + clean["path"])
    logger.info(f"Patient grouping: {clean['group'].nunique()} patients for {len(clean)} images; "
                f"{unparsed} filenames without a parseable patient id (each treated as its own patient)")
    clean["split"] = assign_group_splits(clean, strat_col="subtype", group_col="group", seed=SEED)
    assert_no_overlap(clean, "sha256")

    # 5. CLAHE preprocessing into data/preprocessed/kermany (rebuilt from scratch each run)
    out_root = PREPROCESSED_DIR / "kermany"
    remove_tree(out_root)
    clean["preprocessed_path"] = [
        rel(out_root / r.orig_folder / r.label / (absolute(r.path).stem + ".png")) for r in clean.itertuples()
    ]
    preprocess_files(zip(map(absolute, clean["path"]), map(absolute, clean["preprocessed_path"])))

    save_split_files(clean, SPLITS_DIR / "binary", SPLIT_COLUMNS)

    # 6. Counts per split
    split_counts = clean.pivot_table(index="split", columns="label", values="path", aggfunc="count").reindex(
        ["train", "val", "test"])
    split_counts["total"] = split_counts.sum(axis=1)
    split_counts["patients"] = clean.groupby("split")["group"].nunique()
    subtype_split = clean.pivot_table(index="split", columns="subtype", values="path", aggfunc="count").reindex(
        ["train", "val", "test"])
    split_counts = split_counts.join(subtype_split)
    split_counts.to_csv(RESULTS_DIR / "split_counts.csv")
    logger.info(f"Class counts per split:\n{split_counts.to_string()}")

    # 7. EDA figures
    quality.plot_class_counts(clean, BINARY_CLASSES, EDA_DIR / "class_counts.png")
    quality.plot_split_counts(clean, BINARY_CLASSES, EDA_DIR / "split_counts.png")
    quality.plot_subtypes(clean, EDA_DIR / "pneumonia_subtypes.png")
    quality.plot_samples(clean, BINARY_CLASSES, EDA_DIR / "samples_per_class.png")
    quality.image_size_stats(clean, BINARY_CLASSES, EDA_DIR / "image_size_stats.csv",
                             EDA_DIR / "image_sizes.png")
    quality.plot_intensity_histograms(clean, BINARY_CLASSES, EDA_DIR / "intensity_histograms.png")
    quality.plot_preprocessing_examples(clean, BINARY_CLASSES, RESULTS_DIR / "preprocessing_before_after.png")
    train_example = clean[(clean["split"] == "train") & (clean["label"] == "PNEUMONIA")]["preprocessed_path"].iloc[0]
    quality.plot_augmentation_examples(train_example, RESULTS_DIR / "augmentation_examples.png")

    save_json({
        "quality_check": qc,
        "subtype_counts": {f"{k[0]}/{k[1]}": int(v) for k, v in subtype_counts.items()},
        "patients": int(clean["group"].nunique()),
        "unparsed_patient_ids": int(unparsed),
        "split_counts": split_counts.reset_index().to_dict(orient="records"),
        "image_size_summary": {
            "width_median": float(clean["width"].median()), "height_median": float(clean["height"].median()),
            "width_range": [int(clean["width"].min()), int(clean["width"].max())],
            "height_range": [int(clean["height"].min()), int(clean["height"].max())],
        },
    }, EDA_DIR / "data_summary.json")
    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
