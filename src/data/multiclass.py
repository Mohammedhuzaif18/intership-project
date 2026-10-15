"""
Phase 3 data: merge Kermany, COVID-QU-Ex and the TB Chest X-ray Database into one dataset.

  python -m src.data.multiclass                       # downloads via Kaggle (env-var credentials)
  python -m src.data.multiclass --covidqu-dir D --tb-dir T   # or use folders you downloaded yourself

Classes: NORMAL, PNEUMONIA (= Kermany pneumonia + COVID-QU-Ex "Non-COVID infection"), COVID, TB.

Shortcut / leakage controls:
  * identical preprocessing (CLAHE pipeline) for every source;
  * NORMAL and PNEUMONIA get equal image counts from each source that has them, so the
    source cannot predict the class within those classes;
  * exact duplicates (SHA-256) removed; COVID-QU-Ex reuses Kermany images (resized to
    256x256 PNG), so near-duplicates of Kermany images (32x32 thumbnail correlation) are dropped from the
    other sources, and remaining near-duplicate clusters are kept inside one split;
  * split 80/10/10 stratified by class x source, grouped by patient (Kermany) or
    near-duplicate cluster.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import MULTICLASS_DISPLAY_NAMES, PREPROCESSED_DIR, RESULTS_DIR, SEED, SPLITS_DIR
from src.data import quality
from src.data.download import download_kaggle_dataset, kaggle_credentials_available
from src.data.preprocessing import load_image, preprocess_files, to_gray_uint8
from src.data.splits import assert_no_overlap, assign_group_splits, save_split_files
from src.utils import absolute, logger, rel, remove_tree, save_json

COVIDQU_SLUG = "anasmohammedtahir/covidqu"                       # CC BY-SA 4.0
TB_SLUG = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"    # (c) original authors, research use with citation
COVIDQU_MAP = {"COVID-19": "COVID", "Non-COVID": "PNEUMONIA", "Normal": "NORMAL"}
TB_MAP = {"Normal": "NORMAL", "Tuberculosis": "TB"}
SOURCE_PRIORITY = ["kermany", "covidqu", "tbdb"]   # on near-duplicates, the earlier source keeps the image
THUMB = 32                                         # thumbnail side for near-duplicate detection
NEAR_DUP_CORR = 0.98                               # thumbnail correlation treated as the same image
MIN_SOURCE_IMAGES = 100                            # smaller per-class source leftovers do not set the balancing count
SPLIT_COLUMNS = ["path", "preprocessed_path", "label", "source", "subtype", "patient_id", "group", "sha256"]
OUT_RESULTS = RESULTS_DIR / "multiclass"


# ---------- Collect ----------

def load_kermany() -> pd.DataFrame:
    """The cleaned Kermany table from Review-I (keeps patient ids and the removed-duplicate filtering)."""
    df = pd.concat(pd.read_csv(SPLITS_DIR / "binary" / f"{s}.csv") for s in ("train", "val", "test"))
    return df.drop(columns=["preprocessed_path"]).assign(source="kermany")


def collect_covidqu(root: Path) -> pd.DataFrame:
    """Lung Segmentation Data only: the Infection Segmentation set is a subset of the same images."""
    rows = []
    for p in sorted(root.rglob("*.png")):
        parts = p.parts
        if p.parent.name != "images" or "Lung Segmentation Data" not in parts:
            continue
        cls = p.parent.parent.name
        if cls in COVIDQU_MAP:
            rows.append({"path": str(p), "label": COVIDQU_MAP[cls], "subtype": cls.lower(), "source": "covidqu"})
    return pd.DataFrame(rows)


def collect_tb(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*.png")):
        cls = p.parent.name
        if cls in TB_MAP:
            rows.append({"path": str(p), "label": TB_MAP[cls], "subtype": cls.lower(), "source": "tbdb"})
    return pd.DataFrame(rows)


def verify_source(df: pd.DataFrame, name: str, expected) -> bool:
    found = set(df["label"]) if len(df) else set()
    ok = set(expected) <= found
    counts = df["label"].value_counts().to_dict() if len(df) else {}
    logger.info(f"{name}: {len(df)} images {counts} -> {'verified' if ok else 'MISSING classes ' + str(set(expected) - found)}")
    return ok


# ---------- Near-duplicates ----------

def thumbnail(path) -> np.ndarray:
    """32x32 grayscale thumbnail, mean-centred and unit-length, so a dot product is a correlation."""
    t = cv2.resize(to_gray_uint8(load_image(path)), (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)
    t = t.flatten() - t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def compute_thumbnails(paths, workers: int = 8) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.stack(list(pool.map(thumbnail, paths)))


def max_similarity(a: np.ndarray, b: np.ndarray, chunk: int = 1024):
    """For each row of a: (highest correlation with any row of b, index of that row)."""
    best_s = np.full(len(a), -1.0, dtype=np.float32)
    best_i = np.zeros(len(a), dtype=np.int64)
    for s in range(0, len(a), chunk):
        sim = a[s:s + chunk] @ b.T
        best_i[s:s + chunk] = sim.argmax(axis=1)
        best_s[s:s + chunk] = sim.max(axis=1)
    return best_s, best_i


def near_dup_clusters(t: np.ndarray, threshold: float, chunk: int = 1024) -> np.ndarray:
    """Union-find over all pairs with correlation >= threshold; returns a cluster id per image."""
    parent = np.arange(len(t))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, len(t), chunk):
        sim = t[s:s + chunk] @ t.T
        for i, j in zip(*np.nonzero(sim >= threshold)):
            i += s
            if i < j:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(len(t))])


def drop_cross_source_duplicates(df: pd.DataFrame, thumbs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """An image from a lower-priority source that correlates >= NEAR_DUP_CORR with a higher-priority one is dropped."""
    keep = np.ones(len(df), dtype=bool)
    reasons = np.array([""] * len(df), dtype=object)
    for k, src in enumerate(SOURCE_PRIORITY[1:], start=1):
        cand = np.flatnonzero((df["source"] == src).to_numpy())
        ref = np.flatnonzero(df["source"].isin(SOURCE_PRIORITY[:k]).to_numpy() & keep)
        if not len(cand) or not len(ref):
            continue
        sim, i = max_similarity(thumbs[cand], thumbs[ref])
        hit = sim >= NEAR_DUP_CORR
        keep[cand[hit]] = False
        ref_paths = df["path"].to_numpy()[ref[i[hit]]]
        reasons[cand[hit]] = [f"near-duplicate (thumbnail r>={NEAR_DUP_CORR}) of {p}" for p in ref_paths]
        logger.info(f"{src}: {hit.sum()} images are near-duplicates of {SOURCE_PRIORITY[:k]} images and are dropped")
    removed = df[~keep].assign(remove_reason=reasons[~keep])
    return df[keep].copy(), removed


# ---------- Balance ----------

def balance(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """
    For a class found in several sources, every source with at least MIN_SOURCE_IMAGES images
    of it contributes the same number (the smallest such count, at most `cap`), so the source
    cannot predict the class. Smaller leftovers (e.g. the few TB-database normals that are not
    copies of COVID-QU-Ex images) are kept whole instead of shrinking every other source to
    their size. Single-source classes take up to 2 x cap.
    """
    parts = []
    for label, g in df.groupby("label"):
        per_source = g["source"].value_counts()
        big = per_source[per_source >= MIN_SOURCE_IMAGES]
        if len(per_source) == 1:
            n = min(len(g), 2 * cap)
        else:
            n = min(int(big.min()), cap) if len(big) else int(per_source.max())
        for src, gs in g.groupby("source"):
            take = n if per_source[src] >= MIN_SOURCE_IMAGES else len(gs)
            parts.append(gs.sample(min(take, len(gs)), random_state=seed))
    return pd.concat(parts).sort_values("path").reset_index(drop=True)


# ---------- Main ----------

def main(covidqu_dir=None, tb_dir=None, cap: int = 1200):
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    if covidqu_dir is None:
        if not kaggle_credentials_available():
            msg = ("KAGGLE_USERNAME / KAGGLE_KEY are not set, so COVID-QU-Ex and the TB database cannot be "
                   "downloaded. Multi-class extension skipped; set the variables (or pass --covidqu-dir/--tb-dir) "
                   "and re-run.")
            logger.warning(msg)
            save_json({"status": "skipped", "reason": msg}, OUT_RESULTS / "status.json")
            return False
        covidqu_dir = download_kaggle_dataset(COVIDQU_SLUG, "covidqu")
        tb_dir = download_kaggle_dataset(TB_SLUG, "tb_database")
    if covidqu_dir is None:
        save_json({"status": "skipped", "reason": "COVID-QU-Ex download failed"}, OUT_RESULTS / "status.json")
        return False

    kermany = load_kermany()
    covidqu = collect_covidqu(Path(covidqu_dir))
    if not verify_source(covidqu, "COVID-QU-Ex", ["NORMAL", "PNEUMONIA", "COVID"]):
        raise RuntimeError("COVID-QU-Ex folder structure not recognised (expected Lung Segmentation Data/.../images)")
    tb = collect_tb(Path(tb_dir)) if tb_dir else pd.DataFrame(columns=covidqu.columns)
    use_tb = tb_dir is not None and verify_source(tb, "TB database", ["NORMAL", "TB"])
    if not use_tb:
        logger.warning("TB dataset not available or not verified: building the 3-class system.")
        tb = tb.iloc[0:0]

    # Exact duplicates and corrupt files in the new sources (Kermany was cleaned in Review-I)
    new = quality.mark_duplicates(quality.scan_images(pd.concat([covidqu, tb], ignore_index=True)
                                                      .assign(path=lambda d: d["path"].map(lambda p: rel_or_abs(p)))))
    removed = [new[new["remove_reason"] != ""]]
    new = new[new["remove_reason"] == ""]
    kermany_hashes = set(kermany["sha256"])
    exact_k = new["sha256"].isin(kermany_hashes)
    removed.append(new[exact_k].assign(remove_reason="exact duplicate of a Kermany image"))
    new = new[~exact_k]
    df = pd.concat([kermany, new], ignore_index=True)

    logger.info(f"Computing {THUMB}x{THUMB} thumbnails for near-duplicate checks on {len(df)} images ...")
    thumbs = compute_thumbnails([absolute(p) for p in df["path"]])
    df, near = drop_cross_source_duplicates(df, thumbs)
    removed.append(near)
    df["row_id"] = df.index

    before = df.groupby(["label", "source"]).size().rename("available")
    df = balance(df, cap, SEED)
    after = df.groupby(["label", "source"]).size().rename("selected")
    composition = pd.concat([before, after], axis=1).fillna(0).astype(int)
    composition.to_csv(OUT_RESULTS / "dataset_composition.csv")
    logger.info(f"Composition (available -> selected):\n{composition.to_string()}")

    # Groups: Kermany patients, merged with near-duplicate clusters so look-alikes share a split
    clusters = near_dup_clusters(thumbs[df["row_id"].to_numpy()], NEAR_DUP_CORR)
    base_group = df["patient_id"].where(df["patient_id"].notna(), "file:" + df["path"]).to_numpy()
    df["group"] = _merge_groups(base_group, clusters)
    df["strat"] = df["label"] + "|" + df["source"]
    df["split"] = assign_group_splits(df, strat_col="strat", group_col="group", seed=SEED)
    assert_no_overlap(df, "sha256")
    assert_no_overlap(df.assign(cluster=clusters), "cluster")

    # Identical preprocessing for every source
    for src in df["source"].unique():
        remove_tree(PREPROCESSED_DIR / f"multiclass_{src}")
    df["preprocessed_path"] = [
        rel(PREPROCESSED_DIR / f"multiclass_{r.source}" / r.label / f"{i:05d}_{Path(r.path).stem}.png")
        for i, r in enumerate(df.itertuples())
    ]
    preprocess_files(zip(map(absolute, df["path"]), map(absolute, df["preprocessed_path"])))
    save_split_files(df, SPLITS_DIR / "multiclass", SPLIT_COLUMNS)
    pd.concat(removed)[["path", "label", "source", "remove_reason"]].to_csv(
        SPLITS_DIR / "multiclass" / "removed.csv", index=False)

    # Cross-dataset set: every selected COVID-QU-Ex Normal / Non-COVID image (never seen by the
    # Kermany-only binary model, and not a near-duplicate of any Kermany image)
    cross = df[(df["source"] == "covidqu") & df["label"].isin(["NORMAL", "PNEUMONIA"])]
    (SPLITS_DIR / "cross_dataset").mkdir(parents=True, exist_ok=True)
    cross[SPLIT_COLUMNS + ["split"]].to_csv(SPLITS_DIR / "cross_dataset" / "covidqu_normal_pneumonia.csv", index=False)

    counts = df.pivot_table(index=["split"], columns="label", values="path", aggfunc="count").reindex(
        ["train", "val", "test"])
    counts.to_csv(OUT_RESULTS / "split_counts.csv")
    classes = [c for c in MULTICLASS_DISPLAY_NAMES if c in set(df["label"])]
    quality.plot_class_counts(df, classes, OUT_RESULTS / "class_counts.png")
    quality.plot_split_counts(df, classes, OUT_RESULTS / "split_counts.png")
    save_json({
        "status": "ready", "classes": classes, "tb_included": bool(use_tb), "per_source_cap": cap,
        "near_duplicate_threshold_corr": NEAR_DUP_CORR,
        "removed": pd.concat(removed)["remove_reason"].str.extract(
            r"^(corrupt|exact duplicate|duplicate with conflicting labels|near-duplicate)")[0].value_counts().to_dict(),
        "composition": composition.reset_index().to_dict(orient="records"),
        "split_counts": counts.reset_index().to_dict(orient="records"),
        "datasets": {"COVID-QU-Ex": f"kaggle {COVIDQU_SLUG} (CC BY-SA 4.0)",
                     "TB": f"kaggle {TB_SLUG} (data (c) original authors; cite Rahman et al., IEEE Access 2020)"
                     if use_tb else "not used"},
    }, OUT_RESULTS / "status.json")
    logger.info(f"Multi-class dataset ready: {classes}, {len(df)} images")
    return True


def rel_or_abs(p) -> str:
    """Project-relative path when the file lives inside the project, else absolute."""
    try:
        return rel(p)
    except ValueError:
        return str(Path(p).resolve())


def _merge_groups(base: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Union of 'same patient' and 'same near-duplicate cluster' into one group id."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b, c in zip(base, clusters):
        rb, rc = find(f"g:{b}"), find(f"c:{c}")
        if rb != rc:
            parent[rc] = rb
    return np.array([find(f"g:{b}") for b in base])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--covidqu-dir", type=Path, help="Extracted COVID-QU-Ex folder (skips the Kaggle download)")
    parser.add_argument("--tb-dir", type=Path, help="Extracted TB Chest X-ray Database folder")
    parser.add_argument("--per-source-cap", type=int, default=1200)
    a = parser.parse_args()
    main(a.covidqu_dir, a.tb_dir, a.per_source_cap)
