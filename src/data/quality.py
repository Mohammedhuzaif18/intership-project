"""Quality checks (corrupt files, exact duplicates) and exploratory figures."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import DISPLAY_NAMES, PREPROCESS_VARIANT
from src.data.preprocessing import fit_to_input, load_image, preprocess_gray, to_gray_uint8
from src.plotting import CLASS_COLORS, SERIES, TEXT_2, plt, save
from src.utils import absolute, logger, sha256_file

MIN_SIDE = 64


def _inspect(rel_path: str) -> dict:
    path = absolute(rel_path)
    info = {"path": rel_path, "corrupt": False, "error": "", "width": np.nan, "height": np.nan, "mode": ""}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            info.update(width=im.width, height=im.height, mode=im.mode)
        if min(info["width"], info["height"]) < MIN_SIDE:
            info.update(corrupt=True, error=f"smaller than {MIN_SIDE}px")
    except Exception as e:
        info.update(corrupt=True, error=f"{type(e).__name__}: {e}")
    try:
        info["sha256"] = sha256_file(path)
    except Exception as e:
        info.update(corrupt=True, error=f"unreadable: {e}", sha256="")
    return info


def scan_images(df: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    """Adds width, height, mode, sha256 and corrupt/error columns."""
    logger.info(f"Scanning {len(df)} files (decode check + SHA-256) ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_inspect, df["path"]))
    return df.merge(pd.DataFrame(rows), on="path", how="left")


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact duplicates by SHA-256 among non-corrupt files. Within a group of identical
    files the first path (sorted) is kept when all labels agree; if labels disagree,
    every copy is dropped because the true label is unknown.
    """
    df = df.copy()
    df["remove_reason"] = np.where(df["corrupt"], "corrupt: " + df["error"].astype(str), "")
    ok = df[~df["corrupt"]]
    for digest, group in ok.groupby("sha256"):
        if len(group) < 2:
            continue
        group = group.sort_values("path")
        if group["label"].nunique() > 1:
            df.loc[group.index, "remove_reason"] = "duplicate with conflicting labels"
        else:
            keep = group["path"].iloc[0]
            df.loc[group.index[1:], "remove_reason"] = f"exact duplicate of {keep}"
    return df


# ---------- Figures ----------

def plot_class_counts(df: pd.DataFrame, classes, path):
    counts = df["label"].value_counts().reindex(classes, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar([DISPLAY_NAMES.get(c, c) for c in classes], counts.values,
                  color=[CLASS_COLORS.get(c, SERIES[i]) for i, c in enumerate(classes)], width=0.55)
    ax.bar_label(bars, labels=[f"{v:,}  ({v / counts.sum():.0%})" for v in counts.values],
                 padding=4, color=TEXT_2, fontsize=9)
    ax.set_ylabel("Images")
    ax.set_title("Images per class (after cleaning)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, counts.max() * 1.15)
    save(fig, path)


def plot_split_counts(df: pd.DataFrame, classes, path):
    table = df.pivot_table(index="split", columns="label", values="path", aggfunc="count").reindex(
        ["train", "val", "test"])[classes]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(table.index))
    width = 0.36
    for i, cls in enumerate(classes):
        bars = ax.bar(x + (i - (len(classes) - 1) / 2) * width, table[cls].values, width - 0.03,
                      color=CLASS_COLORS.get(cls, SERIES[i]), label=DISPLAY_NAMES.get(cls, cls))
        ax.bar_label(bars, padding=3, color=TEXT_2, fontsize=8)
    ax.set_xticks(x, [s.capitalize() for s in table.index])
    ax.set_ylabel("Images")
    ax.set_title("Class counts per split (patient-level 80/10/10)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    save(fig, path)


def plot_subtypes(df: pd.DataFrame, path):
    pn = df[df["label"] == "PNEUMONIA"]["subtype"].value_counts().reindex(["bacteria", "virus"], fill_value=0)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bars = ax.barh(["Bacterial", "Viral"], pn.values, color=[SERIES[1], SERIES[4]], height=0.5)
    ax.bar_label(bars, labels=[f"{v:,}  ({v / pn.sum():.0%})" for v in pn.values], padding=4,
                 color=TEXT_2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Images")
    ax.set_title("Pneumonia subtypes (from filenames)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, pn.max() * 1.3)
    save(fig, path)


def plot_samples(df: pd.DataFrame, classes, path, per_class: int = 8, seed: int = 42):
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.6, len(classes) * 1.9))
    axes = np.atleast_2d(axes)
    for r, cls in enumerate(classes):
        part = df[df["label"] == cls]
        subtypes = sorted(part["subtype"].unique())
        # Equal share per subtype (e.g. 4 bacterial + 4 viral) so the grid shows both
        picks = pd.concat([part[part["subtype"] == s].sample(per_class // len(subtypes), random_state=seed)
                           for s in subtypes])
        for c, (_, row) in enumerate(picks.iterrows()):
            ax = axes[r, c]
            ax.imshow(to_gray_uint8(load_image(absolute(row["path"]))), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            sub = row.get("subtype", "")
            ax.set_title(sub if sub and sub != "normal" else "", fontsize=8, color=TEXT_2)
        axes[r, 0].set_ylabel(DISPLAY_NAMES.get(cls, cls), fontsize=11, color="#0b0b0b")
    fig.suptitle("Random samples per class (raw images)", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def image_size_stats(df: pd.DataFrame, classes, csv_path, fig_path):
    d = df.assign(aspect=df["width"] / df["height"], megapixels=df["width"] * df["height"] / 1e6)
    stats = d.groupby("label")[["width", "height", "aspect", "megapixels"]].describe().T
    stats.to_csv(csv_path)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for i, cls in enumerate(classes):
        part = d[d["label"] == cls]
        ax.scatter(part["width"], part["height"], s=8, alpha=0.35, color=CLASS_COLORS.get(cls, SERIES[i]),
                   label=f"{DISPLAY_NAMES.get(cls, cls)} (n={len(part):,})", linewidths=0)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Native image sizes")
    ax.legend(markerscale=2.5)
    save(fig, fig_path)
    return stats


def plot_intensity_histograms(df: pd.DataFrame, classes, path, n: int = 300, seed: int = 42):
    """Mean grayscale histogram per class, before and after CLAHE (both at 224x224)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    bins = np.arange(257)
    for i, cls in enumerate(classes):
        picks = df[df["label"] == cls].sample(min(n, (df["label"] == cls).sum()), random_state=seed)
        raw_h, pre_h = np.zeros(256), np.zeros(256)
        for p in picks["path"]:
            img = load_image(absolute(p))
            raw_h += np.histogram(fit_to_input(to_gray_uint8(img)), bins=bins, density=True)[0]
            pre_h += np.histogram(preprocess_gray(img), bins=bins, density=True)[0]
        color = CLASS_COLORS.get(cls, SERIES[i])
        label = DISPLAY_NAMES.get(cls, cls)
        axes[0].plot(bins[:-1], raw_h / len(picks), color=color, label=label)
        axes[1].plot(bins[:-1], pre_h / len(picks), color=color, label=label)
    axes[0].set_title("Pixel intensities: raw (resized)")
    axes[1].set_title("Pixel intensities: after CLAHE")
    for ax in axes:
        ax.set_xlabel("Intensity (0-255)")
        ax.set_xlim(0, 255)
    axes[0].set_ylabel("Density (mean over images)")
    axes[1].legend()
    fig.tight_layout()
    save(fig, path)


def plot_preprocessing_examples(df: pd.DataFrame, classes, path, per_class: int = 2, seed: int = 7):
    """Raw -> resized -> resized + CLAHE for a few images per class."""
    rows = pd.concat([df[df["label"] == c].sample(per_class, random_state=seed) for c in classes])
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.2, 2.5 * len(rows)))
    fit = {"squash": "Grayscale, 224x224", "crop": "Centre square, 224x224",
           "lungbox": "Lung box, 224x224"}[PREPROCESS_VARIANT]
    titles = ["Raw (native size)", fit, "+ CLAHE (model input)"]
    for r, (_, row) in enumerate(rows.iterrows()):
        img = load_image(absolute(row["path"]))
        gray = to_gray_uint8(img)
        panels = [gray, fit_to_input(gray), preprocess_gray(img)]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        axes[r, 0].set_ylabel(f"{DISPLAY_NAMES.get(row['label'], row['label'])}\n{gray.shape[1]}x{gray.shape[0]}",
                              fontsize=9, color="#0b0b0b")
    fig.suptitle("Preprocessing: before and after", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def plot_augmentation_examples(rel_preprocessed_path: str, path, n: int = 7, seed: int = 42):
    """One preprocessed training image and n random training-time augmentations of it."""
    import tensorflow as tf

    from src.data.loaders import get_augmentation

    img = np.asarray(Image.open(absolute(rel_preprocessed_path)).convert("RGB"), dtype=np.float32)
    aug = get_augmentation(seed=seed)
    batch = tf.repeat(tf.convert_to_tensor(img)[None], n, axis=0)
    out = aug(batch, training=True).numpy()
    fig, axes = plt.subplots(1, n + 1, figsize=(1.9 * (n + 1), 2.3))
    for i, ax in enumerate(axes):
        ax.imshow((img if i == 0 else out[i - 1]).astype(np.uint8))
        ax.set_title("Preprocessed" if i == 0 else f"Augmented {i}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Training-only augmentation: rotation 15°, shift 0.1, zoom 0.1, h-flip, brightness/contrast",
                 fontsize=10)
    fig.tight_layout()
    save(fig, path)
