"""
Centre-crop ablation (follow-up to Review-II): does removing the aspect-ratio squash change the results?

The default pipeline squashes every X-ray to 224x224; XRAY_PREPROCESS=crop takes the central square
first. Both binary pipelines use the same patient-level splits (seed 42). This script compares:

  1. held-out Kermany test metrics, paired on the same images (McNemar, patient-level bootstrap);
  2. size-only probes: native width / height / aspect vs the crop side, and the lung framing that
     each model input shows (from lung masks);
  3. how much Grad-CAM heat falls inside the lungs. COVID-QU-Ex ships lung masks, and its copies of
     Kermany images (matched in src.data.multiclass) give exact masks for most Kermany test images;
     every Kermany test image also gets a central-box approximation;
  4. accuracy on the unseen COVID-QU-Ex Normal / Non-COVID images (cross-dataset).

  XRAY_PREPROCESS=crop python -m src.data.prepare
  XRAY_PREPROCESS=crop python -m src.train --task binary
  python -m src.ablation_crop                      # -> results/ablation_crop/comparison

Reads both pipelines' outputs and writes only to results/ablation_crop/comparison.
"""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.stats import binomtest, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import BINARY_CLASSES, IMAGE_SIZE, ROOT, SEED, variant_subdir
from src.data.preprocessing import center_square, fit_to_input, load_image, preprocess_image
from src.evaluate import compute_metrics
from src.gradcam import GradCAM, overlay_heatmap
from src.utils import absolute, load_json, logger, save_json, set_seed

VARIANTS = ("squash", "crop")
MODELS = {"ResNet50": "resnet50", "Baseline CNN": "baseline_cnn"}
OUT = ROOT / "results" / variant_subdir("crop") / "comparison"
CROSS_CSV = ROOT / "splits" / "cross_dataset" / "covidqu_normal_pneumonia.csv"
CENTRAL_BOX = (0.15, 0.85, 0.15, 0.85)  # x0, x1, y0, y1 as fractions of the native image
BORDER = 0.10                           # outer band of the model input, as a fraction of its side
N_BOOT = 2000
BATCH = 32
REGIONS = ("lung", "box", "border")


def variant_dir(kind: str, variant: str):
    return ROOT / kind / variant_subdir(variant)


# ---------- Statistics ----------

def wilson(k: int, n: int, z: float = 1.96) -> list:
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [centre - half, centre + half]


def bootstrap_ci(groups, stat, n_boot: int = N_BOOT, seed: int = SEED) -> list:
    """95% percentile interval of stat(indices) over resamples of whole patients (or near-duplicate clusters)."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    vals = [stat(np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))]))
            for _ in range(n_boot)]
    return np.nanpercentile(vals, [2.5, 97.5]).tolist()


def _auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def fmt_ci(ci, pct: bool = False) -> str:
    return f"{100 * ci[0]:.1f}-{100 * ci[1]:.1f}%" if pct else f"{ci[0]:+.3f} to {ci[1]:+.3f}"


def markdown(df: pd.DataFrame, path, title: str):
    df.to_csv(path.with_suffix(".csv"), index=False)
    with open(path.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n" + df.to_markdown(index=False, floatfmt=".4f") + "\n")


# ---------- 1. Test metrics ----------

def test_metrics() -> dict:
    rows, paired = [], []
    for name, slug in MODELS.items():
        preds = {v: pd.read_csv(variant_dir("results", v) / f"test_predictions_{slug}.csv") for v in VARIANTS}
        assert preds["squash"]["path"].tolist() == preds["crop"]["path"].tolist(), "test sets differ"
        y = (preds["squash"]["label"] == "PNEUMONIA").to_numpy().astype(int)
        probs = {v: d[["p_NORMAL", "p_PNEUMONIA"]].to_numpy() for v, d in preds.items()}
        right = {v: (d["true"] == d["pred"]).to_numpy() for v, d in preds.items()}
        for v in VARIANTS:
            m = compute_metrics(y, probs[v], BINARY_CLASSES)
            rows.append({"Model": name, "Preprocessing": v, "n": m["n"], "Accuracy": m["accuracy"],
                         "Accuracy 95% CI": fmt_ci(wilson(int(right[v].sum()), len(y)), pct=True),
                         "Precision": m["precision"], "Recall": m["recall"], "Specificity": m["specificity"],
                         "F1": m["f1"], "ROC-AUC": m["roc_auc"], "FP": m["fp"], "FN": m["fn"]})
        only_squash = int((right["squash"] & ~right["crop"]).sum())
        only_crop = int((right["crop"] & ~right["squash"]).sum())
        groups = preds["squash"]["patient_id"].to_numpy()
        auc_diff = _auc(y, probs["crop"][:, 1]) - _auc(y, probs["squash"][:, 1])
        acc_diff = right["crop"].mean() - right["squash"].mean()
        paired.append({
            "Model": name, "Accuracy crop - squash": acc_diff,
            "Accuracy diff 95% CI": fmt_ci(bootstrap_ci(groups, lambda i: right["crop"][i].mean()
                                                        - right["squash"][i].mean())),
            "Only squash right": only_squash, "Only crop right": only_crop,
            "McNemar exact p": binomtest(min(only_squash, only_crop), only_squash + only_crop).pvalue
            if only_squash + only_crop else 1.0,
            "ROC-AUC crop - squash": auc_diff,
            "AUC diff 95% CI": fmt_ci(bootstrap_ci(groups, lambda i: _auc(y[i], probs["crop"][i, 1])
                                                   - _auc(y[i], probs["squash"][i, 1]))),
            "Prediction agreement": float((preds["squash"]["pred"] == preds["crop"]["pred"]).mean()),
        })
    rows, paired = pd.DataFrame(rows), pd.DataFrame(paired)
    markdown(rows, OUT / "test_metrics", "Held-out Kermany test set (n=583): squash vs centre-crop preprocessing")
    markdown(paired, OUT / "test_paired", "Paired comparison on the same test images (crop minus squash; "
                                         "CIs from 2,000 patient-level bootstrap resamples)")
    selected = {v: load_json(variant_dir("results", v) / "metrics.json")["selected"] for v in VARIANTS}
    return {"metrics": rows.to_dict(orient="records"), "paired": paired.to_dict(orient="records"),
            "selected_on_validation": selected}


# ---------- Lung masks ----------

def kermany_mask_paths() -> dict:
    """Kermany image -> lung mask of its COVID-QU-Ex copy (the same X-ray, resized by COVID-QU-Ex to 256x256)."""
    removed = pd.read_csv(ROOT / "splits" / "multiclass" / "removed.csv")
    kermany = removed["remove_reason"].str.extract(r"^near-duplicate .* of (data/raw/extracted/chest_xray/.+)$")[0]
    pairs = removed.assign(kermany=kermany).dropna(subset=["kermany"])
    masks = pairs["path"].str.replace("/images/", "/lung masks/", regex=False)
    return {k: m for k, m in zip(pairs["kermany"], masks) if absolute(m).exists()}


def native_lung_mask(mask_path, width: int, height: int) -> np.ndarray:
    mask = np.asarray(Image.open(absolute(mask_path)).convert("L")) > 127
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def native_box(width: int, height: int) -> np.ndarray:
    x0, x1, y0, y1 = CENTRAL_BOX
    box = np.zeros((height, width), dtype=bool)
    box[round(y0 * height):round(y1 * height), round(x0 * width):round(x1 * width)] = True
    return box


def to_input(native_mask: np.ndarray, variant: str) -> np.ndarray:
    """Bool mask at native size -> bool mask in the 224x224 model input, via the images' own geometric step."""
    return fit_to_input(native_mask.astype(np.uint8) * 255, variant) >= 128


def crop_retained(native_mask: np.ndarray) -> float:
    """Share of the mask that survives the centre-square crop."""
    left, top, side = center_square(native_mask.shape[1], native_mask.shape[0])
    total = native_mask.sum()
    return float(native_mask[top:top + side, left:left + side].sum() / total) if total else np.nan


def border_band() -> np.ndarray:
    b = round(BORDER * IMAGE_SIZE[0])
    band = np.ones(IMAGE_SIZE[::-1], dtype=bool)
    band[b:-b, b:-b] = False
    return band


BAND = border_band()


# ---------- 2. Size-only probes ----------

def _probe(train_x, train_y, test_x, test_y) -> dict:
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    clf.fit(train_x, train_y)
    p = clf.predict_proba(test_x)[:, 1]
    return {"test_roc_auc": roc_auc_score(test_y, p), "test_balanced_accuracy": balanced_accuracy_score(test_y, p >= 0.5)}


def _framing(m: np.ndarray) -> dict:
    """Where the lungs sit in a 224x224 model input."""
    ys, xs = np.nonzero(m)
    if not len(xs):
        return {}
    s = IMAGE_SIZE[0]
    return {"lung_area": float(m.mean()), "lung_x0": xs.min() / s, "lung_x1": (xs.max() + 1) / s,
            "lung_y0": ys.min() / s, "lung_y1": (ys.max() + 1) / s}


def _framing_one(args):
    path, mask_path, w, h = args
    native = native_lung_mask(mask_path, w, h)
    return {"path": path, "lung_retained_by_crop": crop_retained(native),
            **{f"{v}:{k}": val for v in VARIANTS for k, val in _framing(to_input(native, v)).items()}}


def size_probes(masks: dict) -> dict:
    split_dir = variant_dir("splits", "squash") / "binary"
    train, test = (pd.read_csv(split_dir / f"{s}.csv") for s in ("train", "test"))

    def geometry(df):
        w, h = df["width"].to_numpy(float), df["height"].to_numpy(float)
        return pd.DataFrame({"log_width": np.log(w), "log_height": np.log(h), "aspect": w / h,
                             "log_crop_side": np.log(np.minimum(w, h))})

    gtr, gte = geometry(train), geometry(test)
    ytr, yte = (train["label"] == "PNEUMONIA").astype(int), (test["label"] == "PNEUMONIA").astype(int)
    feature_sets = {
        "Native width + height + aspect ratio (Review-I probe)": ["log_width", "log_height", "aspect"],
        "Aspect ratio only (kept by squash as distorted anatomy, removed by crop)": ["aspect"],
        "Crop side only (native resolution of the centre square: all the size left after cropping)":
            ["log_crop_side"],
    }
    rows = [{"Probe features": k, "n train": len(train), "n test": len(test),
             **_probe(gtr[c], ytr, gte[c], yte)} for k, c in feature_sets.items()]

    # Lung framing inside each model input, for the images with a lung mask
    jobs = [(r.path, masks[r.path], int(r.width), int(r.height))
            for df in (train, test) for r in df.itertuples() if r.path in masks]
    logger.info(f"Lung framing for {len(jobs)} train/test images with masks ...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        framing = pd.DataFrame(list(pool.map(_framing_one, jobs)))
    labelled = pd.concat([train.assign(split="train"), test.assign(split="test")])[["path", "label", "split"]]
    framing = labelled.merge(framing, on="path").dropna()
    ftr, fte = framing[framing["split"] == "train"], framing[framing["split"] == "test"]
    keys = ["lung_area", "lung_x0", "lung_x1", "lung_y0", "lung_y1"]
    for v in VARIANTS:
        cols = [f"{v}:{k}" for k in keys]
        rows.append({"Probe features": f"Lung area + lung bounding box in the {v} model input (no pixels)",
                     "n train": len(ftr), "n test": len(fte),
                     **_probe(ftr[cols], ftr["label"] == "PNEUMONIA", fte[cols], fte["label"] == "PNEUMONIA")})
    probes = pd.DataFrame(rows)
    markdown(probes, OUT / "size_probe", "Size-only probes (logistic regression; train split -> test split)")

    # Per feature: class medians, and how well the feature alone separates the classes on the test images
    features = ["lung_retained_by_crop"] + [f"{v}:{k}" for k in keys for v in VARIANTS]
    medians = framing.groupby("label")[features].median().T
    single_auc = [max(a, 1 - a) for a in (roc_auc_score(fte["label"] == "PNEUMONIA", fte[f]) for f in features)]
    framing_summary = pd.DataFrame({"Feature (fraction of the model input side or area)": features,
                                    "Median Normal": medians["NORMAL"].to_numpy(),
                                    "Median Pneumonia": medians["PNEUMONIA"].to_numpy(),
                                    "Test ROC-AUC, this feature alone": single_auc})
    markdown(framing_summary, OUT / "lung_framing", "Lung framing in the model input (Kermany train + test images "
                                                    "with a COVID-QU-Ex lung mask; x0/x1 = left/right, "
                                                    "y0/y1 = top/bottom edge of the lungs)")
    return {"probes": probes.to_dict(orient="records"), "framing_medians": framing_summary.to_dict(orient="records"),
            "framing_n": len(framing)}


# ---------- 3./4. Grad-CAM heat and cross-dataset accuracy ----------

def batched_cam(cam: GradCAM, images: np.ndarray):
    """GradCAM.__call__ for a batch (binary, predicted class): (p_pneumonia, heatmaps in [0, 1])."""
    x = tf.convert_to_tensor(images.astype("float32"))
    with tf.GradientTape() as tape:
        fmap, preds = cam.grad_model(x, training=False)
        p = preds[:, 0]
        score = tf.where(p >= 0.5, p, 1.0 - p)
    # The tape differentiates the sum of the scores; each score depends only on its own image's
    # feature map, so this gives every image its own gradient
    grads = tape.gradient(score, fmap)
    maps = tf.nn.relu(tf.einsum("bhwc,bc->bhw", fmap, tf.reduce_mean(grads, axis=(1, 2)))).numpy()
    peak = maps.max(axis=(1, 2), keepdims=True)
    return p.numpy(), np.divide(maps, peak, out=np.zeros_like(maps), where=peak > 0)


def _items(dataset: str, masks: dict) -> pd.DataFrame:
    if dataset == "Kermany test":
        df = pd.read_csv(variant_dir("splits", "squash") / "binary" / "test.csv")
        crop = pd.read_csv(variant_dir("splits", "crop") / "binary" / "test.csv")
        return df.assign(group=df["patient_id"], mask=df["path"].map(masks), stored_squash=df["preprocessed_path"],
                         stored_crop=df["path"].map(dict(zip(crop["path"], crop["preprocessed_path"]))),
                         box=True)
    df = pd.read_csv(CROSS_CSV)
    sizes = [Image.open(absolute(p)).size for p in df["path"]]
    # COVID-QU-Ex images are 256x256 squares, so both variants give the same input (checked below)
    return df.assign(width=[s[0] for s in sizes], height=[s[1] for s in sizes],
                     mask=df["path"].str.replace("/images/", "/lung masks/", regex=False),
                     stored_squash=df["preprocessed_path"], stored_crop=None, box=False)


def _dedupe(per_variant: dict) -> dict:
    """Share one array when both variants give the same result (square images), to save memory."""
    a, b = (per_variant[v] for v in VARIANTS)
    if a is not None and b is not None and np.array_equal(a, b):
        per_variant = dict.fromkeys(VARIANTS, a)
    return per_variant


def _prepare_one(r):
    """Model inputs (single-channel; the model sees it repeated to RGB) and regions for one image, per variant."""
    img = load_image(absolute(r.path))
    inputs = _dedupe({v: preprocess_image(img, v)[..., 0] for v in VARIANTS})
    for v in VARIANTS:
        stored = getattr(r, f"stored_{v}")
        if isinstance(stored, str):
            same = np.array_equal(inputs[v], np.asarray(Image.open(absolute(stored))))
            assert same, f"{r.path}: on-the-fly {v} preprocessing differs from the stored training image"
    w, h = int(r.width), int(r.height)
    lung = native_lung_mask(r.mask, w, h) if isinstance(r.mask, str) else None
    box = native_box(w, h) if r.box else None
    lungs = _dedupe({v: to_input(lung, v) if lung is not None else None for v in VARIANTS})
    boxes = _dedupe({v: to_input(box, v) if box is not None else None for v in VARIANTS})
    return {"input": inputs, "retained": crop_retained(lung) if lung is not None else np.nan,
            "regions": {v: {"lung": lungs[v], "box": boxes[v], "border": BAND} for v in VARIANTS}}


def _shares(heat: np.ndarray, regions: dict) -> dict:
    total = heat.sum()
    row = {}
    for k in REGIONS:
        r = regions[k]
        row[f"heat_{k}"] = float((heat * r).sum() / total) if r is not None and total > 0 else np.nan
        row[f"area_{k}"] = float(r.mean()) if r is not None else np.nan
    return row


def gradcam_analysis(masks: dict):
    models = {(name, v): tf.keras.models.load_model(variant_dir("models", v) / "binary" / f"{slug}.keras")
              for name, slug in MODELS.items() for v in VARIANTS}
    cams = {k: GradCAM(m) for k, m in models.items()}
    rows, keep = [], {}
    for dataset in ("Kermany test", "COVID-QU-Ex unseen"):
        items = _items(dataset, masks)
        logger.info(f"{dataset}: preprocessing {len(items)} images for both variants "
                    f"({items['mask'].notna().sum()} with lung masks) ...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            prepared = list(pool.map(_prepare_one, items.itertuples()))
        if dataset != "Kermany test":
            assert all(p["input"]["squash"] is p["input"]["crop"] for p in prepared), "square images should agree"
        for (name, v), cam in cams.items():
            def rgb(batch):
                return np.repeat(np.stack([p["input"][v] for p in batch])[..., None], 3, axis=-1)

            probs, heats = zip(*(batched_cam(cam, rgb(prepared[i:i + BATCH])) for i in range(0, len(prepared), BATCH)))
            probs, heats = np.concatenate(probs), np.concatenate(heats)
            ref_p, ref_heat, _ = cam(rgb(prepared[:1])[0])
            assert abs(ref_p[1] - probs[0]) < 1e-4 and np.allclose(ref_heat, heats[0], atol=1e-4), \
                "batched Grad-CAM differs from GradCAM"
            logger.info(f"{dataset} / {name} / {v}: Grad-CAM at {cam.layer_name} on {len(prepared)} images done")
            for r, p, heat, prep in zip(items.itertuples(), probs, heats, prepared):
                heat224 = np.clip(cv2.resize(heat, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR), 0, None)
                pred = BINARY_CLASSES[int(p >= 0.5)]
                rows.append({"dataset": dataset, "model": name, "variant": v, "path": r.path, "group": r.group,
                             "label": r.label, "pred": pred, "correct": pred == r.label, "p_pneumonia": float(p),
                             "has_mask": isinstance(r.mask, str), "cam_all_zero": heat.max() == 0,
                             "lung_retained_by_crop": prep["retained"], **_shares(heat224, prep["regions"][v])})
            if dataset == "Kermany test":
                keep[name, v] = heats
        if dataset == "Kermany test":
            keep["items"], keep["prepared"] = items, prepared
    return pd.DataFrame(rows), keep


def summarize(per_image: pd.DataFrame) -> dict:
    summary, paired, cross = [], [], []
    for (dataset, name, v), d in per_image.groupby(["dataset", "model", "variant"], sort=False):
        g = d["group"].to_numpy()
        for label in ("all", "NORMAL", "PNEUMONIA"):
            s = d if label == "all" else d[d["label"] == label]
            row = {"Dataset": dataset, "Model": name, "Preprocessing": v, "Class": label, "n": len(s),
                   "Grad-CAM all zero": int(s["cam_all_zero"].sum())}
            for k in REGIONS:
                ok = s[f"heat_{k}"].notna().to_numpy()
                row[f"Heat in {k}"] = s[f"heat_{k}"].mean()
                row[f"Area of {k}"] = s[f"area_{k}"].mean()
                row[f"n {k}"] = int(ok.sum())
                if label == "all" and k == "lung":
                    vals = s["heat_lung"].to_numpy()[ok]
                    row["Heat in lung 95% CI"] = fmt_ci(bootstrap_ci(g[ok], lambda i: vals[i].mean()), pct=True)
            summary.append(row)
        if dataset != "Kermany test":
            y = (d["label"] == "PNEUMONIA").to_numpy().astype(int)
            p = d["p_pneumonia"].to_numpy()
            m = compute_metrics(y, np.stack([1 - p, p], axis=1), BINARY_CLASSES)
            cross.append({"Model": name, "Preprocessing": v, "Tested on": dataset, "n": m["n"],
                          "Accuracy": m["accuracy"], "Accuracy 95% CI": fmt_ci(wilson(round(m["accuracy"] * m["n"]),
                                                                                       m["n"]), pct=True),
                          "Precision": m["precision"], "Recall": m["recall"], "Specificity": m["specificity"],
                          "F1": m["f1"], "ROC-AUC": m["roc_auc"]})

    for (dataset, name), d in per_image.groupby(["dataset", "model"], sort=False):
        wide = d.pivot(index="path", columns="variant")
        g = wide[("group", "squash")].to_numpy()
        for k in REGIONS:
            a, b = wide[(f"heat_{k}", "squash")].to_numpy(), wide[(f"heat_{k}", "crop")].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            if not ok.any():
                continue
            diff = b[ok] - a[ok]
            paired.append({"Dataset": dataset, "Model": name, "Region": k, "n": int(ok.sum()),
                           "Heat share, squash": a[ok].mean(), "Heat share, crop": b[ok].mean(),
                           "Crop - squash": diff.mean(),
                           "95% CI": fmt_ci(bootstrap_ci(g[ok], lambda i: diff[i].mean())),
                           "Wilcoxon p": wilcoxon(diff).pvalue if np.any(diff) else 1.0})

    summary, paired, cross = pd.DataFrame(summary), pd.DataFrame(paired), pd.DataFrame(cross)
    markdown(summary, OUT / "gradcam_heat", "Share of Grad-CAM heat inside each region (mean per image) and the "
                                            "region's share of the model input")
    markdown(paired, OUT / "gradcam_heat_paired", "Grad-CAM heat share, crop minus squash on the same images "
                                                  "(patient / cluster bootstrap CI, Wilcoxon signed-rank)")
    markdown(cross, OUT / "cross_dataset", "Unseen COVID-QU-Ex Normal / Non-COVID images (cross-dataset)")
    return {"heat": summary.to_dict(orient="records"), "heat_paired": paired.to_dict(orient="records"),
            "cross_dataset": cross.to_dict(orient="records")}


# ---------- Figures ----------

def plot_heat(per_image: pd.DataFrame, path):
    from src.plotting import SERIES, TEXT, TEXT_2, plt, save

    rows = [("Kermany test", "lung", "Kermany test\nlung mask"), ("Kermany test", "box", "Kermany test\ncentral box"),
            ("COVID-QU-Ex unseen", "lung", "COVID-QU-Ex (unseen)\nlung mask"),
            ("Kermany test", "border", "Kermany test\nouter 10% border")]
    colors = {"squash": SERIES[0], "crop": SERIES[1]}
    d = per_image[per_image["model"] == "ResNet50"]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for y, (dataset, region, _) in enumerate(rows):
        for j, v in enumerate(VARIANTS):
            s = d[(d["dataset"] == dataset) & (d["variant"] == v)]
            heat, area = s[f"heat_{region}"].mean(), s[f"area_{region}"].mean()
            yy = y + (j - 0.5) * 0.32
            ax.plot([area, heat], [yy, yy], color=colors[v], linewidth=1.2, alpha=0.5)
            ax.scatter([area], [yy], s=46, facecolors="white", edgecolors=colors[v], linewidths=1.6, zorder=3)
            ax.scatter([heat], [yy], s=60, color=colors[v], edgecolors="white", linewidths=1.5, zorder=4,
                       label=f"{v}: Grad-CAM heat share" if y == 0 else None)
            ax.text(heat + (0.015 if heat >= area else -0.015), yy, f"{heat:.0%}", va="center",
                    ha="left" if heat >= area else "right", fontsize=8.5, color=TEXT)
    ax.scatter([], [], s=46, facecolors="white", edgecolors=TEXT_2, linewidths=1.6,
               label="region's share of the input (even heat)")
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Share of the ResNet50 Grad-CAM heat (mean per image)")
    ax.set_title("Where the ResNet50 looks: squash vs centre-crop")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8)
    save(fig, path)


def plot_examples(keep: dict, path, per_class: int = 3):
    from src.plotting import TEXT, plt, save

    items = keep["items"].reset_index(drop=True)
    has_mask = items["mask"].notna()
    picks = pd.concat([items[has_mask & (items["label"] == c)].sample(per_class, random_state=SEED)
                       for c in BINARY_CLASSES])
    fig, axes = plt.subplots(2, len(picks), figsize=(2.3 * len(picks), 5.2))
    for col, (i, r) in enumerate(picks.iterrows()):
        prep = keep["prepared"][i]
        for row, v in enumerate(VARIANTS):
            heat = keep["ResNet50", v][i]
            over = overlay_heatmap(np.repeat(prep["input"][v][..., None], 3, axis=-1), heat)
            contours, _ = cv2.findContours(prep["regions"][v]["lung"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, contours, -1, (255, 255, 255), 1)
            heat224 = np.clip(cv2.resize(heat, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR), 0, None)
            share = _shares(heat224, prep["regions"][v])["heat_lung"]
            ax = axes[row, col]
            ax.imshow(over)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{v}: {share:.0%} heat in lungs", fontsize=8, color=TEXT)
            if row == 0:
                ax.text(0.5, 1.2, f"{r['label'].capitalize()}  ({r['width']}x{r['height']})", transform=ax.transAxes,
                        ha="center", fontsize=9, fontweight="semibold", color=TEXT)
    fig.suptitle("ResNet50 Grad-CAM on Kermany test images (white outline: COVID-QU-Ex lung mask)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, path, dpi=150)


def main():
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    masks = kermany_mask_paths()
    logger.info(f"{len(masks)} Kermany images have a COVID-QU-Ex lung mask")
    result = {"test": test_metrics()}
    result["size_probe"] = size_probes(masks)
    per_image, keep = gradcam_analysis(masks)

    # The on-the-fly predictions must reproduce the training run's test predictions
    for (name, v), d in per_image[per_image["dataset"] == "Kermany test"].groupby(["model", "variant"]):
        ref = pd.read_csv(variant_dir("results", v) / f"test_predictions_{MODELS[name]}.csv")
        gap = np.abs(d["p_pneumonia"].to_numpy() - ref["p_PNEUMONIA"].to_numpy()).max()
        assert gap < 1e-3, f"{name}/{v}: predictions differ from the training run by {gap}"

    per_image.to_csv(OUT / "gradcam_per_image.csv", index=False)
    result.update(summarize(per_image))
    result["settings"] = {"central_box_native_fraction": CENTRAL_BOX, "border_band_fraction": BORDER,
                          "bootstrap_resamples": N_BOOT, "kermany_images_with_mask": len(masks)}
    save_json(result, OUT / "comparison.json")
    plot_heat(per_image, OUT / "gradcam_heat.png")
    plot_examples(keep, OUT / "gradcam_examples.png")
    logger.info(f"Ablation comparison written to {OUT}")


if __name__ == "__main__":
    main()
