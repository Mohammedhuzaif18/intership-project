"""
Lung-box ablation (follow-up to the centre-crop ablation): does cropping to the lungs remove the framing shortcut?

XRAY_PREPROCESS=lungbox finds the lungs with a U-Net (src.lung_segmenter), cuts out their bounding box plus a
margin, centres it on a black square and resizes that to 224x224. Nothing outside the box is kept.
This script compares the three binary pipelines (squash = default, crop = centre square, lungbox) with the
measures of src.ablation_crop:

  1. held-out Kermany test metrics, paired on the same images (McNemar, patient-level bootstrap);
  2. size and framing probes: what geometry is left in each model input (lung box size in native pixels,
     black bars, lung framing measured with the COVID-QU-Ex lung masks);
  3. how much Grad-CAM heat falls inside the lungs, the central box and the border band;
  4. accuracy on the unseen COVID-QU-Ex Normal / Non-COVID images (cross-dataset).

The squash and crop Grad-CAM rows come from results/ablation_crop/comparison/gradcam_per_image.csv (the same
models, unchanged by SHA-256). A sample of them is recomputed here and must match; only the lungbox models
are run on every image.

  python -m src.lung_segmenter
  XRAY_PREPROCESS=lungbox python -m src.data.prepare
  XRAY_PREPROCESS=lungbox python -m src.train --task binary
  python -m src.ablation_lungbox --prepare-only    # optional: the U-Net part, while the classifiers train
  python -m src.ablation_lungbox                   # -> results/ablation_lungbox/comparison

Reads all three pipelines' outputs and writes only to results/ablation_lungbox/comparison.
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import roc_auc_score

from src.ablation_crop import (BAND, BATCH, CENTRAL_BOX, BORDER, MODELS, N_BOOT, REGIONS, _auc, _framing, _probe,
                               _shares, batched_cam, bootstrap_ci, fmt_ci, kermany_mask_paths, markdown, native_box,
                               native_lung_mask, variant_dir, wilson)
from src.ablation_crop import _items as crop_items
from src.ablation_crop import _prepare_one as crop_prepare_one
from src.config import BINARY_CLASSES, IMAGE_SIZE, ROOT, SEED, variant_subdir
from src.data.preprocessing import (apply_clahe, center_square, crop_rect, fit_to_input, load_image, lung_rect,
                                    to_gray_uint8)
from src.evaluate import compute_metrics
from src.gradcam import GradCAM, overlay_heatmap
from src.lung_segmenter import lung_bbox
from src.utils import absolute, load_json, logger, save_json, set_seed

VARIANTS = ("squash", "crop", "lungbox")
NEW = "lungbox"
PAIRS = (("squash", NEW), ("crop", NEW))  # paired differences are lungbox minus the other variant
OUT = ROOT / "results" / variant_subdir(NEW) / "comparison"
CROP_ROWS = ROOT / "results" / variant_subdir("crop") / "comparison" / "gradcam_per_image.csv"
N_RECHECK = 12  # images per dataset whose squash / crop Grad-CAM rows are recomputed and must match
# Lungbox inputs and regions of the Grad-CAM images, so `--prepare-only` can build them while the classifiers train
CACHE_DIR = ROOT / "data" / "preprocessed" / variant_subdir(NEW) / "comparison_cache"


# ---------- Lung-box geometry ----------

def to_input(native_mask: np.ndarray, variant: str, rect=None) -> np.ndarray:
    """Bool mask at native size -> bool mask in the 224x224 model input, via its X-ray's geometric step.
    The lungbox rectangle comes from the X-ray, so it must be passed in (a mask has no lungs to find)."""
    assert variant != NEW or rect is not None, "lungbox needs the X-ray's rectangle"
    return fit_to_input(native_mask.astype(np.uint8) * 255, variant, rect) >= 128


def lungbox_rect(gray: np.ndarray):
    """(rectangle, predicted lung box or None) exactly as preprocessing.input_rect(gray, "lungbox") builds it."""
    box = lung_bbox(gray)
    if box:
        return lung_rect(box), box
    left, top, side = center_square(gray.shape[1], gray.shape[0])
    return (left, top, left + side, top + side), None


def padding(rect, width: int, height: int) -> dict:
    """Black in the lungbox input: letterbox bars (set by the box's aspect ratio) and margin past the image edge."""
    x0, y0, x1, y1 = rect
    rw, rh = x1 - x0, y1 - y0
    side = max(rw, rh)
    inside = max(0, min(x1, width) - max(x0, 0)) * max(0, min(y1, height) - max(y0, 0))
    return {"input_side_px": side, "log_input_side": np.log(side), "box_aspect": rw / rh,
            "black_share": 1 - inside / side ** 2, "letterbox_share": 1 - rw * rh / side ** 2,
            "outside_image_share": (rw * rh - inside) / side ** 2,
            "outside_left": max(0, -x0) / side, "outside_right": max(0, x1 - width) / side,
            "outside_top": max(0, -y0) / side, "outside_bottom": max(0, y1 - height) / side}


def box_iou(a, b) -> float:
    if a is None or b is None:
        return np.nan
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return float(ix * iy / union)


def mask_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def in_rect(native_mask: np.ndarray, rect) -> float:
    """Share of the mask inside the lungbox rectangle (what the crop keeps of the true lungs)."""
    total = native_mask.sum()
    return float(crop_rect(native_mask, rect).sum() / total) if total else np.nan


def _geometry_one(args):
    path, mask_path = args
    gray = to_gray_uint8(load_image(absolute(path)))
    h, w = gray.shape
    rect, box = lungbox_rect(gray)
    row = {"path": path, "width": w, "height": h, **dict(zip(("rect_x0", "rect_y0", "rect_x1", "rect_y1"), rect)),
           "fallback": box is None, **padding(rect, w, h)}
    if isinstance(mask_path, str):
        native = native_lung_mask(mask_path, w, h)
        row.update({"gt_lungs_in_rect": in_rect(native, rect), "box_iou_vs_mask": box_iou(box, mask_box(native)),
                    **{f"{v}:{k}": val for v in VARIANTS
                       for k, val in _framing(to_input(native, v, rect if v == NEW else None)).items()}})
    return row


def lungbox_geometry(masks: dict) -> pd.DataFrame:
    """Lungbox rectangle, black bars and lung framing for every Kermany train / val / test image (cached)."""
    path = OUT / "lungbox_geometry.csv"
    split_dir = variant_dir("splits", "squash") / "binary"
    df = pd.concat([pd.read_csv(split_dir / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")])
    if path.exists():
        geo = pd.read_csv(path)
        if set(geo["path"]) == set(df["path"]):
            return df[["path", "label", "subtype", "split", "patient_id"]].merge(geo, on="path")
    logger.info(f"Lung boxes for {len(df)} Kermany images ({df['path'].isin(masks.keys()).sum()} with a mask) ...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        geo = pd.DataFrame(list(pool.map(_geometry_one, [(p, masks.get(p)) for p in df["path"]])))
    geo.to_csv(path, index=False)
    return df[["path", "label", "subtype", "split", "patient_id"]].merge(geo, on="path")


# ---------- 1. Test metrics ----------

def test_metrics() -> dict:
    rows, paired = [], []
    for name, slug in MODELS.items():
        preds = {v: pd.read_csv(variant_dir("results", v) / f"test_predictions_{slug}.csv") for v in VARIANTS}
        for v in VARIANTS[1:]:
            assert preds[v]["path"].tolist() == preds["squash"]["path"].tolist(), f"test sets differ ({v})"
        y = (preds["squash"]["label"] == "PNEUMONIA").to_numpy().astype(int)
        probs = {v: d[["p_NORMAL", "p_PNEUMONIA"]].to_numpy() for v, d in preds.items()}
        right = {v: (d["true"] == d["pred"]).to_numpy() for v, d in preds.items()}
        for v in VARIANTS:
            m = compute_metrics(y, probs[v], BINARY_CLASSES)
            rows.append({"Model": name, "Preprocessing": v, "n": m["n"], "Accuracy": m["accuracy"],
                         "Accuracy 95% CI": fmt_ci(wilson(int(right[v].sum()), len(y)), pct=True),
                         "Precision": m["precision"], "Recall": m["recall"], "Specificity": m["specificity"],
                         "F1": m["f1"], "ROC-AUC": m["roc_auc"], "FP": m["fp"], "FN": m["fn"]})
        groups = preds["squash"]["patient_id"].to_numpy()
        for a, b in PAIRS:
            only_a, only_b = int((right[a] & ~right[b]).sum()), int((right[b] & ~right[a]).sum())
            paired.append({
                "Model": name, "Comparison": f"{b} - {a}", "Accuracy change": right[b].mean() - right[a].mean(),
                "Accuracy change 95% CI": fmt_ci(bootstrap_ci(groups, lambda i: right[b][i].mean()
                                                              - right[a][i].mean())),
                "Only the other pipeline right": only_a, f"Only {b} right": only_b,
                "McNemar exact p": binomtest(min(only_a, only_b), only_a + only_b).pvalue if only_a + only_b else 1.0,
                "ROC-AUC change": _auc(y, probs[b][:, 1]) - _auc(y, probs[a][:, 1]),
                "AUC change 95% CI": fmt_ci(bootstrap_ci(groups, lambda i: _auc(y[i], probs[b][i, 1])
                                                         - _auc(y[i], probs[a][i, 1]))),
                "Prediction agreement": float((preds[a]["pred"] == preds[b]["pred"]).mean()),
            })
    rows, paired = pd.DataFrame(rows), pd.DataFrame(paired)
    markdown(rows, OUT / "test_metrics", "Held-out Kermany test set (n=583): squash vs centre crop vs lung box")
    markdown(paired, OUT / "test_paired", "Paired comparison on the same test images (lungbox minus the other "
                                         "pipeline; CIs from 2,000 patient-level bootstrap resamples)")
    subtypes = pd.concat([pd.read_csv(variant_dir("results", v) / "subtype_accuracy.csv").assign(preprocessing=v)
                          for v in VARIANTS])
    markdown(subtypes, OUT / "subtype_accuracy", "Test accuracy per subtype")
    selected = {v: load_json(variant_dir("results", v) / "metrics.json")["selected"] for v in VARIANTS}
    val_f1 = {v: {n: r["val"]["f1"] for n, r in load_json(variant_dir("results", v) / "metrics.json").items()
                  if isinstance(r, dict) and "val" in r} for v in VARIANTS}
    return {"metrics": rows.to_dict(orient="records"), "paired": paired.to_dict(orient="records"),
            "subtypes": subtypes.to_dict(orient="records"), "selected_on_validation": selected,
            "validation_f1": val_f1}


# ---------- 2. Size and framing probes ----------

def size_probes(geo: pd.DataFrame) -> dict:
    train, test = geo[geo["split"] == "train"], geo[geo["split"] == "test"]
    ytr, yte = (train["label"] == "PNEUMONIA").astype(int), (test["label"] == "PNEUMONIA").astype(int)

    def native(df):
        w, h = df["width"].to_numpy(float), df["height"].to_numpy(float)
        return pd.DataFrame({"log_width": np.log(w), "log_height": np.log(h), "aspect": w / h,
                             "log_crop_side": np.log(np.minimum(w, h))}, index=df.index)

    feats_tr, feats_te = pd.concat([train, native(train)], axis=1), pd.concat([test, native(test)], axis=1)
    pads = ["letterbox_share", "outside_left", "outside_right", "outside_top", "outside_bottom"]
    feature_sets = {
        "Native width + height + aspect ratio (Review-I probe)": ["log_width", "log_height", "aspect"],
        "Crop side only (all the size left after centre-cropping)": ["log_crop_side"],
        "Lung-box size only (longer side in native pixels: the resolution left after lung-box cropping)":
            ["log_input_side"],
        "Black areas only (letterbox share + margin past each image edge)": pads,
        "Everything the lungbox input shows about the geometry: lung-box size + black areas": ["log_input_side"] + pads,
    }
    rows = [{"Probe features": k, "n train": len(train), "n test": len(test),
             **_probe(feats_tr[c], ytr, feats_te[c], yte)} for k, c in feature_sets.items()]

    # Lung framing inside each model input, for the images with a lung mask (ground truth, not the U-Net)
    keys = ["lung_area", "lung_x0", "lung_x1", "lung_y0", "lung_y1"]
    framed = geo[geo["split"].isin(["train", "test"])].dropna(subset=[f"{v}:{k}" for v in VARIANTS for k in keys])
    ftr, fte = framed[framed["split"] == "train"], framed[framed["split"] == "test"]
    for v in VARIANTS:
        cols = [f"{v}:{k}" for k in keys]
        rows.append({"Probe features": f"Lung area + lung bounding box in the {v} model input (no pixels)",
                     "n train": len(ftr), "n test": len(fte),
                     **_probe(ftr[cols], ftr["label"] == "PNEUMONIA", fte[cols], fte["label"] == "PNEUMONIA")})
    cols = [f"{NEW}:{k}" for k in keys] + ["log_input_side"] + pads
    rows.append({"Probe features": "Lungbox: lung framing + lung-box size + black areas (no pixels)",
                 "n train": len(ftr), "n test": len(fte),
                 **_probe(ftr[cols], ftr["label"] == "PNEUMONIA", fte[cols], fte["label"] == "PNEUMONIA")})
    probes = pd.DataFrame(rows)
    markdown(probes, OUT / "size_probe", "Size and framing probes (logistic regression; train split -> test split)")

    # Per feature: class medians, and how well the feature alone separates the classes on the test images
    features = ([f"{v}:{k}" for k in keys for v in VARIANTS] + ["log_input_side", "box_aspect", "black_share"] + pads
                + ["gt_lungs_in_rect", "box_iou_vs_mask"])
    medians = framed.groupby("label")[features].median().T
    single_auc = [max(a, 1 - a) for a in (roc_auc_score(fte.loc[fte[f].notna(), "label"] == "PNEUMONIA",
                                                        fte.loc[fte[f].notna(), f]) for f in features)]
    framing = pd.DataFrame({"Feature": features, "Median Normal": medians["NORMAL"].to_numpy(),
                            "Median Pneumonia": medians["PNEUMONIA"].to_numpy(),
                            "Test ROC-AUC, this feature alone": single_auc})
    markdown(framing, OUT / "lung_framing", "Lung framing in the model input (Kermany train + test images with a "
                                            "COVID-QU-Ex lung mask; x0/x1 = left/right, y0/y1 = top/bottom edge of "
                                            "the lungs, as a fraction of the input side)")

    # How the lung boxes behave per split and class
    quality = geo.groupby(["split", "label"]).agg(
        images=("path", "size"), fallback_to_centre_square=("fallback", "sum"),
        margin_past_image_edge=("outside_image_share", lambda s: int((s > 0.005).sum())),
        mean_black_share=("black_share", "mean"), mean_letterbox_share=("letterbox_share", "mean"),
        median_box_aspect=("box_aspect", "median"), median_input_side_px=("input_side_px", "median"),
        masked_images=("gt_lungs_in_rect", "count"), median_gt_lungs_in_rect=("gt_lungs_in_rect", "median"),
        min_gt_lungs_in_rect=("gt_lungs_in_rect", "min"),
        mean_box_iou_vs_mask=("box_iou_vs_mask", "mean")).reset_index()
    markdown(quality, OUT / "lungbox_quality", "Lung boxes on Kermany: U-Net box vs the COVID-QU-Ex lung mask, "
                                               "fallbacks and black areas")
    return {"probes": probes.to_dict(orient="records"), "framing": framing.to_dict(orient="records"),
            "lungbox_quality": quality.to_dict(orient="records"), "framing_n": len(framed)}


# ---------- 3./4. Grad-CAM heat and cross-dataset accuracy ----------

def _prepare_lungbox(r):
    """Lungbox model input (single channel) and regions for one image."""
    gray = to_gray_uint8(load_image(absolute(r.path)))
    rect, box = lungbox_rect(gray)
    x = apply_clahe(fit_to_input(gray, NEW, rect))
    if isinstance(r.stored_lungbox, str):
        same = np.array_equal(x, np.asarray(Image.open(absolute(r.stored_lungbox))))
        assert same, f"{r.path}: on-the-fly lungbox preprocessing differs from the stored training image"
    h, w = gray.shape
    lung = native_lung_mask(r.mask, w, h) if isinstance(r.mask, str) else None
    return {"input": x, "fallback": box is None, "black_share": padding(rect, w, h)["black_share"],
            "gt_lungs_in_rect": in_rect(lung, rect) if lung is not None else np.nan,
            "regions": {"lung": to_input(lung, NEW, rect) if lung is not None else None,
                        "box": to_input(native_box(w, h), NEW, rect) if r.box else None, "border": BAND}}


def prepared_lungbox(dataset: str, its: pd.DataFrame) -> list:
    """_prepare_lungbox for every image of a dataset, cached in CACHE_DIR (git-ignored)."""
    path = CACHE_DIR / f"{dataset.replace(' ', '_')}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as npz:
            z = {k: npz[k] for k in npz.files}  # npz[k] re-reads the whole array on every access
        if z["paths"].tolist() == its["path"].tolist():
            logger.info(f"{dataset}: lungbox inputs from {path}")
            return [{"input": z["inputs"][i], "fallback": bool(z["fallback"][i]),
                     "black_share": float(z["black_share"][i]), "gt_lungs_in_rect": float(z["in_rect"][i]),
                     "regions": {"lung": z["lungs"][i] if z["has_lung"][i] else None,
                                 "box": z["boxes"][i] if z["has_box"][i] else None, "border": BAND}}
                    for i in range(len(its))]
    logger.info(f"{dataset}: lungbox preprocessing of {len(its)} images ({its['mask'].notna().sum()} with masks)")
    with ThreadPoolExecutor(max_workers=8) as pool:
        prepared = list(pool.map(_prepare_lungbox, its.itertuples()))
    empty = np.zeros(IMAGE_SIZE[::-1], dtype=bool)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, paths=np.array(its["path"].tolist()), inputs=np.stack([p["input"] for p in prepared]),
             fallback=np.array([p["fallback"] for p in prepared]),
             black_share=np.array([p["black_share"] for p in prepared]),
             in_rect=np.array([p["gt_lungs_in_rect"] for p in prepared]),
             has_lung=np.array([p["regions"]["lung"] is not None for p in prepared]),
             lungs=np.stack([empty if p["regions"]["lung"] is None else p["regions"]["lung"] for p in prepared]),
             has_box=np.array([p["regions"]["box"] is not None for p in prepared]),
             boxes=np.stack([empty if p["regions"]["box"] is None else p["regions"]["box"] for p in prepared]))
    return prepared


def items(dataset: str, masks: dict) -> pd.DataFrame:
    df = crop_items(dataset, masks)
    if dataset == "Kermany test":
        lb = pd.read_csv(variant_dir("splits", NEW) / "binary" / "test.csv")
        return df.assign(stored_lungbox=df["path"].map(dict(zip(lb["path"], lb["preprocessed_path"]))))
    return df.assign(stored_lungbox=None)


def _rgb(singles) -> np.ndarray:
    return np.repeat(np.stack(singles)[..., None], 3, axis=-1)


def _cam_rows(cam, dataset, name, variant, its: pd.DataFrame, prepared, extra):
    probs, heats = zip(*(batched_cam(cam, _rgb([p["input"] for p in prepared[i:i + BATCH]]))
                         for i in range(0, len(prepared), BATCH)))
    probs, heats = np.concatenate(probs), np.concatenate(heats)
    rows = []
    for r, p, heat, prep in zip(its.itertuples(), probs, heats, prepared):
        heat224 = np.clip(cv2.resize(heat, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR), 0, None)
        pred = BINARY_CLASSES[int(p >= 0.5)]
        rows.append({"dataset": dataset, "model": name, "variant": variant, "path": r.path, "group": r.group,
                     "label": r.label, "pred": pred, "correct": pred == r.label, "p_pneumonia": float(p),
                     "has_mask": isinstance(r.mask, str), "cam_all_zero": heat.max() == 0, **extra(prep),
                     **_shares(heat224, prep["regions"])})
    return rows, heats


def recheck_crop_rows(crop_rows: pd.DataFrame, masks: dict, cams: dict):
    """Recompute a sample of the stored squash / crop rows; they must match (same models, same code)."""
    for dataset in ("Kermany test", "COVID-QU-Ex unseen"):
        its = crop_items(dataset, masks)
        its = its[its["mask"].notna()].sample(N_RECHECK, random_state=SEED)
        prepared = [crop_prepare_one(r) for r in its.itertuples()]
        for v in ("squash", "crop"):
            for name in MODELS:
                p_v = [{"input": p["input"][v], "regions": p["regions"][v]} for p in prepared]
                rows, _ = _cam_rows(cams[name, v], dataset, name, v, its, p_v, lambda _: {})
                new = pd.DataFrame(rows).set_index("path")
                old = crop_rows[(crop_rows["dataset"] == dataset) & (crop_rows["model"] == name)
                                & (crop_rows["variant"] == v)].set_index("path").loc[new.index]
                cols = ["p_pneumonia"] + [f"heat_{k}" for k in REGIONS] + [f"area_{k}" for k in REGIONS]
                gap = np.nanmax(np.abs(new[cols].to_numpy(float) - old[cols].to_numpy(float)))
                assert gap < 1e-3, f"{dataset}/{name}/{v}: stored crop-ablation rows differ by {gap}"
    logger.info(f"Recomputed {N_RECHECK} squash/crop rows per dataset, model and variant: they match the stored ones")


def gradcam_analysis(masks: dict):
    models = {(name, v): tf.keras.models.load_model(variant_dir("models", v) / "binary" / f"{slug}.keras")
              for name, slug in MODELS.items() for v in VARIANTS}
    cams = {k: GradCAM(m) for k, m in models.items()}
    crop_rows = pd.read_csv(CROP_ROWS)
    recheck_crop_rows(crop_rows, masks, cams)

    rows, keep = [], {}
    for dataset in ("Kermany test", "COVID-QU-Ex unseen"):
        its = items(dataset, masks)
        prepared = prepared_lungbox(dataset, its)
        for name in MODELS:
            cam = cams[name, NEW]
            new_rows, heats = _cam_rows(cam, dataset, name, NEW, its, prepared, lambda p: {
                "fallback": p["fallback"], "black_share": p["black_share"], "gt_lungs_in_rect": p["gt_lungs_in_rect"]})
            ref_p, ref_heat, _ = cam(_rgb([prepared[0]["input"]])[0])
            assert abs(ref_p[1] - new_rows[0]["p_pneumonia"]) < 1e-4 and np.allclose(ref_heat, heats[0], atol=1e-4), \
                "batched Grad-CAM differs from GradCAM"
            logger.info(f"{dataset} / {name} / {NEW}: Grad-CAM at {cam.layer_name} on {len(prepared)} images done")
            rows += new_rows
            if dataset == "Kermany test":
                keep[name, NEW] = heats
        if dataset == "Kermany test":
            keep["items"], keep["prepared"] = its, prepared
    per_image = pd.concat([crop_rows[crop_rows["variant"].isin(["squash", "crop"])], pd.DataFrame(rows)],
                          ignore_index=True)
    return per_image, keep, cams


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
        for a, b in PAIRS:
            for k in REGIONS:
                va, vb = wide[(f"heat_{k}", a)].to_numpy(float), wide[(f"heat_{k}", b)].to_numpy(float)
                ok = np.isfinite(va) & np.isfinite(vb)
                if not ok.any():
                    continue
                diff = vb[ok] - va[ok]
                paired.append({"Dataset": dataset, "Model": name, "Comparison": f"{b} - {a}", "Region": k,
                               "n": int(ok.sum()), f"Heat share, {a}": va[ok].mean(), f"Heat share, {b}": vb[ok].mean(),
                               "Change": diff.mean(), "95% CI": fmt_ci(bootstrap_ci(g[ok], lambda i: diff[i].mean())),
                               "Wilcoxon p": wilcoxon(diff).pvalue if np.any(diff) else 1.0})
        if dataset != "Kermany test":
            # Paired accuracy on the unseen images
            right = {v: wide[("correct", v)].astype(bool).to_numpy() for v in VARIANTS}
            for a, b in PAIRS:
                only_a, only_b = int((right[a] & ~right[b]).sum()), int((right[b] & ~right[a]).sum())
                paired.append({"Dataset": dataset, "Model": name, "Comparison": f"{b} - {a}", "Region": "accuracy",
                               "n": len(g), f"Heat share, {a}": right[a].mean(), f"Heat share, {b}": right[b].mean(),
                               "Change": right[b].mean() - right[a].mean(),
                               "95% CI": fmt_ci(bootstrap_ci(g, lambda i: right[b][i].mean() - right[a][i].mean())),
                               "Wilcoxon p": binomtest(min(only_a, only_b), only_a + only_b).pvalue
                               if only_a + only_b else 1.0})

    summary, paired, cross = pd.DataFrame(summary), pd.DataFrame(paired), pd.DataFrame(cross)
    markdown(summary, OUT / "gradcam_heat", "Share of Grad-CAM heat inside each region (mean per image) and the "
                                            "region's share of the model input")
    markdown(paired, OUT / "gradcam_heat_paired", "Grad-CAM heat share (and unseen-set accuracy), lungbox minus the "
                                                  "other pipeline on the same images (patient / cluster bootstrap CI; "
                                                  "Wilcoxon signed-rank for heat, exact McNemar for accuracy)")
    markdown(cross, OUT / "cross_dataset", "Unseen COVID-QU-Ex Normal / Non-COVID images (cross-dataset)")
    return {"heat": summary.to_dict(orient="records"), "heat_paired": paired.to_dict(orient="records"),
            "cross_dataset": cross.to_dict(orient="records")}


# ---------- Figures ----------

def plot_heat(per_image: pd.DataFrame, path):
    from src.plotting import SERIES, TEXT, TEXT_2, plt, save

    rows = [("Kermany test", "lung", "Kermany test\nlung mask"), ("Kermany test", "box", "Kermany test\ncentral box"),
            ("COVID-QU-Ex unseen", "lung", "COVID-QU-Ex (unseen)\nlung mask"),
            ("Kermany test", "border", "Kermany test\nouter 10% border")]
    colors = {"squash": SERIES[0], "crop": SERIES[1], "lungbox": SERIES[2]}
    d = per_image[per_image["model"] == "ResNet50"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for y, (dataset, region, _) in enumerate(rows):
        for j, v in enumerate(VARIANTS):
            s = d[(d["dataset"] == dataset) & (d["variant"] == v)]
            heat, area = s[f"heat_{region}"].mean(), s[f"area_{region}"].mean()
            yy = y + (j - 1) * 0.26
            ax.plot([area, heat], [yy, yy], color=colors[v], linewidth=1.2, alpha=0.5)
            ax.scatter([area], [yy], s=40, facecolors="white", edgecolors=colors[v], linewidths=1.6, zorder=3)
            ax.scatter([heat], [yy], s=54, color=colors[v], edgecolors="white", linewidths=1.5, zorder=4,
                       label=f"{v}: Grad-CAM heat share" if y == 0 else None)
            ax.text(heat + (0.015 if heat >= area else -0.015), yy, f"{heat:.0%}", va="center",
                    ha="left" if heat >= area else "right", fontsize=8, color=TEXT)
    ax.scatter([], [], s=40, facecolors="white", edgecolors=TEXT_2, linewidths=1.6,
               label="region's share of the input (even heat)")
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Share of the ResNet50 Grad-CAM heat (mean per image)")
    ax.set_title("Where the ResNet50 looks: squash vs centre crop vs lung box")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
    save(fig, path)


def plot_examples(keep: dict, cams: dict, path, per_class: int = 3):
    """The crop ablation's example images (same picks), one row per pipeline."""
    from src.plotting import TEXT, plt, save

    its = keep["items"].reset_index(drop=True)
    has_mask = its["mask"].notna()
    picks = pd.concat([its[has_mask & (its["label"] == c)].sample(per_class, random_state=SEED)
                       for c in BINARY_CLASSES])
    fig, axes = plt.subplots(len(VARIANTS), len(picks), figsize=(2.3 * len(picks), 9.0))
    for col, (i, r) in enumerate(picks.iterrows()):
        old = crop_prepare_one(next(its.loc[[i]].itertuples()))
        panels = {v: (old["input"][v], old["regions"][v]) for v in ("squash", "crop")}
        panels[NEW] = (keep["prepared"][i]["input"], keep["prepared"][i]["regions"])
        for row, v in enumerate(VARIANTS):
            x, regions = panels[v]
            heat = keep[("ResNet50", NEW)][i] if v == NEW else cams["ResNet50", v](_rgb([x])[0])[1]
            over = overlay_heatmap(_rgb([x])[0], heat)
            contours, _ = cv2.findContours(regions["lung"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, contours, -1, (255, 255, 255), 1)
            heat224 = np.clip(cv2.resize(heat, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR), 0, None)
            share = _shares(heat224, regions)["heat_lung"]
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
    fig.subplots_adjust(top=0.89, bottom=0.01, left=0.01, right=0.99, hspace=0.14, wspace=0.06)
    save(fig, path, dpi=150)


def main():
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    masks = kermany_mask_paths()
    logger.info(f"{len(masks)} Kermany images have a COVID-QU-Ex lung mask")
    if "--prepare-only" in sys.argv:  # U-Net work that does not need the lungbox classifiers
        lungbox_geometry(masks)
        for dataset in ("Kermany test", "COVID-QU-Ex unseen"):
            prepared_lungbox(dataset, items(dataset, masks))
        return
    result = {"test": test_metrics()}
    geo = lungbox_geometry(masks)
    result["size_probe"] = size_probes(geo)
    per_image, keep, cams = gradcam_analysis(masks)

    # The on-the-fly lungbox predictions must reproduce the training run's test predictions
    for name in MODELS:
        d = per_image[(per_image["dataset"] == "Kermany test") & (per_image["model"] == name)
                      & (per_image["variant"] == NEW)]
        ref = pd.read_csv(variant_dir("results", NEW) / f"test_predictions_{MODELS[name]}.csv")
        gap = np.abs(d["p_pneumonia"].to_numpy() - ref["p_PNEUMONIA"].to_numpy()).max()
        assert gap < 1e-3, f"{name}/{NEW}: predictions differ from the training run by {gap}"

    per_image.to_csv(OUT / "gradcam_per_image.csv", index=False)
    result.update(summarize(per_image))
    result["settings"] = {"central_box_native_fraction": CENTRAL_BOX, "border_band_fraction": BORDER,
                          "bootstrap_resamples": N_BOOT, "kermany_images_with_mask": len(masks),
                          "squash_crop_rows_from": str(CROP_ROWS.relative_to(ROOT).as_posix()),
                          "rows_rechecked_per_dataset_model_variant": N_RECHECK}
    save_json(result, OUT / "comparison.json")
    plot_heat(per_image, OUT / "gradcam_heat.png")
    plot_examples(keep, cams, OUT / "gradcam_examples.png")
    logger.info(f"Ablation comparison written to {OUT}")


if __name__ == "__main__":
    main()
