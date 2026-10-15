"""
External test on TBX11K, a dataset none of the models was trained on, after a near-duplicate check.

TBX11K Simplified (Kaggle vbookshelf/tbx11k-simplified, CC BY 4.0; Liu et al., CVPR 2020): 512x512 chest
X-rays labelled healthy, sick but not TB, or TB (active / latent). Public X-ray collections reuse each other's
images: Shenzhen, Montgomery and RSNA are already inside the training sources. So every TBX11K image is first
compared with every raw image of the three training sources (Kermany, COVID-QU-Ex, TB database, all ~47k images,
not only the subsets used) and dropped if it is a near-duplicate. Metrics use only the images left.

Evaluated: the 4-class ResNet50 the app serves, the 4-class baseline CNN, and the Review-I binary ResNet50
(false pneumonia alarms on healthy adults). The chest-X-ray gate's rejection rate is reported too.

  python -m src.external_test           # needs data/raw/tbx11k; -> splits/external/, results/external_tbx11k/
  python -m src.external_test --reuse   # rebuild the tables from the saved split file and predictions
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import MODELS_DIR, RAW_DIR, RESULTS_DIR, ROOT, SEED, SPLITS_DIR
from src.data.multiclass import NEAR_DUP_CORR, collect_covidqu, collect_tb, compute_thumbnails, max_similarity
from src.data.prepare import collect_kermany
from src.data.preprocessing import load_image, preprocess_image
from src.gate import ChestGate, knn_similarity
from src.utils import absolute, load_json, logger, save_json, set_seed

TBX_DIR = RAW_DIR / "tbx11k" / "tbx11k-simplified"
OUT = RESULTS_DIR / "external_tbx11k"
SPLIT_CSV = SPLITS_DIR / "external" / "tbx11k.csv"
NEAR_MISS_CORR = 0.95  # 0.95-0.98: kept, but listed and shown for a visual check
BATCH = 32
N_BOOT = 2000
SEEN_NAME = {"4-class ResNet50 (app)": "ResNet50", "4-class baseline CNN": "Baseline CNN"}  # per_class_metrics names
GROUPS = {"healthy": "Healthy", "sick_but_no_tb": "Sick, not TB", "active_tb": "Active TB", "latent_tb": "Latent TB"}


# ---------- Statistics ----------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(mask) -> str:
    mask = np.asarray(mask, dtype=bool)
    if not len(mask):
        return "-"
    lo, hi = wilson(int(mask.sum()), len(mask))
    return f"{mask.mean():.1%} ({lo:.1%}-{hi:.1%})"


def auc_ci(y, s, n_boot: int = N_BOOT, seed: int = SEED) -> str:
    """ROC-AUC with a 95% image-level bootstrap interval (TBX11K has no patient ids)."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"{roc_auc_score(y, s):.3f} ({lo:.3f}-{hi:.3f})"


def markdown(df: pd.DataFrame, name: str, title: str, note: str = ""):
    df.to_csv(OUT / f"{name}.csv", index=False)
    with open(OUT / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n" + (note + "\n\n" if note else "") + df.to_markdown(index=False) + "\n")


# ---------- Data and duplicate check ----------

def project_path(p) -> str:
    """Project-relative POSIX path. Unlike utils.rel it does not resolve each path, which is slow on OneDrive."""
    p = Path(p)
    return (p.relative_to(ROOT) if p.is_absolute() else p).as_posix()


def load_tbx11k() -> pd.DataFrame:
    meta = pd.read_csv(TBX_DIR / "data.csv")
    # One row per box in the original table: keep one row per image
    df = meta.drop_duplicates("fname").reset_index(drop=True)
    df["group"] = np.where(df["image_type"] == "tb", df["tb_type"], df["image_type"])
    df["path"] = [project_path(TBX_DIR / "images" / f) for f in df["fname"]]
    missing = [p for p in df["path"] if not absolute(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} TBX11K images missing, e.g. {missing[:3]}")
    return df[["fname", "path", "image_type", "tb_type", "group", "source"]].rename(columns={"source": "tbx11k_split"})


def training_images() -> pd.DataFrame:
    """Every raw image of the three training sources, used or not."""
    parts = [collect_kermany()[["path"]].assign(source="kermany"),
             collect_covidqu(RAW_DIR / "covidqu")[["path"]].assign(source="covidqu"),
             collect_tb(RAW_DIR / "tb_database")[["path"]].assign(source="tbdb")]
    train = pd.concat(parts, ignore_index=True)
    train["path"] = [project_path(p) for p in train["path"]]  # COVID-QU-Ex / TB paths come back absolute
    return train


def duplicate_check(ext: pd.DataFrame) -> pd.DataFrame:
    train = training_images()
    logger.info(f"Near-duplicate check: {len(ext)} TBX11K images vs {len(train)} training-source images "
                f"{train['source'].value_counts().to_dict()} ...")
    t_train = compute_thumbnails([absolute(p) for p in train["path"]])
    t_ext = compute_thumbnails([absolute(p) for p in ext["path"]])
    sim, idx = max_similarity(t_ext, t_train)
    out = ext.assign(max_train_corr=sim.round(4), nearest_train=train["path"].to_numpy()[idx],
                     nearest_source=train["source"].to_numpy()[idx])
    out["kept"] = out["max_train_corr"] < NEAR_DUP_CORR
    return out


def recheck_pairs(ext: pd.DataFrame, size: int = 128) -> dict:
    """
    The 32x32 check calls all chest X-rays fairly similar (median r 0.92). Recompute the correlation at 128x128
    for every dropped pair and every kept pair at r >= NEAR_MISS_CORR: copies of one image stay near 1, different
    patients drop. Written to duplicate_recheck.json.
    """
    import cv2

    from src.data.preprocessing import to_gray_uint8

    def corr(a, b):
        x, y = (cv2.resize(to_gray_uint8(load_image(absolute(q))), (size, size), interpolation=cv2.INTER_AREA)
                .astype(np.float32).ravel() for q in (a, b))
        x, y = x - x.mean(), y - y.mean()
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    out = {"size": size}
    for key, part in [("dropped", ext[~ext["kept"]]),
                      ("kept_near_miss", ext[ext["kept"] & (ext["max_train_corr"] >= NEAR_MISS_CORR)])]:
        r = np.array([corr(a, b) for a, b in zip(part["path"], part["nearest_train"])])
        out[key] = {"n": len(part), "median": round(float(np.median(r)), 4), "min": round(float(r.min()), 4),
                    "max": round(float(r.max()), 4), "at_least_0.98": int((r >= NEAR_DUP_CORR).sum())}
    save_json(out, OUT / "duplicate_recheck.json")
    return out


def plot_nearest(df: pd.DataFrame, path, n: int = 6):
    """The external images most similar to a training image, next to that training image."""
    from src.data.preprocessing import to_gray_uint8
    from src.plotting import TEXT, plt, save

    top = df.sort_values("max_train_corr", ascending=False).head(n)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.8))
    for c, (_, r) in enumerate(top.iterrows()):
        for row, (p, label) in enumerate([(r["path"], f"TBX11K {r['fname']}"),
                                          (r["nearest_train"], f"nearest: {r['nearest_source']}")]):
            ax = axes[row, c]
            ax.imshow(to_gray_uint8(load_image(absolute(p))), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(label, fontsize=7.5, color=TEXT)
        axes[0, c].text(0.5, 1.22, f"r = {r['max_train_corr']:.3f}", transform=axes[0, c].transAxes,
                        ha="center", fontsize=9, fontweight="semibold", color=TEXT)
    fig.suptitle(f"Most similar TBX11K / training pairs (dropped if r >= {NEAR_DUP_CORR})", fontsize=11)
    fig.subplots_adjust(top=0.82, bottom=0.02, left=0.01, right=0.99, hspace=0.25, wspace=0.08)
    save(fig, path, dpi=150)


# ---------- Predictions ----------

def predict(model_file, meta_file, df: pd.DataFrame, gate_ref=None) -> pd.DataFrame:
    """Class probabilities for every image (plus the chest-gate similarity when gate_ref is given)."""
    model = tf.keras.models.load_model(model_file)
    meta = load_json(meta_file)
    classes = meta["classes"]
    variant = meta.get("preprocessing", {}).get("variant", "squash")
    outputs = [model.output] + ([model.get_layer("gap").output] if gate_ref is not None else [])
    runner = tf.keras.Model(model.input, outputs)
    probs, sims = [], []
    paths = df["path"].tolist()
    for i in range(0, len(paths), BATCH):
        x = np.stack([preprocess_image(load_image(absolute(p)), variant) for p in paths[i:i + BATCH]])
        res = runner(x.astype("float32"), training=False)
        res = res if isinstance(res, (list, tuple)) else [res]
        p = np.asarray(res[0])
        probs.append(np.concatenate([1 - p, p], axis=1) if p.shape[1] == 1 else p)
        if gate_ref is not None:
            f = np.asarray(res[1])
            sims.append(knn_similarity(f / np.linalg.norm(f, axis=1, keepdims=True), gate_ref))
        if (i // BATCH) % 40 == 0:
            logger.info(f"  {Path(model_file).stem} ({meta['task']}): {i + len(x)}/{len(paths)}")
    probs = np.concatenate(probs)
    out = df[["fname", "group"]].copy()
    for j, c in enumerate(classes):
        out[f"p_{c}"] = probs[:, j]
    out["pred"] = np.array(classes)[probs.argmax(axis=1)]
    if gate_ref is not None:
        out["chest_similarity"] = np.concatenate(sims)
    return out


# ---------- Main ----------

def prediction_file(name: str):
    return OUT / f"predictions_{name.split(' (')[0].lower().replace(' ', '_').replace('-', '')}.csv"


def main(reuse: bool = False):
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if reuse and SPLIT_CSV.exists():
        ext = pd.read_csv(SPLIT_CSV)
    else:
        ext = duplicate_check(load_tbx11k())
        ext.to_csv(SPLIT_CSV, index=False)
    dropped, near = ext[~ext["kept"]], ext[ext["kept"] & (ext["max_train_corr"] >= NEAR_MISS_CORR)]
    dup = {"tbx11k_images": len(ext), "dropped_near_duplicates": len(dropped), "threshold": NEAR_DUP_CORR,
           "dropped_by_nearest_source": dropped["nearest_source"].value_counts().to_dict(),
           "dropped_by_group": dropped["group"].value_counts().to_dict(),
           "kept_between_0.95_and_threshold": len(near),
           "kept": int(ext["kept"].sum()), "kept_by_group": ext[ext["kept"]]["group"].value_counts().to_dict(),
           "max_corr_quantiles": ext["max_train_corr"].quantile([0.5, 0.9, 0.99, 1.0]).round(3).to_dict()}
    save_json(dup, OUT / "duplicate_check.json")
    plot_nearest(ext, OUT / "nearest_training_pairs.png")
    dup["recheck_128px"] = recheck_pairs(ext)
    logger.info(f"Duplicate check: {json.dumps(dup)}")
    test = ext[ext["kept"]].reset_index(drop=True)

    # Models: the app's 4-class ResNet50 (with the chest gate), the 4-class baseline, the binary ResNet50
    app_meta = load_json(MODELS_DIR / "multiclass" / "class_names.json")
    app_file = MODELS_DIR / "multiclass" / app_meta["model_file"]
    gate = ChestGate.load(tf.keras.models.load_model(app_file), app_file)
    if gate is None:
        raise RuntimeError("Chest gate missing or fitted for another model: run python -m src.gate")
    jobs = {
        "4-class ResNet50 (app)": (app_file, MODELS_DIR / "multiclass" / "class_names.json", gate.reference),
        "4-class baseline CNN": (MODELS_DIR / "multiclass" / "baseline_cnn.keras",
                                 MODELS_DIR / "multiclass" / "class_names.json", None),
        "Binary ResNet50 (Kermany)": (MODELS_DIR / "binary" / "resnet50.keras",
                                      MODELS_DIR / "binary" / "class_names.json", None),
    }
    runs = {}
    for name, (model_file, meta_file, ref) in jobs.items():
        if reuse and prediction_file(name).exists():
            runs[name] = pd.read_csv(prediction_file(name))
            assert runs[name]["fname"].tolist() == test["fname"].tolist(), f"{name}: saved predictions are stale"
        else:
            runs[name] = predict(model_file, meta_file, test, ref)
            runs[name].to_csv(prediction_file(name), index=False)

    # TB detection (4-class models), with the seen-source held-out test for reference
    seen = pd.read_csv(RESULTS_DIR / "multiclass" / "per_class_metrics.csv")
    tb_rows = []
    for name in ("4-class ResNet50 (app)", "4-class baseline CNN"):
        p = runs[name]
        is_tb = p["group"].isin(["active_tb", "latent_tb"])
        healthy = p["group"] == "healthy"
        sel = is_tb | healthy
        ref = seen[(seen["Model"] == SEEN_NAME[name]) & (seen["Class"] == "Tuberculosis")]
        tb_rows.append({
            "Model": name,
            "TB found (all TB)": rate(p.loc[is_tb, "pred"] == "TB"),
            "TB found (active)": rate(p.loc[p["group"] == "active_tb", "pred"] == "TB"),
            "TB found (latent)": rate(p.loc[p["group"] == "latent_tb", "pred"] == "TB"),
            "Healthy called TB": rate(p.loc[healthy, "pred"] == "TB"),
            "Sick-not-TB called TB": rate(p.loc[p["group"] == "sick_but_no_tb", "pred"] == "TB"),
            "P(TB) AUC, TB vs healthy": auc_ci(is_tb[sel].astype(int), p.loc[sel, "p_TB"]),
            "P(TB) AUC, TB vs all non-TB": auc_ci(is_tb.astype(int), p["p_TB"]),
            "Seen-source test: TB recall / AUC": (f"{ref['recall'].iloc[0]:.1%} / {ref['roc_auc'].iloc[0]:.3f}"
                                                  if len(ref) else "-"),
        })
    markdown(pd.DataFrame(tb_rows), "tb_detection", "TB detection on TBX11K (external; 95% CIs)",
             "TB found = predicted class is TB. AUC uses the predicted TB probability. The last column is the "
             "same model on its held-out test split from the training sources, for reference.")

    # Any finding vs healthy: 'not Normal' as the score, for all three models
    abn_rows = []
    for name, p in runs.items():
        sick = p["group"] != "healthy"
        abn_rows.append({
            "Model": name,
            "Healthy flagged (any disease)": rate(p.loc[~sick, "pred"] != "NORMAL"),
            "Sick-not-TB flagged": rate(p.loc[p["group"] == "sick_but_no_tb", "pred"] != "NORMAL"),
            "TB flagged (any disease)": rate(p.loc[p["group"].isin(["active_tb", "latent_tb"]), "pred"] != "NORMAL"),
            "AUC, 1 - P(Normal): sick or TB vs healthy": auc_ci(sick.astype(int), 1 - p["p_NORMAL"]),
        })
    markdown(pd.DataFrame(abn_rows), "any_finding", "Any finding vs healthy on TBX11K (external; 95% CIs)",
             "Flagged = predicted class is not Normal, whatever the disease name.")

    # What each model calls each kind of image
    dist = []
    for name, p in runs.items():
        for g, label in GROUPS.items():
            s = p[p["group"] == g]
            row = {"Model": name, "TBX11K images": label, "n": len(s)}
            row.update({f"-> {c}": f"{(s['pred'] == c).mean():.1%}" for c in sorted(p["pred"].unique(),
                        key=["NORMAL", "PNEUMONIA", "COVID", "TB"].index)})
            dist.append(row)
    markdown(pd.DataFrame(dist).fillna("0.0%"), "predicted_classes", "Predicted class by TBX11K image type")

    # Healthy adults: false alarms, and the chest gate
    app = runs["4-class ResNet50 (app)"]
    healthy_rows = [{"Check": "4-class ResNet50 (app): healthy called Normal",
                     "Rate (95% CI)": rate(app.loc[app["group"] == "healthy", "pred"] == "NORMAL")},
                    {"Check": "Binary ResNet50 (Kermany): healthy called Pneumonia",
                     "Rate (95% CI)": rate(runs["Binary ResNet50 (Kermany)"].query("group == 'healthy'")["pred"]
                                           == "PNEUMONIA")}]
    for g, label in GROUPS.items():
        healthy_rows.append({"Check": f"Chest gate rejects real X-rays: {label}",
                             "Rate (95% CI)": rate(app.loc[app["group"] == g, "chest_similarity"]
                                                   < gate.threshold)})
    markdown(pd.DataFrame(healthy_rows), "false_alarms_and_gate",
             "False alarms on healthy adults, and chest-gate rejections of real X-rays (TBX11K)")

    plot_roc(runs, OUT / "tb_roc.png")
    save_json({"duplicate_check": dup, "tb_detection": tb_rows, "any_finding": abn_rows, "predicted_classes": dist,
               "false_alarms_and_gate": healthy_rows, "gate_threshold": gate.threshold}, OUT / "external_test.json")
    logger.info(f"External test written to {OUT}")


def plot_roc(runs: dict, path):
    from src.plotting import GRID, SERIES, plt, save

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for i, name in enumerate(("4-class ResNet50 (app)", "4-class baseline CNN")):
        p = runs[name]
        is_tb = p["group"].isin(["active_tb", "latent_tb"])
        sel = is_tb | (p["group"] == "healthy")
        y, s = is_tb[sel].astype(int), p.loc[sel, "p_TB"]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=SERIES[1] if i == 0 else SERIES[0],
                label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1.2, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate (healthy called TB)")
    ax.set_ylabel("True positive rate (TB found)")
    ax.set_title("TB vs healthy on external TBX11K")
    ax.legend(loc="lower right")
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true",
                        help="reuse splits/external/tbx11k.csv and the saved predictions instead of recomputing")
    main(parser.parse_args().reuse)
