"""
Training entry point.

  python -m src.train --task binary       # Review-I: baseline CNN + ResNet50 on Kermany
  python -m src.train --task multiclass   # Review-II Phase 3 (after src.data.multiclass)

Models are selected on validation F1 (macro for multi-class); all reported metrics
come from the held-out test split. QUICK_TRAIN=1 runs a few batches per stage.
"""
import argparse
import os
import shutil

import pandas as pd
import tensorflow as tf

from src.config import (
    BINARY_CLASSES,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    DISPLAY_NAMES,
    IMAGE_SIZE,
    MODELS_DIR,
    MULTICLASS_DISPLAY_NAMES,
    MULTICLASS_ORDER,
    PREPROCESS_VARIANT,
    RESULTS_DIR,
    SEED,
)
from src.data.loaders import class_weights, load_split, make_dataset
from src.data.splits import assert_no_overlap
from src.evaluate import (
    compute_metrics,
    comparison_table,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_curves,
    predict_labels,
    predict_probs,
    subtype_accuracy,
)
from src.gradcam import find_target_layer
from src.models import build_baseline_cnn, build_transfer_model, compile_model, unfreeze_top
from src.utils import detect_device, logger, save_json, set_seed

QUICK = os.environ.get("QUICK_TRAIN", "0") == "1"
# Epoch budgets from the project spec; early stopping may end a stage sooner
BUDGET = {"baseline": 8, "frozen": 5, "finetune": 3}
FINETUNE_LAYERS = 20
DISPLAY_MODEL = {"baseline": "Baseline CNN", "resnet50": "ResNet50", "mobilenet_v2": "MobileNetV2"}


def _callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7, verbose=1),
    ]


def _fit(model, train_ds, val_ds, epochs, weights):
    if QUICK:
        train_ds, val_ds, epochs = train_ds.take(3), val_ds.take(2), 1
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=weights,
                     callbacks=_callbacks(), verbose=2)
    return hist.history


def train_baseline(num_classes, train_ds, val_ds, weights):
    model = compile_model(build_baseline_cnn(num_classes), num_classes, 1e-3)
    logger.info(f"Baseline CNN: {model.count_params():,} parameters")
    return model, _fit(model, train_ds, val_ds, BUDGET["baseline"], weights), None


def train_transfer(backbone, num_classes, train_ds, val_ds, weights):
    model, base = build_transfer_model(backbone, num_classes)
    compile_model(model, num_classes, 1e-3)
    logger.info(f"{backbone}: phase 1, frozen backbone, {BUDGET['frozen']} epochs at lr 1e-3")
    h1 = _fit(model, train_ds, val_ds, BUDGET["frozen"], weights)
    best_frozen = min(h1["val_loss"])
    frozen_weights = model.get_weights()

    logger.info(f"{backbone}: phase 2, fine-tune last {FINETUNE_LAYERS} layers, {BUDGET['finetune']} epochs at lr 1e-5")
    unfreeze_top(base, FINETUNE_LAYERS)
    compile_model(model, num_classes, 1e-5)
    h2 = _fit(model, train_ds, val_ds, BUDGET["finetune"], weights)
    if min(h2["val_loss"]) > best_frozen:
        logger.info("Fine-tuning did not beat the frozen-backbone val_loss; keeping phase-1 weights.")
        model.set_weights(frozen_weights)
    history = {k: h1[k] + h2[k] for k in h1 if k in h2}
    return model, history, len(h1["loss"]) + 1


def run(task: str, backbone: str):
    set_seed(SEED)
    device = detect_device()
    logger.info(f"Task={task} device={device} backbone={backbone} preprocessing={PREPROCESS_VARIANT} quick={QUICK}")

    results_dir = RESULTS_DIR if task == "binary" else RESULTS_DIR / task
    model_dir = MODELS_DIR / task
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = (load_split(task, s) for s in ("train", "val", "test"))
    if task == "binary":
        class_names, display = BINARY_CLASSES, DISPLAY_NAMES
    else:
        present = set(train_df["label"])
        class_names = [c for c in MULTICLASS_ORDER if c in present]
        display = MULTICLASS_DISPLAY_NAMES
    num_classes = len(class_names)

    # Re-check the split files before spending an hour training on them
    all_df = pd.concat([d.assign(split=s) for d, s in ((train_df, "train"), (val_df, "val"), (test_df, "test"))])
    assert_no_overlap(all_df, "sha256")
    if "patient_id" in all_df:
        assert_no_overlap(all_df.dropna(subset=["patient_id"]), "patient_id")
    logger.info(f"Classes: {class_names}; train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}")

    train_ds = make_dataset(train_df, class_names, training=True)
    val_ds = make_dataset(val_df, class_names, training=False)
    test_ds = make_dataset(test_df, class_names, training=False)
    weights = class_weights(train_df, class_names)

    runs = {}
    for key in ("baseline", backbone):
        logger.info(f"===== Training {DISPLAY_MODEL[key]} =====")
        if key == "baseline":
            model, history, ft_start = train_baseline(num_classes, train_ds, val_ds, weights)
        else:
            model, history, ft_start = train_transfer(key, num_classes, train_ds, val_ds, weights)
        name = DISPLAY_MODEL[key]
        slug = name.lower().replace(" ", "_")
        model.save(model_dir / f"{slug}.keras")
        pd.DataFrame(history).rename_axis("epoch").to_csv(results_dir / f"history_{slug}.csv")
        plot_training_curves(history, name, results_dir / f"training_curves_{slug}.png", ft_start)

        y_val, p_val = predict_probs(model, val_ds)
        val_metrics = compute_metrics(y_val, p_val, class_names)
        y_test, p_test = predict_probs(model, test_ds)
        test_metrics = compute_metrics(y_test, p_test, class_names)
        y_pred = predict_labels(p_test)
        logger.info(f"{name}: val F1={val_metrics['f1']:.4f} | test acc={test_metrics['accuracy']:.4f} "
                    f"F1={test_metrics['f1']:.4f} AUC={test_metrics['roc_auc']:.4f}")

        preds = test_df.copy()
        preds["true"] = [class_names[i] for i in y_test]
        preds["pred"] = [class_names[i] for i in y_pred]
        for i, c in enumerate(class_names):
            preds[f"p_{c}"] = p_test[:, i]
        preds.to_csv(results_dir / f"test_predictions_{slug}.csv", index=False)
        plot_confusion_matrix(test_metrics["confusion_matrix"], class_names, f"{name}: test confusion matrix",
                              results_dir / f"confusion_matrix_{slug}.png", display)
        runs[name] = dict(model=model, val=val_metrics, test=test_metrics, y_test=y_test, p_test=p_test,
                          y_pred=y_pred, slug=slug, epochs=len(history["loss"]))

    plot_roc_curves({n: (r["y_test"], r["p_test"]) for n, r in runs.items()}, class_names,
                    results_dir / "roc_curves.png", display)

    # Comparison table (test metrics) with the validation F1 used for selection
    rows = []
    for n, r in runs.items():
        t = r["test"]
        row = {"Model": n, "Accuracy": t["accuracy"], "Precision": t["precision"], "Recall": t["recall"],
               "F1": t["f1"], "ROC-AUC": t["roc_auc"]}
        if num_classes == 2:
            row.update({"Specificity": t["specificity"], "TP": t["tp"], "TN": t["tn"], "FP": t["fp"], "FN": t["fn"]})
        row.update({"Val F1 (selection)": r["val"]["f1"], "Epochs": r["epochs"]})
        rows.append(row)
    label = "binary Normal vs Pneumonia" if task == "binary" else f"{num_classes}-class"
    comparison_table(rows, results_dir / "model_comparison.csv", results_dir / "model_comparison.md",
                     f"Model comparison, held-out test set ({label}, n={len(test_df)})")

    if num_classes > 2:
        per_class = [{"Model": n, "Class": display.get(c, c), **v}
                     for n, r in runs.items() for c, v in r["test"]["per_class"].items()]
        comparison_table(per_class, results_dir / "per_class_metrics.csv", results_dir / "per_class_metrics.md",
                         f"Per-class test metrics ({num_classes}-class, one-vs-rest ROC-AUC)")

    if "source" in test_df and test_df["source"].nunique() > 1:
        # Shortcut check: does accuracy for a class depend on which dataset the image came from?
        per_source = []
        for n, r in runs.items():
            d = test_df.assign(correct=(r["y_test"] == r["y_pred"]))
            g = d.groupby(["label", "source"])["correct"].agg(n="size", accuracy="mean").reset_index()
            per_source.append(g.assign(model=n))
        pd.concat(per_source).to_csv(results_dir / "per_source_accuracy.csv", index=False)

    if task == "binary":
        sub_rows = []
        for n, r in runs.items():
            s = subtype_accuracy(test_df, r["y_test"], r["y_pred"], class_names)
            sub_rows.append(s.assign(model=n))
        pd.concat(sub_rows)[["model", "subtype", "n", "correct", "accuracy"]].to_csv(
            results_dir / "subtype_accuracy.csv", index=False)

    best_name = max(runs, key=lambda n: runs[n]["val"]["f1"])
    best = runs[best_name]
    logger.info(f"Selected {best_name} (highest validation F1 = {best['val']['f1']:.4f})")
    leakage_note = None
    if best["test"]["accuracy"] > 0.99:
        leakage_note = ("Test accuracy > 99%: patient and file-hash overlap between splits were re-checked "
                        "(none found); see README for the source-shortcut analysis.")
        logger.warning(leakage_note)

    meta = {
        "task": task,
        "classes": class_names,
        "display_names": {c: display.get(c, c) for c in class_names},
        "model": best_name,
        "model_file": f"{best['slug']}.keras",
        "target_layer": find_target_layer(best["model"]),
        "threshold": 0.5 if num_classes == 2 else None,
        "preprocessing": {"variant": PREPROCESS_VARIANT,
                          "steps": ["grayscale"] + (["centre-crop to square"] if PREPROCESS_VARIANT == "crop" else [])
                          + [f"resize {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} (INTER_AREA)", "CLAHE", "3-channel RGB"],
                          "clahe_clip_limit": CLAHE_CLIP_LIMIT, "clahe_tile_grid": list(CLAHE_TILE_GRID)},
        "selection": "highest validation F1" + (" (macro)" if num_classes > 2 else ""),
        "val_f1": best["val"]["f1"],
        "test_metrics": {k: v for k, v in best["test"].items() if k != "per_class"},
        "quick_train": QUICK,
    }
    save_json(meta, model_dir / "class_names.json")
    # The app serves whichever task was trained most recently
    shutil.copy(model_dir / meta["model_file"], BEST_MODEL_PATH)
    save_json(meta, CLASS_NAMES_PATH)
    save_json({n: {"val": r["val"], "test": r["test"]} for n, r in runs.items()} | {
        "selected": best_name, "leakage_note": leakage_note}, results_dir / "metrics.json")
    logger.info(f"Saved {BEST_MODEL_PATH} and {CLASS_NAMES_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--backbone", choices=["resnet50", "mobilenet_v2"], default="resnet50")
    args = parser.parse_args()
    run(args.task, args.backbone)


if __name__ == "__main__":
    main()
