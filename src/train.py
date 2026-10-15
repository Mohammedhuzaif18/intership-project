import glob
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import IMAGE_SIZE
from src.dataset import create_datasets, compute_class_weights, get_default_data_dir
from src.evaluate import evaluate_model_on_test, plot_training_curves, plot_roc_comparison, validation_f1
from src.gradcam import find_target_conv_layer, make_gradcam_heatmap, overlay_gradcam
from src.models import build_baseline_cnn, build_transfer_model, compile_model
from src.utils import detect_device, logger, set_seed


def _training_budget():
    """Maximum epochs per stage over the full training set; early stopping usually ends sooner."""
    if os.environ.get("QUICK_TRAIN", "0") == "1":
        # Pipeline check only: a few batches per stage, not a usable model
        return {"baseline_epochs": 1, "phase1_epochs": 1, "phase2_epochs": 1, "steps": 20, "val_steps": 5}
    return {"baseline_epochs": 12, "phase1_epochs": 8, "phase2_epochs": 8, "steps": None, "val_steps": None}


def _callbacks():
    return [
        # Keep the epoch with the lowest val_loss, not whatever the last epoch was
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7, verbose=1),
    ]


def _fit_kwargs(budget, class_weights):
    kwargs = {
        "epochs": None,
        "class_weight": class_weights,
        "callbacks": _callbacks(),
        "verbose": 2,
    }
    if budget["steps"]:
        kwargs["steps_per_epoch"] = budget["steps"]
        kwargs["validation_steps"] = budget["val_steps"]
    return kwargs


def _combine_histories(*histories):
    combined = {}
    for hist in histories:
        for k, v in hist.history.items():
            combined.setdefault(k, []).extend(v)

    class DummyHistory:
        pass

    dh = DummyHistory()
    dh.history = combined
    return dh


def _pred_index_and_label(raw_pred, class_names):
    raw_pred = np.array(raw_pred)
    if raw_pred.ndim == 2:
        raw_pred = raw_pred[0]
    if raw_pred.shape[-1] == 1 or raw_pred.ndim == 0:
        p = float(np.squeeze(raw_pred))
        idx = 1 if p >= 0.5 else 0
        conf = p if idx == 1 else 1.0 - p
        return idx, class_names[idx], conf
    idx = int(np.argmax(raw_pred))
    return idx, class_names[idx], float(raw_pred[idx])


def _train_transfer(name, train_ds, val_ds, val_eval_ds, test_ds, class_names, class_weights, budget,
                    metrics_list, roc_dict):
    num_classes = len(class_names)
    logger.info(f"Training transfer model: {name}")
    model, backbone = build_transfer_model(model_name=name, input_shape=(224, 224, 3), num_classes=num_classes)
    compile_model(model, num_classes, learning_rate=1e-3)

    fit_common = _fit_kwargs(budget, class_weights)
    fit_common["epochs"] = budget["phase1_epochs"]
    logger.info("Phase 1: frozen backbone, train classification head")
    hist1 = model.fit(train_ds, validation_data=val_ds, **fit_common)
    phase1_best_val_loss = min(hist1.history["val_loss"])
    phase1_weights = model.get_weights()

    logger.info("Phase 2: unfreeze last 20 backbone layers, fine-tune")
    backbone.trainable = True
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    # BatchNorm stays in inference mode: re-estimating its ImageNet statistics from
    # small batches is the usual way fine-tuning goes wrong
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, num_classes, learning_rate=1e-5)
    fit_common = _fit_kwargs(budget, class_weights)
    fit_common["epochs"] = budget["phase2_epochs"]
    hist2 = model.fit(train_ds, validation_data=val_ds, **fit_common)
    if min(hist2.history["val_loss"]) > phase1_best_val_loss:
        logger.info("Fine-tuning did not beat the frozen-backbone val_loss; keeping phase-1 weights.")
        model.set_weights(phase1_weights)

    display_name = f"Transfer {name}"
    plot_training_curves(_combine_histories(hist1, hist2), display_name)
    metrics, roc, *_ = evaluate_model_on_test(model, test_ds, display_name, class_names=class_names)
    metrics["Val F1-Score"] = validation_f1(model, val_eval_ds)
    metrics_list.append(metrics)
    roc_dict[display_name] = roc
    return model, metrics


def run_training_pipeline():
    set_seed(42)
    device = detect_device()
    budget = _training_budget()
    logger.info(f"Execution Device: {device} | budget={budget}")

    train_ds, val_ds, test_ds, class_names = create_datasets()
    num_classes = len(class_names)
    class_weights = compute_class_weights(class_names=class_names)

    # One full pass over val for model selection; val_ds itself may be repeated for fit()
    val_eval_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    if budget["steps"]:
        train_ds = train_ds.repeat()
        val_ds = val_ds.repeat()
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    metrics_list = []
    roc_dict = {}

    logger.info("Training Baseline CNN...")
    baseline_model = build_baseline_cnn(input_shape=(224, 224, 3), num_classes=num_classes)
    compile_model(baseline_model, num_classes, learning_rate=1e-3)
    fit_common = _fit_kwargs(budget, class_weights)
    fit_common["epochs"] = budget["baseline_epochs"]
    history_baseline = baseline_model.fit(train_ds, validation_data=val_ds, **fit_common)
    plot_training_curves(history_baseline, "Baseline CNN")
    baseline_metrics, baseline_roc, *_ = evaluate_model_on_test(
        baseline_model, test_ds, "Baseline CNN", class_names=class_names
    )
    baseline_metrics["Val F1-Score"] = validation_f1(baseline_model, val_eval_ds)
    metrics_list.append(baseline_metrics)
    roc_dict["Baseline CNN"] = baseline_roc
    candidates = [("Baseline CNN", baseline_model, baseline_metrics)]

    resnet_model, resnet_metrics = _train_transfer(
        "resnet50", train_ds, val_ds, val_eval_ds, test_ds, class_names, class_weights, budget,
        metrics_list, roc_dict
    )
    candidates.append(("Transfer resnet50", resnet_model, resnet_metrics))

    include_mobilenet = os.environ.get("INCLUDE_MOBILENET", "0") == "1"
    if include_mobilenet:
        mobilenet_model, mobilenet_metrics = _train_transfer(
            "mobilenet_v2", train_ds, val_ds, val_eval_ds, test_ds, class_names, class_weights, budget,
            metrics_list, roc_dict
        )
        candidates.append(("Transfer mobilenet_v2", mobilenet_model, mobilenet_metrics))

    # Pick on validation F1 so the test set stays an unbiased estimate of the chosen model
    best_name, best_model, _ = max(candidates, key=lambda c: c[2]["Val F1-Score"])
    for name, _, m in candidates:
        logger.info(f"{name}: val F1 = {m['Val F1-Score']:.4f}")

    comparison_df = pd.DataFrame(metrics_list)
    os.makedirs("./results", exist_ok=True)
    comparison_df.to_csv("./results/model_comparison.csv", index=False)
    print("\n--- MODEL COMPARISON TABLE ---")
    print(comparison_df.to_string(index=False))
    try:
        md_table = comparison_df.to_markdown(index=False)
    except Exception:
        md_table = comparison_df.to_string(index=False)
    with open("./results/model_comparison.md", "w") as f:
        f.write("# Model Performance Comparison (Held-Out Test Set)\n\n")
        f.write(md_table + "\n")
        f.write(f"\nClasses: {', '.join(class_names)}\n")
        f.write(f"\nSaved model: **{best_name}** (highest validation F1; test metrics were not used to choose it).\n")
        f.write("\nResNet50 is the required transfer-learning backbone. ")
        f.write("MobileNetV2 is optional (`INCLUDE_MOBILENET=1`).\n")

    plot_roc_comparison(roc_dict)

    os.makedirs("./models", exist_ok=True)
    best_model.save("./models/best_model.keras")
    logger.info(f"Saved best model ({best_name}) to ./models/best_model.keras")

    logger.info("Generating Grad-CAM visualizations for test images...")
    test_dir = os.path.join(get_default_data_dir(), "test")
    target_conv_name = find_target_conv_layer(best_model)
    gradcam_samples_dir = "./results/gradcam_samples"
    os.makedirs(gradcam_samples_dir, exist_ok=True)

    selected_files = []
    per_class = max(1, 10 // max(len(class_names), 1))
    for idx, cls in enumerate(class_names):
        files = glob.glob(os.path.join(test_dir, cls, "*.*"))
        selected_files.extend([(p, idx) for p in files[:per_class]])
    selected_files = selected_files[:10]

    gradcam_report = []
    for sample_i, (img_path, true_label_idx) in enumerate(selected_files):
        pil_img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
        img_arr = np.array(pil_img)
        img_input = np.expand_dims(img_arr, axis=0)
        raw = best_model.predict(img_input, verbose=0)
        pred_idx, pred_label, confidence = _pred_index_and_label(raw, class_names)
        true_label = class_names[true_label_idx]
        try:
            heatmap = make_gradcam_heatmap(img_input, best_model, target_conv_name, pred_index=pred_idx)
            overlay, _ = overlay_gradcam(img_arr, heatmap, alpha=0.45)
            side_by_side = np.hstack([img_arr, overlay])
            out_filename = f"sample_{sample_i + 1}_{true_label}_pred_{pred_label}_{confidence:.2f}.jpg"
            Image.fromarray(side_by_side).save(os.path.join(gradcam_samples_dir, out_filename))
            gradcam_report.append({
                "Sample": sample_i + 1,
                "File": os.path.basename(img_path),
                "True Class": true_label,
                "Predicted Class": pred_label,
                "Confidence": f"{confidence * 100:.2f}%",
                "Correct": true_label == pred_label,
                "Saved Image": out_filename,
            })
            logger.info(f"Saved Grad-CAM: {out_filename}")
        except Exception as e:
            logger.error(f"Grad-CAM generation failed for {img_path}: {e}")

    pd.DataFrame(gradcam_report).to_csv(os.path.join(gradcam_samples_dir, "gradcam_summary.csv"), index=False)
    logger.info("Grad-CAM generation completed.")
    return comparison_df


if __name__ == "__main__":
    run_training_pipeline()
