import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from src.utils import logger


def _unpack_labels_and_probs(model, test_ds):
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        label_np = labels.numpy()
        if label_np.ndim > 1 and label_np.shape[-1] > 1:
            y_true.extend(np.argmax(label_np, axis=1))
        else:
            y_true.extend(label_np.flatten())
        y_pred_probs.append(probs)

    y_true = np.array(y_true, dtype=int)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    return y_true, y_pred_probs


def validation_f1(model, val_ds):
    """F1 on the validation set (macro for multi-class), used to pick the best model."""
    y_true, y_pred_probs = _unpack_labels_and_probs(model, val_ds)
    if y_pred_probs.ndim == 2 and y_pred_probs.shape[-1] > 1:
        return float(f1_score(y_true, np.argmax(y_pred_probs, axis=1), average="macro", zero_division=0))
    return float(f1_score(y_true, (y_pred_probs.flatten() >= 0.5).astype(int), zero_division=0))


def evaluate_model_on_test(model, test_ds, model_name: str, results_dir: str = "./results", class_names=None):
    """
    Held-out test evaluation: accuracy, precision, recall, F1, confusion matrix, ROC-AUC.
    Supports binary sigmoid and multi-class softmax outputs.
    """
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Evaluating {model_name} on held-out test set...")

    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    y_true, y_pred_probs = _unpack_labels_and_probs(model, test_ds)
    multi = y_pred_probs.ndim == 2 and y_pred_probs.shape[-1] > 1

    if multi:
        y_pred = np.argmax(y_pred_probs, axis=1)
        avg = "macro"
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        try:
            auc_val = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
        except ValueError:
            auc_val = float("nan")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        positive_scores = y_pred_probs
    else:
        scores = y_pred_probs.flatten()
        y_pred = (scores >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_val = roc_auc_score(y_true, scores)
        cm = confusion_matrix(y_true, y_pred)
        positive_scores = scores

    metrics = {
        "Model": model_name,
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(auc_val) if auc_val == auc_val else None,
    }

    logger.info(f"--- Test Results for {model_name} ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {auc_val}")
    logger.info(f"Confusion Matrix:\n{cm}")

    if acc > 0.99:
        logger.warning(f"Test accuracy {acc:.4f} > 99%. Checking for potential data leakage...")

    plt.figure(figsize=(max(6, len(class_names) + 2), max(5, len(class_names) + 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {cm_path}")

    if multi:
        fpr, tpr = None, None
        roc_payload = ("multiclass", y_true, positive_scores, auc_val)
    else:
        fpr, tpr, _ = roc_curve(y_true, positive_scores)
        roc_payload = (fpr, tpr, auc_val)

    return metrics, roc_payload, y_true, y_pred, y_pred_probs


def plot_training_curves(history, model_name: str, results_dir: str = "./results"):
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("loss", []), label="Train Loss", color="crimson")
    if "val_loss" in history.history:
        plt.plot(history.history.get("val_loss", []), label="Val Loss", color="navy", linestyle="--")
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    acc_key = "accuracy" if "accuracy" in history.history else "binary_accuracy"
    plt.plot(history.history.get(acc_key, []), label="Train Acc", color="crimson")
    if f"val_{acc_key}" in history.history:
        plt.plot(history.history.get(f"val_{acc_key}", []), label="Val Acc", color="navy", linestyle="--")
    plt.title(f"{model_name} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(results_dir, f"training_curves_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved training curves to: {save_path}")


def plot_roc_comparison(roc_curves: dict, results_dir: str = "./results"):
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (name, payload) in enumerate(roc_curves.items()):
        color = colors[idx % len(colors)]
        # Binary payloads start with an fpr array, so check the type before comparing
        if isinstance(payload[0], str) and payload[0] == "multiclass":
            _, y_true, probs, auc_val = payload
            n_classes = probs.shape[1]
            mean_tpr = []
            mean_fpr = np.linspace(0, 1, 100)
            for k in range(n_classes):
                y_bin = (y_true == k).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                fpr_k, tpr_k, _ = roc_curve(y_bin, probs[:, k])
                mean_tpr.append(np.interp(mean_fpr, fpr_k, tpr_k))
            if mean_tpr:
                plt.plot(mean_fpr, np.mean(mean_tpr, axis=0), color=color, lw=2,
                         label=f"{name} (macro AUC = {auc_val:.4f})")
        else:
            fpr, tpr, auc_val = payload
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("ROC Curves Comparison (Test Set)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "roc_curve_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC comparison plot to: {save_path}")
