"""Metrics and figures. Reported numbers come from the test split only."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import BINARY_THRESHOLD
from src.plotting import CLASS_COLORS, GRID, MODEL_COLORS, SEQ_BLUE, SERIES, TEXT_2, plt, save


def predict_probs(model, ds):
    """Returns (y_true class indices, probabilities of shape (n, n_classes))."""
    y_true, probs = [], []
    for images, labels in ds:
        p = model.predict_on_batch(images)
        p = np.asarray(p)
        labels = labels.numpy()
        if p.shape[-1] == 1:
            p = np.concatenate([1.0 - p, p], axis=1)
            y_true.append(labels[:, 0].astype(int))
        else:
            y_true.append(labels.argmax(axis=1))
        probs.append(p)
    return np.concatenate(y_true), np.concatenate(probs)


def predict_labels(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] == 2:
        return (probs[:, 1] >= BINARY_THRESHOLD).astype(int)
    return probs.argmax(axis=1)


def compute_metrics(y_true, probs, class_names) -> dict:
    y_pred = predict_labels(probs)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    m = {"accuracy": accuracy_score(y_true, y_pred), "n": int(len(y_true)), "confusion_matrix": cm.tolist()}
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        m.update({
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, probs[:, 1]) if len(set(y_true)) > 1 else float("nan"),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })
    else:
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        m.update({
            "precision": float(p.mean()),
            "recall": float(r.mean()),
            "f1": float(f.mean()),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc": _ovr_auc(y_true, probs, labels),
            "per_class": {
                c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i]),
                    "roc_auc": _single_auc(y_true == i, probs[:, i])}
                for i, c in enumerate(class_names)
            },
        })
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def _single_auc(y_bin, scores):
    return float(roc_auc_score(y_bin, scores)) if 0 < y_bin.sum() < len(y_bin) else float("nan")


def _ovr_auc(y_true, probs, labels):
    present = [i for i in labels if (y_true == i).any()]
    if len(present) < 2:
        return float("nan")
    return float(np.mean([_single_auc(y_true == i, probs[:, i]) for i in present]))


def subtype_accuracy(df: pd.DataFrame, y_true, y_pred, class_names) -> pd.DataFrame:
    """Accuracy per subtype (bacterial / viral pneumonia, normal) where the subtype is known."""
    d = df.assign(correct=(y_true == y_pred))
    out = d.groupby("subtype")["correct"].agg(n="size", correct="sum")
    out["accuracy"] = out["correct"] / out["n"]
    return out.reset_index()


# ---------- Figures ----------

def plot_confusion_matrix(cm, class_names, title, path, display=None):
    cm = np.asarray(cm)
    display = display or {}
    names = [display.get(c, c) for c in class_names]
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    size = 3.2 + 0.9 * len(class_names)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    ax.imshow(pct, cmap=SEQ_BLUE, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            dark = pct[i, j] > 0.55
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1%}", ha="center", va="center", fontsize=10,
                    color="white" if dark else "#0b0b0b")
    ax.set_xticks(range(len(names)), names, rotation=20 if len(names) > 2 else 0, ha="right" if len(names) > 2 else "center")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, path)


def plot_roc_curves(curves: dict, class_names, path, display=None):
    """curves: {model name: (y_true, probs)}. Binary: one panel; multi-class: one OvR panel per model."""
    display = display or {}
    if len(class_names) == 2:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        for i, (name, (y_true, probs)) in enumerate(curves.items()):
            fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
            auc = roc_auc_score(y_true, probs[:, 1])
            ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, SERIES[i]), label=f"{name} (AUC {auc:.3f})")
        _roc_axes(ax, "ROC curves, test set")
        save(fig, path)
        return
    fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4.6), squeeze=False)
    for ax, (name, (y_true, probs)) in zip(axes[0], curves.items()):
        for i, cls in enumerate(class_names):
            y_bin = (y_true == i)
            if not 0 < y_bin.sum() < len(y_bin):
                continue
            fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
            ax.plot(fpr, tpr, color=CLASS_COLORS.get(cls, SERIES[i]),
                    label=f"{display.get(cls, cls)} (AUC {roc_auc_score(y_bin, probs[:, i]):.3f})")
        _roc_axes(ax, f"{name}: one-vs-rest ROC, test set")
    fig.tight_layout()
    save(fig, path)


def _roc_axes(ax, title):
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1.2, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")


def plot_training_curves(history: dict, name, path, fine_tune_start=None):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, label in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(epochs, history[key], color=SERIES[0], label="Train", marker="o", markersize=4)
        if f"val_{key}" in history:
            ax.plot(epochs, history[f"val_{key}"], color=SERIES[1], label="Validation", marker="o", markersize=4)
        if fine_tune_start:
            ax.axvline(fine_tune_start - 0.5, color=TEXT_2, linestyle=":", linewidth=1.2)
            ax.text(fine_tune_start - 0.4, ax.get_ylim()[1], " fine-tuning", va="top", fontsize=8, color=TEXT_2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_xticks(epochs)
        ax.set_title(f"{name}: {label.lower()}")
    axes[1].legend()
    fig.tight_layout()
    save(fig, path)


def comparison_table(rows: list, csv_path, md_path, title):
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    fmt = df.copy()
    for col in fmt.columns:
        if fmt[col].dtype.kind == "f":
            fmt[col] = fmt[col].map(lambda v: f"{v:.4f}")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(fmt.to_markdown(index=False) + "\n")
    return df
