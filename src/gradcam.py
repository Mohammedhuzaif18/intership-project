"""
Grad-CAM (Selvaraju et al., 2017) and the Review-II heatmap report.

  python -m src.gradcam [--task binary|multiclass]   # 10 test images -> results/gradcam
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import MODELS_DIR, RESULTS_DIR
from src.utils import absolute, load_json, logger, save_json

L = tf.keras.layers


def find_target_layer(model: tf.keras.Model) -> str:
    """
    Last convolutional feature map before global pooling, found programmatically:
    take the last Conv2D / DepthwiseConv2D, then move forward through layers that keep
    its spatial size (BatchNorm, Add, activation). Gives conv5_block3_out for ResNet50,
    block4_out for the baseline CNN and out_relu for MobileNetV2.
    """
    layers = model.layers
    conv_idx = [i for i, l in enumerate(layers) if isinstance(l, (L.Conv2D, L.DepthwiseConv2D))]
    if not conv_idx:
        four_d = [l.name for l in layers if len(getattr(l.output, "shape", ())) == 4]
        if not four_d:
            raise ValueError("No convolutional feature map found in the model.")
        return four_d[-1]
    target = layers[conv_idx[-1]]
    spatial = tuple(target.output.shape[1:3])
    for layer in layers[conv_idx[-1] + 1:]:
        shape = tuple(layer.output.shape)
        if len(shape) == 4 and shape[1:3] == spatial:
            target = layer
        else:
            break
    return target.name


class GradCAM:
    """One forward pass gives both the prediction and the heatmap for a 224x224x3 uint8 image."""

    def __init__(self, model: tf.keras.Model, layer_name: str | None = None):
        self.model = model
        self.layer_name = layer_name or find_target_layer(model)
        self.grad_model = tf.keras.Model(model.inputs, [model.get_layer(self.layer_name).output, model.output])

    def __call__(self, image_rgb: np.ndarray, class_index: int | None = None):
        """Returns (probabilities over classes, heatmap in [0, 1] at feature-map resolution, class index used)."""
        x = tf.convert_to_tensor(image_rgb[None].astype("float32"))
        with tf.GradientTape() as tape:
            fmap, preds = self.grad_model(x, training=False)
            if preds.shape[-1] == 1:
                p = float(preds[0, 0])
                probs = np.array([1.0 - p, p])
                if class_index is None:
                    class_index = int(p >= 0.5)
                score = preds[:, 0] if class_index == 1 else 1.0 - preds[:, 0]
            else:
                probs = preds[0].numpy()
                if class_index is None:
                    class_index = int(np.argmax(probs))
                score = preds[:, class_index]
        grads = tape.gradient(score, fmap)
        if grads is None:
            raise RuntimeError(f"No gradient from the score to layer {self.layer_name}.")
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(fmap[0] * weights, axis=-1)).numpy()
        peak = cam.max()
        cam = cam / peak if peak > 0 else np.zeros_like(cam)
        return probs, cam, class_index


def colorize_heatmap(heatmap: np.ndarray, size) -> np.ndarray:
    """JET-colored heatmap, uint8 RGB, resized to size=(width, height)."""
    resized = cv2.resize(heatmap.astype("float32"), size, interpolation=cv2.INTER_LINEAR)
    jet = cv2.applyColorMap(np.uint8(255 * np.clip(resized, 0, 1)), cv2.COLORMAP_JET)
    return cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    jet = colorize_heatmap(heatmap, (w, h)).astype("float32")
    return np.clip(alpha * jet + (1 - alpha) * image_rgb.astype("float32"), 0, 255).astype(np.uint8)


# ---------- Report ----------

def _pick_samples(preds: pd.DataFrame, class_names, n_total=10, n_wrong=4, seed=42) -> pd.DataFrame:
    """Most confident mistakes (spread over true classes) plus confident correct cases per class."""
    preds = preds.assign(conf=preds[[f"p_{c}" for c in class_names]].max(axis=1))
    wrong = preds[preds["true"] != preds["pred"]].sort_values("conf", ascending=False)
    wrong_pick = (wrong.groupby("true", group_keys=False).head((n_wrong + 1) // 2)
                  .sort_values("conf", ascending=False).head(n_wrong))
    n_correct = n_total - len(wrong_pick)
    correct = preds[preds["true"] == preds["pred"]]
    per_class = max(1, n_correct // len(class_names))
    correct_pick = pd.concat([g.sample(min(per_class, len(g)), random_state=seed)
                              for _, g in correct.groupby("true")])
    picks = pd.concat([correct_pick, wrong_pick]).head(n_total)
    return picks.reset_index(drop=True)


def report(task: str = "binary"):
    from PIL import Image

    from src.plotting import plt, save

    model_dir = MODELS_DIR / task
    meta = load_json(model_dir / "class_names.json")
    class_names, display = meta["classes"], meta["display_names"]
    model = tf.keras.models.load_model(model_dir / meta["model_file"])
    cam = GradCAM(model)
    logger.info(f"Grad-CAM on {meta['model']} at layer {cam.layer_name}")

    results_dir = RESULTS_DIR if task == "binary" else RESULTS_DIR / task
    slug = meta["model_file"].removesuffix(".keras")
    preds = pd.read_csv(results_dir / f"test_predictions_{slug}.csv")
    picks = _pick_samples(preds, class_names)

    out_dir = (RESULTS_DIR / "gradcam") if task == "binary" else (results_dir / "gradcam")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for i, r in picks.iterrows():
        img = np.asarray(Image.open(absolute(r["preprocessed_path"])).convert("RGB"))
        probs, heat, idx = cam(img)
        pred = class_names[idx]
        assert pred == r["pred"], f"Grad-CAM prediction {pred} differs from evaluation prediction {r['pred']}"
        over = overlay_heatmap(img, heat)
        correct = r["true"] == pred
        title = (f"True {display[r['true']]} | Pred {display[pred]} ({probs[idx]:.0%})"
                 f"{'' if correct else '  WRONG'}")
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
        for ax, panel, t in zip(axes, [img, colorize_heatmap(heat, img.shape[1::-1]), over],
                                ["Model input (CLAHE)", "Grad-CAM heatmap", "Overlay"]):
            ax.imshow(panel)
            ax.set_title(t, fontsize=9)
            ax.axis("off")
        fig.suptitle(title, fontsize=10, color="#0b0b0b" if correct else "#c0302f")
        fname = f"{i + 1:02d}_{r['true']}_pred_{pred}_{'ok' if correct else 'wrong'}.png"
        save(fig, out_dir / fname, dpi=150)
        tiles.append((over, title, correct))
        rows.append({"file": r["path"], "subtype": r.get("subtype", ""), "true": r["true"], "pred": pred,
                     "confidence": round(float(probs[idx]), 4), "correct": bool(correct), "figure": fname})

    cols = 5
    fig, axes = plt.subplots(int(np.ceil(len(tiles) / cols)), cols, figsize=(3.1 * cols, 3.6 * np.ceil(len(tiles) / cols)))
    for ax, (over, title, correct) in zip(np.ravel(axes), tiles):
        ax.imshow(over)
        ax.set_title(title.replace(" | ", "\n"), fontsize=8, color="#0b0b0b" if correct else "#c0302f")
        ax.axis("off")
    for ax in np.ravel(axes)[len(tiles):]:
        ax.axis("off")
    fig.suptitle(f"Grad-CAM, {meta['model']} ({cam.layer_name}), held-out test images", fontsize=12)
    fig.tight_layout()
    save(fig, out_dir / "gradcam_grid.png", dpi=150)
    pd.DataFrame(rows).to_csv(out_dir / "gradcam_summary.csv", index=False)
    save_json({"model": meta["model"], "layer": cam.layer_name, "n": len(rows),
               "wrong": int(sum(not r["correct"] for r in rows))}, out_dir / "gradcam_meta.json")
    logger.info(f"Saved {len(rows)} Grad-CAM figures ({sum(not r['correct'] for r in rows)} wrong) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    report(parser.parse_args().task)
