# Interpretable Deep Learning for Respiratory Disorder Detection from Chest X-Rays

> **DISCLAIMER**: This is an educational and research prototype. It is **not** an FDA/CE certified medical diagnostic device and must never be used to make clinical patient decisions.

End-to-end internship pipeline that meets every listed objective:

1. Collect and preprocess chest X-rays (resize, normalize, augment).
2. Train a **baseline CNN**.
3. Train **ResNet50 transfer learning** and compare it with the baseline.
4. Evaluate with accuracy, precision, recall, F1, confusion matrix, and ROC-AUC, with **class-weight** imbalance handling.
5. Generate **Grad-CAM** explanations.
6. Serve a **FastAPI + web UI** that returns predicted class, confidence, and Grad-CAM.
7. **Optional multi-class** extension: COVID-19 and tuberculosis (TB).

---

## What the system does

Upload a chest radiograph in the web app. The API classifies it (NORMAL / PNEUMONIA, and COVID / TB when those classes were trained) and overlays a Grad-CAM heatmap showing which lung regions most influenced the decision.

---

## Objective 1 — Collect and preprocess

| Step | Script | What it does |
|------|--------|----------------|
| Download + split | `python src/prepare_data.py` | Kermany chest X-ray set from Hugging Face (`AiresPucrs/chest-xray`). Stratified **80/10/10** train/val/test **split by patient** (`StratifiedGroupKFold` on the patient ID in each filename), so no patient's X-rays appear in two splits. Asserts zero patient and zero file overlap. Rebuilds `data/processed/` from scratch. |
| Optional 4-class | `python src/prepare_multiclass.py` | Adds COVID (ieee8023 GitHub CXR set, grouped by its `patientid`) and TB (NLM Montgomery / Open-i, one image per patient). If Kaggle keys exist, also tries COVID-QU-Ex and a TB Kaggle set. You can drop extra files in `data/raw/extra/COVID` and `data/raw/extra/TB` (no patient IDs there, so each image is treated as its own patient). |
| Resize | `python -m src.preprocess` | Deletes `data/preprocessed_224/`, then writes multi-threaded **224×224 RGB** JPEGs there, so no files from an earlier split survive. |
| Normalize | Inside each model | Baseline: `Rescaling(1/255)`. ResNet50: `resnet50.preprocess_input`. MobileNetV2 (optional): `mobilenet_v2.preprocess_input`. |
| Augment (train only) | `src/dataset.py` | Rotation ±15°, shift ±10%, zoom ±10%, horizontal flip. |

If COVID/TB downloads fail, training still runs on the classes that exist (typically NORMAL + PNEUMONIA). Re-run `prepare_multiclass.py` after adding images locally.

---

## Objective 2 — Baseline CNN

Custom 4-block CNN in `src/models.py` (`build_baseline_cnn`):

- Conv 32 → 64 → 128 → 256, BatchNorm, MaxPool
- GlobalAveragePooling → Dense(128) → Dropout(0.4) → sigmoid (2-class) or softmax (3+ classes)
- Grad-CAM target: last conv block `conv4_out`

---

## Objective 3 — ResNet50 transfer learning vs baseline

`src/train.py` trains **both** models on the same splits and writes `results/model_comparison.csv` / `.md`.

**ResNet50 protocol**

1. ImageNet backbone, frozen; train the new head (`lr=1e-3`, up to 8 epochs).
2. Unfreeze the **last 20** backbone layers, with BatchNorm kept frozen; fine-tune (`lr=1e-5`, up to 8 epochs). If fine-tuning doesn't beat phase 1 on validation loss, the phase-1 weights are kept.

Every stage (the baseline gets up to 12 epochs) uses early stopping on validation loss (patience 3, best weights restored) and reduces the learning rate when validation loss stalls.
3. The model with the highest **validation** F1 (baseline, ResNet50, or MobileNetV2 when enabled) is saved to `models/best_model.keras`. The test set is never used for this choice.

On native Windows, TensorFlow is typically **CPU-only**. Training still uses **ResNet50** (not a substitute backbone) and always runs full epochs over the training set. On an 8-core laptop CPU an epoch takes about 3.5 minutes, so a full run takes 1–2 hours. `QUICK_TRAIN=1` runs a few batches per stage to check the pipeline; it does not produce a usable model.

Optional extra comparison: `INCLUDE_MOBILENET=1 python src/train.py`.

A GPU Colab notebook remains at `notebooks/ResNet50_Training_Colab.ipynb`.

---

## Objective 4 — Metrics and class imbalance

Held-out **test set only** (`src/evaluate.py`):

- Accuracy, precision, recall, F1
- Confusion matrix PNG
- ROC-AUC (binary) or **macro OvR ROC-AUC** (multi-class)
- Combined ROC plot: `results/roc_curve_comparison.png`

Imbalance: `sklearn.utils.class_weight.compute_class_weight(class_weight="balanced")` is passed to `model.fit`.

Previous CPU run (binary MobileNetV2 vs baseline, for reference). **These numbers are optimistic:** that run split by image, and about 74% of the pneumonia patients in its test set (278 of 374) had other X-rays in the training set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Baseline CNN | 73.04% | 73.04% | 100% | 84.42% | 0.670 |
| Transfer MobileNetV2 | 93.00% | 95.74% | 94.63% | 95.18% | 0.979 |

Re-run `python src/train.py` to refresh these tables for **ResNet50** and any extra classes.

---

## Objective 5 — Grad-CAM

`src/gradcam.py` finds the last Conv2D / DepthwiseConv2D layer **programmatically** (works for the baseline CNN, ResNet50 `conv5_block3_out`, MobileNetV2).

Training writes 10 side-by-side samples under `results/gradcam_samples/`. The API returns a JPEG overlay as base64.

Warm colors = stronger contribution to the predicted class.

---

## Objective 6 — Web system

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

| Method | Endpoint | Returns |
|--------|----------|---------|
| GET | `/health` | Model status, class names, Grad-CAM layer |
| POST | `/predict` | `predicted_class`, `confidence`, `class_probabilities`, `gradcam_overlay_base64` |

The frontend (`frontend/`) supports drag-and-drop upload, sample cases (including COVID/TB demos), probability bars for every class, and heatmap opacity.

---

## Objective 7 — Multi-class COVID-19 and TB (optional)

Classes, when present on disk: **NORMAL, PNEUMONIA, COVID, TB**.

- 2 classes → binary sigmoid  
- 3+ classes → softmax + macro metrics  

API and UI pick up class names from `models/class_indices.json` after training.

---

## Project layout

```
intership/
├── data/                      # Created by prepare scripts
│   ├── raw/                   # Downloads + optional extra/COVID, extra/TB
│   ├── processed/             # Binary stratified split
│   ├── processed_multiclass/  # 2–4 class stratified split
│   └── preprocessed_224/      # 224×224 RGB
├── src/
│   ├── config.py
│   ├── prepare_data.py
│   ├── prepare_multiclass.py
│   ├── preprocess.py
│   ├── splits.py              # patient-grouped stratified splitting
│   ├── dataset.py             # loaders, augmentation, class weights
│   ├── models.py              # Baseline CNN + ResNet50 (+ optional MobileNetV2)
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── utils.py
├── backend/main.py
├── frontend/                  # index.html, style.css, app.js
├── models/                    # best_model.keras, class_indices.json
├── results/
├── notebooks/ResNet50_Training_Colab.ipynb
├── tests/test_integration.py
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
pip install -r requirements.txt

python src/prepare_data.py
python src/prepare_multiclass.py
python -m src.preprocess
python src/train.py

uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
python tests/test_integration.py
```

**Environment flags**

| Variable | Effect |
|----------|--------|
| `QUICK_TRAIN=1` | A few batches per stage, for checking the pipeline only |
| `INCLUDE_MOBILENET=1` | Also train MobileNetV2 for a third comparison row |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Extra COVID / TB Kaggle datasets |

---

## Environment

- Python 3.12, TensorFlow 2.x (CPU on native Windows ≥ 2.11)
- Seed **42** (Python, NumPy, TensorFlow)
- Not for clinical use
