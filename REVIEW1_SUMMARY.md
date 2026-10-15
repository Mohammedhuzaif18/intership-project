# Review-I Summary: Normal vs Pneumonia from Chest X-Rays

> Educational prototype, not a diagnostic tool. All metrics below come from the held-out **test split only**.

## 1. Problem statement

Pneumonia is a leading cause of death in children under five, and chest X-rays are the standard first test, but reading them needs trained radiologists who are scarce in many settings. Review-I builds and compares two image classifiers that label a pediatric chest X-ray as **Normal** or **Pneumonia**: a small CNN trained from scratch (baseline) and an ImageNet-pretrained **ResNet50** fine-tuned for the task. The emphasis is on an honest evaluation: patient-level splits, a held-out test set, and checks for leakage and dataset shortcuts.

## 2. Dataset facts

| Item | Value |
|---|---|
| Source | Kermany et al. 2018, *Chest X-Ray Images (Pneumonia)*; Hugging Face mirror `AiresPucrs/chest-xray` (`hf_hub_download`) |
| Population | Pediatric patients aged 1–5, Guangzhou Women and Children's Medical Center |
| Images found | 5,856 (provided folders: train 5,216, val 16, test 624) |
| Corrupt files | 0 (decode check with Pillow, minimum side 64 px) |
| Exact duplicates removed | 32 (SHA-256; all copies had the same label; listed in `splits/binary/removed.csv`) |
| Images kept | **5,824**: Normal 1,579 (27.1%), Pneumonia 4,245 (72.9%) |
| Pneumonia subtypes (from filenames) | Bacterial 2,760 (65%), viral 1,485 (35%) |
| Patients | 3,117; every filename has a parseable patient ID |
| Native image size (median) | Normal 1654x1323 px, Pneumonia 1160x768 px (stored as 8-bit grayscale; 283 are RGB) |

**Split.** The provided folders were merged (the val folder has only 16 images) and re-split 80/10/10 with `StratifiedGroupKFold` (seed 42): grouped by patient, stratified by subtype. No patient and no file hash appears in two splits; both are asserted in code and re-checked before every training run.

| Split | Normal | Pneumonia (bacterial / viral) | Total | Patients |
|---|---|---|---|---|
| Train | 1,263 | 3,395 (2,208 / 1,187) | 4,658 | 2,490 |
| Val | 158 | 425 (276 / 149) | 583 | 313 |
| Test | 158 | 425 (276 / 149) | 583 | 314 |

**Why patient-level matters.** The dataset has several X-rays per patient. In an earlier image-level split of this folder, 278 of 374 pneumonia test patients (74%) also had X-rays in the training set, which inflates test scores.

**Dataset bias found in EDA.** Pneumonia images are much smaller than normal ones and have a wider aspect ratio (1.51 vs 1.25). A logistic regression on **image width, height and aspect ratio only, with no pixels,** reaches **test ROC-AUC 0.912** (balanced accuracy 0.84; `results/eda/size_shortcut_probe.json`). Resizing every image to 224x224 removes the absolute size, but the aspect-ratio squash and resampling differences remain, so part of any model's score may come from acquisition artifacts rather than lung findings.

## 3. Preprocessing

One function (`src/data/preprocessing.py`) is shared by training, Grad-CAM and the API:

1. Convert to 8-bit grayscale.
2. Resize to **224x224** (`INTER_AREA`).
3. **CLAHE** contrast enhancement (clip limit 2.0, 8x8 tiles). Applied after resizing, so it behaves the same for any source resolution.
4. Copy to 3 channels (**RGB**).
5. **Normalization** inside each model: `Rescaling(1/255)` for the baseline, `resnet50.preprocess_input` for ResNet50.

**Augmentation, training only:** rotation ±15°, translation ±10%, zoom ±10%, horizontal flip, brightness ±10%, contrast ±10%, black fill. These are Keras layers, active only in training mode. **Class imbalance:** `class_weight="balanced"` gives Normal 1.844 and Pneumonia 0.686.

## 4. Models

| | Baseline CNN | ResNet50 transfer learning |
|---|---|---|
| Architecture | 4 blocks of Conv3x3 -> BatchNorm -> ReLU -> MaxPool (32, 64, 128, 256 filters), GAP, Dense 128, Dropout 0.4, sigmoid | ImageNet ResNet50 (no top), GAP, Dense 128, Dropout 0.4, sigmoid |
| Parameters | 422,881 | 23.85 M (9.18 M trainable while fine-tuning) |
| Training | Adam 1e-3, up to 8 epochs | Phase 1: backbone frozen, Adam 1e-3, up to 5 epochs. Phase 2: last 20 backbone layers unfrozen (BatchNorm kept frozen), Adam 1e-5, 3 epochs |
| Epochs run | 8 | 4 + 3 (phase 1 early-stopped after 4; best weights from epoch 1 restored) |
| Common | Class weights, batch 32, early stopping on val loss (patience 3, best weights restored), LR x0.3 on plateau, seed 42 | |

The spec's no-GPU fallback is MobileNetV2 with fewer epochs. ResNet50 was kept because it measured about 3.5 min per epoch on this CPU (Intel i5-13420H), so the full schedule ran in 32 minutes; the whole Review-I training took 64 minutes. A Colab GPU notebook is in `notebooks/`.

## 5. Results (held-out test set, n = 583)

| Model | Accuracy (95% CI) | Precision | Recall (sensitivity) | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline CNN | 94.2% (92.0–95.8) | 0.935 | **0.988** | 0.817 | 0.961 | 0.986 |
| **ResNet50** | **95.7%** (93.8–97.1) | **0.972** | 0.969 | **0.924** | **0.971** | **0.993** |

Confusion matrices (rows = true, columns = predicted):

| | Baseline: Normal | Baseline: Pneumonia | ResNet50: Normal | ResNet50: Pneumonia |
|---|---|---|---|---|
| True Normal (158) | 129 | 29 | 146 | 12 |
| True Pneumonia (425) | 5 | 420 | 13 | 412 |

**Accuracy by pneumonia subtype** (share of each subtype classified as pneumonia):

| Subtype | n | Baseline CNN | ResNet50 |
|---|---|---|---|
| Bacterial | 276 | 99.6% | 98.2% |
| Viral | 149 | 97.3% | 94.6% |
| Normal (specificity) | 158 | 81.6% | 92.4% |

**Selected model: ResNet50.** It had the higher validation F1 (0.962 vs 0.946); the test set was not used for the choice. It is saved as `models/best_model.keras`, with `models/class_names.json`.

## 6. Key findings

- **ResNet50 is better balanced.** It cuts false positives on normal X-rays from 29 to 12 (specificity 81.6% to 92.4%) at the cost of 8 more missed pneumonias, and it ranks cases better (AUC 0.993 vs 0.986).
- **The accuracy gap is small and not statistically significant on this test set.** An exact McNemar test gives p = 0.18: on 22 images only ResNet50 is right, on 13 only the baseline. The confidence intervals overlap.
- **Viral pneumonia is harder than bacterial** for both models (94.6% vs 98.2% for ResNet50); viral changes are subtler and closer to normal.
- **No sign of split leakage.** Test accuracy is below 99%, and no patient or file is shared across splits. The **image-size bias** (Section 2) is the main threat to validity, and Review-II's Grad-CAM shows the model sometimes looks at image borders.

## 7. Figures for the slides

| Slide | File |
|---|---|
| Class balance | `results/eda/class_counts.png` |
| Patient-level split | `results/eda/split_counts.png` |
| Example images | `results/eda/samples_per_class.png` |
| Bacterial vs viral | `results/eda/pneumonia_subtypes.png` |
| Image sizes (bias) | `results/eda/image_sizes.png` |
| Intensity histograms, raw vs CLAHE | `results/eda/intensity_histograms.png` |
| Preprocessing before/after | `results/preprocessing_before_after.png` |
| Augmentation examples | `results/augmentation_examples.png` |
| Training curves | `results/training_curves_baseline_cnn.png`, `results/training_curves_resnet50.png` |
| Confusion matrices | `results/confusion_matrix_baseline_cnn.png`, `results/confusion_matrix_resnet50.png` |
| ROC curves | `results/roc_curves.png` |
| Comparison table | `results/model_comparison.md` |

## 8. Plan for Review-II

1. **Grad-CAM** on `conv5_block3_out`, found programmatically so it also works for the baseline and MobileNetV2, for 10 test images including mistakes.
2. **FastAPI backend** (`/health`, `/predict` with class, confidence, all probabilities and a Grad-CAM overlay) using the exact training preprocessing, plus a **web frontend** with upload, probability bars, heatmap overlay and a clear disclaimer.
3. **Integration test** that starts the server and checks real test images against the offline predictions.
4. **Multi-class extension:** Normal, Pneumonia/Non-COVID infection (COVID-QU-Ex merged with Kermany), COVID-19 and TB. Includes balancing across sources, duplicate and near-duplicate checks across sources (COVID-QU-Ex reuses Kermany images), and a **cross-dataset test** on an unseen source.
5. **Address the size bias:** document it, and suggest crop-to-square preprocessing as a follow-up.
