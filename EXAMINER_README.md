# Project Guide for the Examiner

**Interpretable Deep Learning for Respiratory Disorder Detection from Chest X-Rays**

> This is a student project for learning. It is **not** a medical tool and must not be used to diagnose anyone.

This page explains the project in simple words. The full technical details are in [README.md](README.md), [REVIEW1_SUMMARY.md](REVIEW1_SUMMARY.md) and [REVIEW2_SUMMARY.md](REVIEW2_SUMMARY.md).

---

## 1. What the project does

You give the system a chest X-ray. It tells you:

1. **What it sees**: Normal, Pneumonia, COVID-19 or Tuberculosis (TB).
2. **How sure it is**: a confidence value for each class.
3. **Where it looked**: a **Grad-CAM heatmap**, a coloured overlay that shows which parts of the X-ray drove the decision.

It runs as a small website: a **FastAPI** backend plus an HTML/JavaScript frontend.

## 2. The work, step by step

| Stage | What was done |
|---|---|
| **Review-I** | Normal vs Pneumonia: data cleaning, a CNN built from scratch, ResNet50 transfer learning, full evaluation |
| **Review-II, Phase 2** | Grad-CAM explanations, the FastAPI backend, the web app, an integration test |
| **Review-II, Phase 3** | 4 classes (Normal, Pneumonia, COVID-19, TB) using two more datasets, and a test on a dataset the model never saw |
| **Extra checks** | Experiments that test whether the model learned real disease signs or shortcuts, and a test on an outside dataset (TBX11K) |

## 3. Data and how it was cleaned

**Main dataset:** Kermany chest X-rays, 5,856 images of children (Normal or Pneumonia).

What was done before training:

- **Removed bad files:** 0 corrupt images and 32 exact duplicates (found by comparing file fingerprints, SHA-256). 5,824 images remain.
- **Split by patient, not by image:** one child can have several X-rays. If one X-ray is in training and another in testing, the test is unfair. So all of a patient's images go into the same split:
  - 80% train, 10% validation, 10% test (4,658 / 583 / 583 images, 3,117 patients)
  - No patient appears in two splits (checked automatically).
- **Image preparation:** grayscale, resize to 224x224, then **CLAHE** contrast enhancement (makes lung detail clearer), then copy to 3 channels.
- **Class imbalance:** there are about 3x more pneumonia images than normal ones, so class weights were used (mistakes on normal images count more).
- **Augmentation, training only:** small random rotation, shift, zoom, flip and brightness changes.
- **Fixed seed (42)** everywhere, so the results can be reproduced.

For the 4-class model, two more datasets were added: **COVID-QU-Ex** (COVID-19, other infections, normal) and the **TB Chest X-ray Database**.

**Important discovery:** many images in the new datasets were copies of images from the other datasets:
- 5,172 COVID-QU-Ex images were Kermany images.
- 3,472 of the 3,500 TB-database normal images were COVID-QU-Ex images.

A simple file-fingerprint check misses these copies because they were resized. They were found by comparing small thumbnails, removed, and checked by eye. Without this step, the test results would have been falsely high.

## 4. Models

| Model | Description |
|---|---|
| **Baseline CNN** | A small network built from scratch (4 convolution blocks, about 0.42 million parameters) |
| **ResNet50** | A large network pre-trained on ImageNet, then fine-tuned on X-rays (transfer learning) |

The best model was picked on the **validation** set. All the numbers below come from the **test** set, which was never used for training or for choosing models.

## 5. Results

### Normal vs Pneumonia (test set, 583 images)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Baseline CNN | 94.2% | 0.961 | 0.986 |
| **ResNet50** | **95.7%** | **0.971** | **0.993** |

- ResNet50 makes far fewer false alarms on healthy X-rays (12 vs 29).
- The accuracy difference between the two models is not statistically significant (McNemar test, p = 0.18).
- ResNet50 accuracy by group:
  - bacterial pneumonia 98.2%
  - viral pneumonia 94.6%
  - normal 92.4%

### 4 classes: Normal, Pneumonia, COVID-19, TB (test set, 793 images)

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Baseline CNN | 72.6% | 0.734 | 0.934 |
| **ResNet50** | **82.3%** | **0.844** | **0.962** |

The app uses this 4-class ResNet50.

Confusion matrices, ROC curves and training curves are in [results/](results/) and [results/multiclass/](results/multiclass/).

## 6. Grad-CAM (explainability)

Grad-CAM makes a heatmap of the regions that pushed the model towards its answer.

- Examples: [results/gradcam/](results/gradcam/) (10 test images, 4 of them wrong predictions on purpose) and the overview [results/gradcam/gradcam_grid.png](results/gradcam/gradcam_grid.png).
- **Finding:** for many correct cases the heat is on the lungs. For some cases, including wrong ones, the heat is on the image border or on text markers. So the model does not always look at the right place. This is why Grad-CAM matters.

## 7. The web app

- **Page 1:** enter the patient's name and age, then upload an X-ray or pick a sample image.
- **Page 2:** a report showing:
  - the detected condition and its confidence
  - probability bars for all classes
  - the Grad-CAM heatmap, with views to compare the original and the overlay
  - a short note about the condition, and a disclaimer
  - a Print / Save as PDF button
- **Safety features:**
  - **Non-chest image check:** if you upload a photo that is not a chest X-ray (for example a face), the app says "Not a chest X-ray" instead of naming a disease.
  - **TB warning:** a TB prediction shows a warning that TB results are not reliable on new X-rays (see section 8).
- **Integration test** ([tests/test_integration.py](tests/test_integration.py)): starts the server and checks that the app gives the same answers as the offline evaluation. It passes.

Screenshots: [results/app/](results/app/).

## 8. Honest limitations (what the extra checks found)

High accuracy is not the whole story. These checks were done to test whether the model really learned disease signs:

1. **Image-size shortcut.** In Kermany, pneumonia X-rays are smaller and wider than normal ones. Image size alone predicts the class with AUC 0.912, without looking at the image content at all. So the model can partly "cheat" by using size.
2. **Crop experiment:** cutting a centre square instead of squashing the image. ResNet50 accuracy dropped from 95.7% to 93.7%, and the shortcut moved instead of going away. ([results/ablation_crop/REPORT.md](results/ablation_crop/REPORT.md))
3. **Lung-box experiment:** a lung-segmentation network (U-Net) finds the lungs and only the lungs are kept. Results:
   - Accuracy dropped to 90.9%.
   - The model's heatmap moved more onto the lungs.
   - Part of the original 95.7% came from things outside the lungs.

   ([results/ablation_lungbox/REPORT.md](results/ablation_lungbox/REPORT.md))
4. **New-dataset tests:**
   - The Kermany model falls from 95.7% to 73.9% on a different dataset (COVID-QU-Ex).
   - On a fully external TB dataset (TBX11K), the 4-class model found only 5.8% of TB cases, against 94.3% on its own test set. Its TB class mostly learned "which dataset the image came from", not TB itself.
   - Healthy adult X-rays are still called Normal 92.9% of the time.

   ([results/external_tbx11k/REPORT.md](results/external_tbx11k/REPORT.md))

**Main lesson:** a model can score very high on its own test set and still fail on X-rays from another hospital. Grad-CAM and these extra checks are what reveal it.

## 9. How to run the demo

From the project folder (Windows, Git Bash):

```bash
.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in a browser.

To run the integration test:

```bash
.venv/Scripts/python tests/test_integration.py
```

The commands to retrain everything from scratch are in [README.md](README.md) ("How to run").

## 10. Where things are

| Folder | Contents |
|---|---|
| `src/` | All the code: data, models, training, evaluation, Grad-CAM |
| `backend/` | FastAPI server |
| `frontend/` | Web pages (HTML, CSS, JavaScript) |
| `results/` | All figures, tables and reports |
| `splits/` | Which image is in train / validation / test |
| `tests/` | Integration test |
| `notebooks/` | Google Colab notebook for GPU training |

## 11. Possible future work

- Test on an outside dataset of children's X-rays.
- Resize every lung region to a common resolution, to remove the remaining size shortcut.
- Collect TB images from several sources, so the TB class learns the disease and not the source.
