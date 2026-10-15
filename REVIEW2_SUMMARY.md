# Review-II Summary: Explainability, Application and Multi-class Extension

> Educational prototype, not a diagnostic tool. All metrics come from held-out **test** images only.

## 1. Grad-CAM explanations

**Method.** Grad-CAM (Selvaraju et al., 2017) weights the last convolutional feature maps by the gradient of the predicted-class score and keeps the positive part. For the binary sigmoid model the score is *p* for Pneumonia and *1 - p* for Normal. `src/gradcam.py` finds the target layer programmatically: the last `Conv2D`/`DepthwiseConv2D`, then forward through the layers that keep its spatial size (BatchNorm, Add, ReLU). That gives `conv5_block3_out` for ResNet50 (7x7x2048), `block4_out` for the baseline CNN and `out_relu` for MobileNetV2. If no gradient reaches the layer, it raises an error instead of drawing a fake map.

**Report.** `python -m src.gradcam` explains 10 held-out test images with the selected ResNet50: 6 correct (3 Normal, 3 Pneumonia) and the 4 most confident mistakes (2 normals called pneumonia, 2 viral pneumonias called normal). Each figure shows the model input, the heatmap and the overlay. As a consistency check, every Grad-CAM forward pass reproduces exactly the prediction from the test evaluation.

| # | True | Predicted (confidence) | Where the heat is |
|---|---|---|---|
| 1–3 | Normal | Normal (92%, 97%, 88%) | Lower mediastinum, heart and diaphragm |
| 4 | Pneumonia (bacterial) | Pneumonia (98%) | Right lateral border and image edge, partly outside the lung |
| 5 | Pneumonia (viral) | Pneumonia (87%) | Both lung fields: the clinically expected pattern |
| 6 | Pneumonia (bacterial) | Pneumonia (>99.9%) | Lateral image borders and soft tissue |
| 7 | Normal | **Pneumonia** (94%) | Left chest wall and image border |
| 8 | Normal | **Pneumonia** (92%) | Image borders, near burned-in text (timestamp, "KV59") |
| 9–10 | Pneumonia (viral) | **Normal** (88%, 85%) | Trachea / mediastinum, the same pattern as the normal cases |

**What it shows.** Only some of the evidence lies in the lung fields. Several pneumonia decisions, including both false alarms, lean on image borders, chest wall and acquisition artifacts. That matches the Review-I finding that image size and aspect ratio alone predict the class (test ROC-AUC 0.912): pneumonia images are natively smaller and wider, so after squashing to 224x224 their borders and anatomy look systematically different. Grad-CAM is doing its job here by exposing that the model partly learned the dataset rather than the disease.

Figures: `results/gradcam/gradcam_grid.png` (all 10), `results/gradcam/01_...png` to `10_...png` (input | heatmap | overlay), `results/gradcam/gradcam_summary.csv`.

## 2. Web application

**Backend** (`backend/main.py`, FastAPI):

| Route | Purpose |
|---|---|
| `GET /health` | Model status, model name, task, class names and display names, Grad-CAM layer, stored test metrics |
| `POST /predict` | Image upload -> `predicted_class`, `confidence`, `class_probabilities` (all classes), `gradcam_overlay_base64`, plus the bare heatmap and the model input as base64 JPEGs |
| `GET /samples/{class}` | A random **held-out test** X-ray of that class, for the UI's sample buttons |
| `/` | The frontend |

The API applies exactly the training preprocessing (`src.data.preprocessing.preprocess_image`) and reads the classes from `models/class_names.json`, so swapping in the multi-class model needs no code change. CORS is enabled. Uploads over 20 MB return 413, unreadable files 400, and a missing model 503.

**Frontend** (`frontend/`, plain HTML/CSS/JS, dark radiology-viewer style):

- Drag-and-drop or file upload, or one click on a held-out test image per class; those show the true label and whether the model got it right.
- Prediction with confidence, probability bars for every class, and a low-confidence note below 70%.
- Grad-CAM viewer with four views: **Original**, **Model input** (the CLAHE 224x224 image the network sees), **Overlay** (heatmap stretched exactly over the full-resolution upload, with an opacity slider) and **Compare** (side by side).
- A heatmap legend that explains warm regions as *what the model relied on, not confirmed pathology*, plus a technical-details panel.
- A permanent disclaimer: "Educational prototype, not a diagnostic tool".
- Responsive layout: two columns on desktop, stacked on phones, checked at 1280 px and 375 px.

**Screenshots** (`results/app/`):

- `app_desktop_compare.jpg`: a held-out Normal image in Compare view; prediction Normal 72.5%, ground truth shown as matching.
- `app_analysis_panel.jpg`: the analysis panel for a viral pneumonia test image, with probability bars and heatmap guide.
- `app_mobile_overlay.jpg`: the phone layout with the Grad-CAM overlay and opacity slider.

**Run it**

```bash
.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

**Integration test** (`python tests/test_integration.py`) starts the server on a free port and checks:

- `/health` loads ResNet50 with the right classes and layer.
- For three real test X-rays (one per class, plus a misclassified one), `/predict` returns the **same class and probabilities as the offline test evaluation**, within 0.002. This proves the API uses the training preprocessing.
- Probabilities sum to 1, and all base64 images decode to 224x224.
- A non-image upload returns 400; `/samples` returns real test images and 404 for unknown classes; the page is served with the disclaimer.

Result: **all checks pass.** Fixed during testing:

- Browsers kept serving the previous UI from cache. The API now sends `Cache-Control: no-cache` for UI files, and the asset links are versioned.
- The Model input view was drawn at 224 px. The frames are now sized from each image's aspect ratio.

## 3. Multi-class extension (Normal, Pneumonia/Non-COVID infection, COVID-19, TB)

### Data

| Source | Kaggle slug | License | Classes used |
|---|---|---|---|
| Kermany (Review-I, cleaned) | n/a (Hugging Face mirror) | CC BY 4.0 | Normal, Pneumonia |
| COVID-QU-Ex | `anasmohammedtahir/covidqu` | CC BY-SA 4.0 | Normal, Non-COVID infection (merged into Pneumonia), COVID-19 |
| TB Chest X-ray Database | `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` | (c) original authors, research use with citation | Normal, Tuberculosis |

**Chosen TB dataset:** the Tuberculosis (TB) Chest X-ray Database (Rahman et al., IEEE Access 2020). It was verified through the Kaggle metadata API and after download: `Normal` (3,500) and `Tuberculosis` (700) folders, matching its card. It was picked over the only CC-licensed alternative because its provenance is documented and peer-reviewed. Classes were also verified for COVID-QU-Ex: 11,956 COVID-19, 11,263 Non-COVID and 10,701 Normal images in `Lung Segmentation Data`; the Infection Segmentation subset repeats the same images, so it is not used.

**Cross-source duplicates.** COVID-QU-Ex lists Kermany among its sources, and the TB database is from the same group, so exact (SHA-256) and near-duplicate checks were run across all three. Near-duplicates are defined as a 32x32 mean-centred thumbnail correlation r >= 0.98, calibrated so that resized copies score r = 1.000 while different patients score at most 0.972. Results:

- **5,172 COVID-QU-Ex images are Kermany images** (1,567 Normal, 3,605 Non-COVID), stored as 256x256 PNGs.
- **3,472 of the 3,500 TB-database normals are COVID-QU-Ex normals**, stored at 512x512.
- 3 exact duplicates inside the TB set.

Every match is label-consistent, and the sampled pairs are pixel-for-pixel the same image (`results/multiclass/near_duplicate_examples.png`). The lower-priority copy is dropped (Kermany > COVID-QU-Ex > TB database). Without this step the split would leak, and a "cross-dataset" test would largely re-test training images.

**Shortcut controls and split.**

- Identical CLAHE preprocessing for every source.
- In Normal and Pneumonia, each source with at least 100 images contributes 1,200 images; the 28 unique TB-database normals are kept whole.
- COVID-19 is capped at 2,400; TB keeps all 697 unique images.
- The split is 80/10/10, stratified by class x source, and grouped by patient (Kermany) or near-duplicate cluster (7,449 groups). Hash and cluster overlap across splits are asserted.

| Class | Train | Val | Test | Sources (per split, train) |
|---|---|---|---|---|
| Normal | 1,942 | 243 | 243 | Kermany 960, COVID-QU-Ex 960, TB database 22 |
| Pneumonia / Non-COVID infection | 1,920 | 240 | 240 | Kermany 960, COVID-QU-Ex 960 |
| COVID-19 | 1,920 | 240 | 240 | COVID-QU-Ex |
| Tuberculosis | 557 | 70 | 70 | TB database |
| **Total** | **6,339** | **793** | **793** | |

### Results (held-out test set, n = 793)

Both models were retrained with the Review-I protocol (class weights; ResNet50 5 frozen + 3 fine-tune epochs, BatchNorm frozen) and selected on validation macro F1.

| Model | Accuracy (95% CI) | Macro precision | Macro recall | Macro F1 | Macro OvR ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN | 72.6% (69.4–75.6) | 0.813 | 0.712 | 0.734 | 0.934 |
| **ResNet50** (selected) | **82.4%** (79.5–84.8) | **0.847** | **0.845** | **0.844** | **0.962** |

ResNet50 is significantly better here (exact McNemar p = 7e-8; 140 vs 63 discordant images). The baseline over-predicts COVID-19: recall 0.96 but precision 0.59, and 72 normals labelled COVID-19.

Per class, ResNet50 on the test set:

| Class | Precision | Recall | F1 | OvR ROC-AUC | Support |
|---|---|---|---|---|---|
| Normal | 0.761 | 0.864 | 0.809 | 0.945 | 243 |
| Pneumonia / Non-COVID infection | 0.801 | 0.788 | 0.794 | 0.935 | 240 |
| COVID-19 | 0.895 | 0.783 | 0.836 | 0.968 | 240 |
| Tuberculosis | 0.930 | 0.943 | 0.936 | 0.999 | 70 |

Confusion matrix (rows true, columns predicted: N, P, C, TB): N [210, 20, 8, 5], P [40, 189, 11, 0], C [25, 27, 188, 0], TB [1, 0, 3, 66]. Most errors are Pneumonia <-> Normal and COVID-19 -> Pneumonia/Normal, the clinically hard boundaries.

**Source effect within a class** (ResNet50 test accuracy, 95% CI):

| Class | Kermany | COVID-QU-Ex |
|---|---|---|
| Normal | 90.8% (84.3–94.8) | 83.3% (75.7–88.9) |
| Pneumonia / Non-COVID | 87.5% (80.4–92.3) | **70.0%** (61.3–77.5) |

For the same label, the model is much better on Kermany's pediatric single-centre images than on COVID-QU-Ex's adult multi-centre ones. The Pneumonia gap is outside both confidence intervals. **TB's near-perfect AUC (0.999) should not be read as solved TB detection:** every TB image comes from one dataset, so the model can partly recognise that dataset's look; this data cannot separate the two.

**Grad-CAM, multi-class** (`results/multiclass/gradcam/`):

- **Plausible:** the COVID-19 case highlights bilateral lung opacities, and the TB case highlights the left upper lobe, a typical TB site.
- **Shortcuts:** a COVID-19 image misread as Pneumonia has its peak on a burned-in "D" side marker, and other errors sit on the borders or mediastinum.

The multi-class ResNet50 is now the model the app serves (`models/best_model.keras` + `models/class_names.json`); the UI picks up the four class names from the API.

The integration test was re-run against the 4-class model and passes. One real test image per class, from all three sources, plus a misclassified COVID-19 case, gives the same class and probabilities through the API as in the offline evaluation. The UI shows four sample buttons and four probability bars (`results/app/app_multiclass_tb.jpg`). In that screenshot the TB prediction (99.9%) has its strongest Grad-CAM heat on a bright label patch at the bottom edge, a visible example of the single-source TB concern above.

## 4. Cross-dataset generalization

**Design.** Normal vs Pneumonia/infection is the only task that two independent sources share, so it is the one that can be tested on an unseen source. Both directions use ResNet50 with the same protocol and test images only.

- **A: trained on Kermany** (the Review-I model; pediatric, one centre). Tested on its own Kermany test split and on **2,400 COVID-QU-Ex** Normal / Non-COVID images (adults, many centres) that it never saw. None of them is a copy of a Kermany image, thanks to the deduplication above.
- **B: trained on COVID-QU-Ex** Normal / Non-COVID (1,920 train images). Tested on COVID-QU-Ex test images and on the **Kermany** test split.

| Trained on | Tested on | Seen in training | n | Accuracy | Recall (pneumonia) | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Kermany | Kermany | yes | 583 | 95.7% | 0.969 | 0.924 | 0.971 | 0.993 |
| Kermany | COVID-QU-Ex | **no** | 2,400 | **73.9%** | 0.897 | **0.581** | 0.774 | 0.873 |
| COVID-QU-Ex | COVID-QU-Ex | yes | 240 | 87.5% | 0.825 | 0.925 | 0.868 | 0.933 |
| COVID-QU-Ex | Kermany | **no** | 583 | **74.3%** | 0.918 | **0.272** | 0.839 | 0.660 |

**Accuracy drop on the unseen source: 21.8 points (Kermany -> COVID-QU-Ex) and 13.2 points (COVID-QU-Ex -> Kermany).**

**What it means.**

- Both models keep catching pneumonia on the new source (recall about 0.9), but they **mistake the other population's healthy X-rays for disease**. Specificity falls to 0.58 and 0.27.
- What each model learned as "normal" is tied to its training population: pediatric single-centre vs adult multi-centre images, with different devices, positioning and post-processing. Kermany's 95.7% in-distribution accuracy therefore overstates how the model would do in another hospital.
- The ranking quality drops too (AUC 0.993 -> 0.873, and 0.933 -> 0.660), so this is not only a threshold problem.

Files: `results/cross_dataset/cross_dataset_results.md`, `accuracy_drop.csv`, and four confusion matrices.

## 5. Limitations

1. **Dataset shortcuts are real and measurable.**
   - In Kermany, image size alone predicts the class (test AUC 0.912).
   - In the multi-class set, accuracy depends on the source within a class, and TB comes from a single source.
   - Grad-CAM repeatedly shows heat on borders, side markers and label patches.

   The reported accuracies are upper bounds for these datasets, not estimates of clinical performance.
2. **Poor cross-population transfer:** an accuracy drop of 13–22 points on an unseen source.
3. **Duplicates across public datasets.** About 5,200 COVID-QU-Ex images are Kermany images, and almost all TB-database normals are COVID-QU-Ex images. These were removed here, but published results that mix these datasets without checking are likely inflated. The near-duplicate check catches resized or re-encoded copies, not crops or heavy edits.
4. **No patient IDs** in COVID-QU-Ex or the TB database, so different X-rays of the same patient there could still sit in different splits. Only near-identical images are grouped.
5. **Labels** come from the source datasets (image-level, no radiologist re-read); "Pneumonia" merges bacterial, viral and other non-COVID infections.
6. **Compute:** CPU-only training with a fixed small epoch budget (5 + 3) and no hyperparameter search. A GPU run (Colab notebook) could train longer.
7. **Grad-CAM is coarse** (7x7 for ResNet50) and shows influence, not causation or localization of disease.

## 6. Future scope

- **Remove the size/aspect shortcut:** crop to the lung region (COVID-QU-Ex ships lung masks) or centre-crop to square before resizing; then re-check the size-only probe and Grad-CAM.
- **Train on several sources and evaluate leave-one-source-out**, add site/source-invariant training (e.g. domain-adversarial heads), and calibrate the probabilities per source.
- Add an external TB source (e.g. NIAID TB Portals, which needs a data agreement), so that TB is not single-source.
- Use lung-mask-guided attention or report how much of the Grad-CAM energy falls inside the lung masks, as an automatic shortcut metric.
- Report sensitivity/specificity at clinically chosen thresholds, with confidence intervals and calibration plots, and add DICOM upload to the app.

## 7. How to run everything

```bash
.venv/Scripts/python -m src.data.prepare                 # Review-I data
.venv/Scripts/python -m src.train --task binary          # Review-I models
.venv/Scripts/python -m src.gradcam                      # Grad-CAM report
# Phase 3 (Kaggle credentials in KAGGLE_USERNAME / KAGGLE_KEY, or local folders)
.venv/Scripts/python -m src.data.multiclass --covidqu-dir data/raw/covidqu --tb-dir data/raw/tb_database
.venv/Scripts/python -m src.train --task multiclass      # also makes it the app's model
.venv/Scripts/python -m src.gradcam --task multiclass
.venv/Scripts/python -m src.cross_dataset --reverse
# App
.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000   # open http://127.0.0.1:8000
.venv/Scripts/python tests/test_integration.py
```

To serve the binary model again instead of the 4-class one, copy `models/binary/resnet50.keras` to `models/best_model.keras` and `models/binary/class_names.json` to `models/class_names.json`.

