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

<!-- PHASE3 -->
