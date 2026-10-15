# Interpretable Deep Learning for Respiratory Disorder Detection from Chest X-Rays

> **Educational prototype, not a diagnostic tool.** Not an FDA/CE certified medical device. Never use it for clinical decisions.

An end-to-end pipeline that classifies chest X-rays as **Normal** or **Pneumonia**, explains each prediction with **Grad-CAM**, and serves both through a **FastAPI** backend and a web frontend. The project is split into Review-I (data, baseline CNN, ResNet50 transfer learning, evaluation) and Review-II (Grad-CAM, app, multi-class extension). Progress is tracked step by step in [PROGRESS.md](PROGRESS.md); the Review write-ups are [REVIEW1_SUMMARY.md](REVIEW1_SUMMARY.md) and [REVIEW2_SUMMARY.md](REVIEW2_SUMMARY.md).

## Results (held-out test set only)

### Review-I: Normal vs Pneumonia (Kermany, patient-level split, test n = 583)

| Model | Accuracy (95% CI) | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline CNN | 94.2% (92.0–95.8) | 0.935 | 0.988 | 0.817 | 0.961 | 0.986 |
| **ResNet50** (selected on validation F1) | **95.7%** (93.8–97.1) | 0.972 | 0.969 | 0.924 | 0.971 | 0.993 |

ResNet50 accuracy by subtype: bacterial pneumonia 98.2% (276 images), viral 94.6% (149), normal 92.4% (158). The accuracy difference between the two models is not statistically significant (exact McNemar p = 0.18), but ResNet50 has far fewer false alarms on normal X-rays (12 vs 29). Details: [REVIEW1_SUMMARY.md](REVIEW1_SUMMARY.md).

**Leakage and shortcut checks.** No patient and no file hash is shared between splits. Test accuracy is below 99%, so the >99% leakage rule did not trigger. EDA did find a dataset bias: image width, height and aspect ratio alone predict the class with test ROC-AUC 0.912, because pneumonia images are natively smaller and wider. Grad-CAM shows some predictions leaning on image borders.

### Review-II Phase 2: Grad-CAM and the app

- **Grad-CAM** on ResNet50 `conv5_block3_out` (found programmatically) for 10 test images, 4 of them wrong: `results/gradcam/`. Correct viral cases light up both lung fields. Several pneumonia calls, including both false alarms, lean on image borders, chest wall and burned-in text, which is consistent with the image-size bias.
- **App**: FastAPI `/health`, `/predict`, `/samples` and a responsive web UI with probability bars, an aligned heatmap overlay with four views, and a disclaimer. Screenshots are in `results/app/`.
- **Integration test passes**: through the API, real test X-rays get the same class and probabilities as the offline test evaluation. Details: [REVIEW2_SUMMARY.md](REVIEW2_SUMMARY.md).

<!-- RESULTS-III -->

## Datasets

| Dataset | Use | Source | License |
|---|---|---|---|
| Chest X-Ray Images (Pneumonia), Kermany et al. 2018 | Core Normal vs Pneumonia (5,856 images, pediatric, Guangzhou) | Hugging Face mirror `AiresPucrs/chest-xray` via `hf_hub_download` (no Kaggle needed) | CC BY 4.0, Mendeley Data doi:10.17632/rscbjbr9sj (the record also states research use) |
| COVID-QU-Ex (Tahir et al. 2021) | Phase 3: COVID-19, Non-COVID infection, Normal | Kaggle `anasmohammedtahir/covidqu` | CC BY-SA 4.0 (Kaggle metadata) |
| Tuberculosis (TB) Chest X-ray Database (Rahman et al., IEEE Access 2020) | Phase 3: TB, Normal | Kaggle `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` | "Data files © Original Authors", research use with citation |

The TB choice: the Kaggle metadata API was queried for the candidate TB sets. This one is peer-reviewed, published by its authors, documents its image sources (NLM Montgomery/Shenzhen, Belarus, NIAID, RSNA) and has the expected `Normal` (3,500) and `Tuberculosis` (700 public) folders. Its terms are the authors' copyright with a request for citation, which fits non-commercial educational use; the images are never redistributed (data is git-ignored). The only CC-licensed alternative found (`yasserhessein/tuberculosis-chest-x-rays-images`, 29 MB) has no stated provenance, so it was rejected.

## Decisions log

| # | Decision | Why |
|---|---|---|
| D1 | **Patient-level split.** All three provided folders (train, val, test) are merged, then re-split 80/10/10 with `StratifiedGroupKFold` (seed 42) grouped by patient ID and stratified by subtype (normal / bacterial / viral). | Kermany has several X-rays per patient. In an earlier image-level split, 278 of 374 pneumonia test patients (74%) also had X-rays in the training set. Patient IDs come from filenames: `person<id>_...` for pneumonia, `[NORMAL2-]IM-<id>-...` for normal. Pneumonia IDs are treated as global across the original folders (170 IDs occur in both original train and test), which can only make groups larger. All 5,824 kept images have a parseable ID, so no random fallback was needed. |
| D2 | The **original test folder is pooled too**, not kept as a separate test set. | The spec asks for an 80/10/10 split of the whole set, and the original folders are not patient-disjoint for pneumonia (see D1). |
| D3 | Exact duplicates (SHA-256) and corrupt files are removed **before** splitting; split files are re-checked for patient and hash overlap before every training run. | A duplicate in two splits is leakage. |
| D4 | **CLAHE after resizing**: grayscale, 224x224 (INTER_AREA), CLAHE (clip 2.0, 8x8 tiles), copied to 3 channels. | Applying CLAHE at a fixed resolution makes it behave the same for every source, which matters once datasets with different native sizes are mixed (Phase 3). One function (`src/data/preprocessing.py`) is used by training, Grad-CAM and the API. |
| D5 | Normalization lives **inside the models** (`Rescaling(1/255)` for the baseline, `resnet50.preprocess_input` for ResNet50). | Every caller feeds the same uint8 RGB array, so the API cannot drift from training. |
| D6 | Augmentation (rotation 15°, shift 0.1, zoom 0.1, horizontal flip, brightness and contrast ±10%) runs as Keras layers inside the model. Constant (black) fill. | These layers are active only in training mode, so validation, test and the API never see augmented images. Black fill avoids mirrored anatomy at the borders. |
| D7 | **Kept ResNet50 although there is no GPU.** The spec's fallback is MobileNetV2 with fewer epochs on CPU-only machines. | TensorFlow has no GPU support on native Windows. ResNet50 was measured at about 3.5 min per epoch on this CPU (i5-13420H), so the full spec schedule (5 frozen + 3 fine-tune epochs) takes about 30 minutes. ResNet50 is objective 3. MobileNetV2 stays available (`--backbone mobilenet_v2`), and the Colab notebook covers GPU training. |
| D8 | ResNet50 fine-tuning keeps **BatchNorm frozen**; if fine-tuning does not beat the frozen-backbone validation loss, the phase-1 weights are kept. Early stopping (patience 3, best weights restored) and LR reduction on plateau in every stage. | Re-estimating ImageNet BatchNorm statistics from batches of 32 is a common way fine-tuning goes wrong; the safeguard means phase 2 can only help. |
| D9 | **Model selection on validation F1**; every reported metric is from the test split. | The test set stays an unbiased estimate of the chosen model. |
| D10 | Class imbalance handled with `class_weight="balanced"` (Normal 1.844, Pneumonia 0.686), not resampling. | Keeps every image; the spec asks for class weights. |
| D11 | Venv created with `--system-site-packages` from Python 3.12. | The folder sits in OneDrive; reusing the already installed, version-pinned TensorFlow 2.21 avoids syncing ~2 GB of packages. The previous `.venv` was an empty Python 3.14 environment. |
| D12 | **Multi-class extension run with Kaggle data.** The credentials (env vars only) were set during the session; the downloads went through the Kaggle API, and the prep runs from the local folders (`--covidqu-dir`, `--tb-dir`). Before the real run, the whole Phase 3 code path was tested on stand-in folders with the real COVID-QU-Ex / TB layouts (not reported as results). | That test caught a flaw: a 64-bit dHash flagged most *different* chest X-rays as duplicates, so the check was replaced by thumbnail correlation (see D13). |
| D13 | **Cross-source near-duplicate removal.** COVID-QU-Ex's dataset card lists Kermany as a source, and the TB database is from the same group. Each image gets a 32x32 mean-centred thumbnail; a correlation of r >= 0.98 with an image from a higher-priority source (Kermany > COVID-QU-Ex > TB database) means it is dropped. Remaining near-duplicate clusters are kept inside one split. | Calibrated before use: resized and re-encoded copies score r = 1.000, and different patients score at most 0.972 (400 test vs 4,658 train images). Result: **5,172 COVID-QU-Ex images are Kermany images** (1,567 Normal, 3,605 Non-COVID) and **3,472 of the 3,500 TB-database normals are COVID-QU-Ex images**. Every match is label-consistent and the sampled pairs are visually identical (`results/multiclass/near_duplicate_examples.png`). SHA-256 alone would have missed all of them. |
| D14 | Phase 3 balancing: in Normal and Pneumonia, every source with at least 100 images contributes the same count (1,200); smaller leftovers (28 TB-database normals) are kept whole. COVID-19 (COVID-QU-Ex only) is capped at 2,400; TB keeps all 697 unique images. The split is stratified by class x source. | Stops the source from predicting the class within the shared classes, and keeps CPU training time reasonable. A first version used the smallest source count, which collapsed Normal to 84 images once the duplicates were gone; it was caught before training finished and fixed. TB comes from a single dataset, so a TB-vs-source shortcut cannot be ruled out (see the per-source accuracy). |

## Project structure

```
intership/
├── src/
│   ├── config.py            # paths, seed, image size, CLAHE settings, class names
│   ├── utils.py             # logging, seeding, device detection, hashing
│   ├── plotting.py          # figure style (validated categorical palette)
│   ├── data/
│   │   ├── download.py      # Hugging Face download; Kaggle (env-var credentials only)
│   │   ├── quality.py       # corrupt/duplicate checks, EDA figures
│   │   ├── splits.py        # patient-grouped stratified splits
│   │   ├── preprocessing.py # CLAHE pipeline shared by training, Grad-CAM and the API
│   │   ├── loaders.py       # tf.data pipelines, augmentation, class weights
│   │   ├── prepare.py       # Review-I data pipeline entry point
│   │   └── multiclass.py    # Phase 3 multi-source dataset
│   ├── models.py            # baseline CNN, ResNet50 / MobileNetV2 transfer models
│   ├── train.py             # training, evaluation, model selection
│   ├── evaluate.py          # metrics and figures
│   ├── gradcam.py           # Grad-CAM + 10-image report
│   └── cross_dataset.py     # Phase 3 cross-source generalization test
├── backend/main.py          # FastAPI app
├── frontend/                # index.html, style.css, app.js
├── tests/test_integration.py
├── notebooks/               # Colab notebook + generator
├── splits/                  # file lists: binary/{train,val,test,removed}.csv
├── results/                 # figures, tables, logs
├── models/                  # trained models + class_names.json (git-ignored)
└── data/                    # raw + preprocessed images (git-ignored)
```

## How to run

From the project root, in Git Bash (use `.venv\Scripts\python` in cmd/PowerShell):

```bash
# 1. Environment (Python 3.12)
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r requirements.txt

# 2. Data: download, quality checks, EDA, patient-level split, CLAHE preprocessing
.venv/Scripts/python -m src.data.prepare

# 3. Train + evaluate the baseline CNN and ResNet50 (~1 h on CPU)
.venv/Scripts/python -m src.train --task binary

# 4. Grad-CAM report (10 test images)
.venv/Scripts/python -m src.gradcam

# 5. App: then open http://127.0.0.1:8000
.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 6. Integration test (starts its own server on a free port)
.venv/Scripts/python tests/test_integration.py
```

Phase 3 (needs `KAGGLE_USERNAME` and `KAGGLE_KEY` in the environment):

```bash
.venv/Scripts/python -m src.data.multiclass          # or: --covidqu-dir <folder> --tb-dir <folder>
.venv/Scripts/python -m src.train --task multiclass  # also makes it the model the app serves
.venv/Scripts/python -m src.gradcam --task multiclass
.venv/Scripts/python -m src.cross_dataset            # add --reverse to also train on COVID-QU-Ex
```

`QUICK_TRAIN=1` runs a few batches per stage to check the pipeline; it does not produce usable models. On a GPU, use [notebooks/ResNet50_Training_Colab.ipynb](notebooks/ResNet50_Training_Colab.ipynb).

## Environment

Python 3.12.10, TensorFlow 2.21 / Keras 3.15 (CPU), Windows 11, Intel i5-13420H, 16 GB RAM. Seed 42 for Python, NumPy, TensorFlow and every split.
