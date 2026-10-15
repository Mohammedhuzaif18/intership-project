# Interpretable Deep Learning for Respiratory Disorder Detection from Chest X-Rays

Internship project: Normal vs Pneumonia classification (Review-I), then Grad-CAM, a FastAPI + web app, and a multi-class extension (Review-II). Educational prototype, not a diagnostic tool.

**Before doing anything, read `PROGRESS.md` and continue from the first unticked step.**

## Folder structure

```
intership/
├── PROGRESS.md              # step checklist (source of truth for resuming)
├── README.md                # project overview, decisions log, results, commands
├── REVIEW1_SUMMARY.md       # Review-I write-up
├── REVIEW2_SUMMARY.md       # Review-II write-up
├── requirements.txt
├── src/
│   ├── config.py            # paths, seed, image size, class names
│   ├── utils.py             # logging, seeding, device detection, hashing, file helpers
│   ├── data/
│   │   ├── download.py      # Kermany via Hugging Face; Kaggle extras (env-var credentials only)
│   │   ├── quality.py       # EDA figures, corrupt-file and duplicate checks
│   │   ├── splits.py        # patient-grouped stratified splits -> ./splits/*.csv
│   │   ├── preprocessing.py # CLAHE -> RGB -> 224x224 (shared by training and the API)
│   │   ├── loaders.py       # tf.data pipelines, augmentation, class weights
│   │   ├── prepare.py       # runs download -> QC -> split -> preprocess for the core dataset
│   │   └── multiclass.py    # Phase 3: merge COVID-QU-Ex / TB sources
│   ├── models.py            # baseline CNN, ResNet50 (+ MobileNetV2 fallback)
│   ├── train.py             # training entry point (binary / multiclass)
│   ├── evaluate.py          # test-set metrics and figures
│   ├── gradcam.py           # Grad-CAM + sample report
│   └── cross_dataset.py     # Phase 3 cross-source generalization test
├── backend/main.py          # FastAPI app (/health, /predict, /samples)
├── frontend/                # index.html, style.css, app.js
├── tests/test_integration.py
├── notebooks/               # Colab-ready training notebook (+ generator script)
├── splits/                  # committed file lists per split
├── results/                 # figures, tables, logs (committed)
├── models/                  # trained models + class_names.json (git-ignored)
└── data/                    # raw + preprocessed images (git-ignored)
```

## Dataset rules

- Core: Kermany chest X-ray (Normal vs Pneumonia). Download only with
  `hf_hub_download(repo_id="AiresPucrs/chest-xray", filename="chest_xray.zip", repo_type="dataset")`.
  Never use `datasets.load_dataset()` for this repo (label-parsing error).
- Merge the provided folders and re-split 80/10/10, stratified, seed 42, **grouped by patient ID** parsed from filenames
  (`person<id>_...` for pneumonia, `[NORMAL2-]IM-<id>-...` for normal). No patient may appear in two splits.
- Remove corrupt files and exact duplicates (SHA-256) before splitting. Never let a duplicate cross splits.
- Report metrics from the **test split only**. Model selection uses the validation split.
- If test accuracy > 99%, investigate leakage and source shortcuts and report it.
- Kaggle credentials only from env vars `KAGGLE_USERNAME` / `KAGGLE_KEY`. Never hardcode, print or commit them.
  If missing, skip the multi-class extension and say so in the README.
- Seed 42 everywhere (Python, NumPy, TensorFlow, split functions).

## Run commands (Windows, Git Bash; from the project root)

```bash
python -m venv --system-site-packages .venv   # created from Python 3.12
.venv/Scripts/python -m pip install -r requirements.txt

.venv/Scripts/python -m src.data.prepare        # download, QC, EDA, split, CLAHE preprocessing
.venv/Scripts/python -m src.train --task binary # baseline CNN + ResNet50, evaluation, best model
.venv/Scripts/python -m src.gradcam             # Grad-CAM report -> results/gradcam

.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000   # app at http://127.0.0.1:8000
.venv/Scripts/python tests/test_integration.py

# Phase 3 (needs KAGGLE_USERNAME / KAGGLE_KEY)
.venv/Scripts/python -m src.data.multiclass
.venv/Scripts/python -m src.train --task multiclass
.venv/Scripts/python -m src.cross_dataset
```

`QUICK_TRAIN=1` runs a few batches per stage to check the pipeline; it does not produce usable models.
