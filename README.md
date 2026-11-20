# Interpretable Deep Learning for Respiratory Disorder Detection from Chest X-Rays

> **Educational prototype, not a diagnostic tool.** Not an FDA/CE certified medical device. Never use it for clinical decisions.

An end-to-end pipeline that classifies chest X-rays as **Normal** or **Pneumonia** (Review-I) and then as **Normal, Pneumonia / Non-COVID infection, COVID-19 or Tuberculosis** (Review-II). It explains each prediction with **Grad-CAM**, serves the model through a **FastAPI** backend and a web frontend, and measures how well the models transfer to an unseen dataset. The app also includes technical image-quality screening, persistent patient-wise X-ray history with visual comparisons, and an optional patient/doctor review workflow. The project is split into Review-I (data, baseline CNN, ResNet50 transfer learning, evaluation) and Review-II (Grad-CAM, app, multi-class extension). Progress is tracked step by step in [PROGRESS.md](PROGRESS.md); the Review write-ups are [REVIEW1_SUMMARY.md](REVIEW1_SUMMARY.md) and [REVIEW2_SUMMARY.md](REVIEW2_SUMMARY.md).

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
- **App**: FastAPI `/health`, `/quality`, `/predict`, `/samples` and patient history / comparison endpoints. The responsive UI includes probability bars, Grad-CAM views, patient history, technical image checks, and a disclaimer. Screenshots are in `results/app/`.
- **Integration test passes**: through the API, real test X-rays get the same class and probabilities as the offline test evaluation. Details: [REVIEW2_SUMMARY.md](REVIEW2_SUMMARY.md).

### Review-II Phase 3: Multi-class extension (Kermany + COVID-QU-Ex + TB database, test n = 793)

| Model | Accuracy (95% CI) | Macro precision | Macro recall | Macro F1 | Macro OvR ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN | 72.6% (69.4–75.6) | 0.813 | 0.712 | 0.734 | 0.934 |
| **ResNet50** (selected on validation macro F1) | **82.3%** (79.5–84.8) | 0.847 | 0.845 | 0.844 | 0.962 |

Per class, ResNet50 F1 is Normal 0.809, Pneumonia / Non-COVID infection 0.794, COVID-19 0.836 and Tuberculosis 0.936 (`results/multiclass/per_class_metrics.md`). ResNet50 beats the baseline significantly (McNemar p = 7e-8). Within a class, accuracy depends on the source: for Pneumonia it is 87.5% on Kermany images vs 70.0% on COVID-QU-Ex images. TB comes from one dataset only, so its AUC of 0.999 likely includes a source shortcut.

**Cross-dataset test (Normal vs Pneumonia, ResNet50, test images only):**

| Trained on | Seen-source accuracy | Unseen-source accuracy | Drop |
|---|---|---|---|
| Kermany | 95.7% (Kermany) | 73.9% (COVID-QU-Ex, n = 2,400) | **21.8 points** |
| COVID-QU-Ex | 87.5% (COVID-QU-Ex) | 74.3% (Kermany, n = 583) | **13.2 points** |

Both models keep about 90% pneumonia recall on the new source, but their specificity falls to 0.58 and 0.27: each mistakes the other population's healthy X-rays for disease. **Dataset hygiene finding:** 5,172 COVID-QU-Ex images are copies of Kermany images and 3,472 of the 3,500 TB-database normals are copies of COVID-QU-Ex images. They were found by thumbnail correlation, removed, and confirmed visually.

**External test (TBX11K; no model was trained on it):** `results/external_tbx11k/REPORT.md`. Shenzhen, Montgomery and RSNA could not serve: they are already inside the TB database. 46 TBX11K images that were copies of training images were dropped first. On the rest, the app's 4-class ResNet50 finds **5.8% of 799 TB images** (TB-vs-healthy AUC 0.838), against 94.3% recall and AUC 0.999 on its held-out test split. The TB result does not transfer to a new source. Healthy adults are still called Normal 92.9% of the time.

## Objectives

| # | Objective | Status | Where |
|---|---|---|---|
| 1 | Collect, analyse, clean and preprocess (EDA, quality checks, CLAHE, resize, normalize, imbalance, augmentation) | Done | `src/data/`, `results/eda/`, `results/preprocessing_before_after.png` |
| 2 | Baseline CNN | Done | `src/models.py`; test acc 94.2% |
| 3 | ResNet50 transfer learning vs baseline | Done | test acc 95.7%, AUC 0.993; `results/model_comparison.md` |
| 4 | Accuracy, precision, recall, F1, confusion matrix, ROC-AUC; class imbalance | Done | `results/`, class weights, CIs + McNemar |
| 5 | Grad-CAM | Done | `src/gradcam.py`, `results/gradcam/`, `results/multiclass/gradcam/` |
| 6 | FastAPI backend + web frontend (class, confidence, Grad-CAM) | Done | `backend/`, `frontend/`, `tests/test_integration.py` (passes) |
| 7 | Multi-class (Normal, Pneumonia/Non-COVID, COVID-19, TB) + cross-dataset test | Done (TB included) | `results/multiclass/`, `results/cross_dataset/` |
| 8 | Technical image quality screening + persistent patient history and visual comparisons | Done | `backend/image_quality.py`, `backend/patient_history.py`, `frontend/`, `tests/test_integration.py` |
| 9 | Optional patient/doctor authentication, review requests, doctor decisions, and separate AI/doctor reports | Done | `backend/auth.py`, `backend/review_store.py`, `backend/main.py`, `frontend/auth.html`, `frontend/auth.js` |


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
├── backend/                 # FastAPI app, image quality checks, patient history storage
├── frontend/                # index.html, results.html, shared/home/results scripts, style.css
├── tests/test_integration.py
├── notebooks/               # Colab notebook + generator
├── splits/                  # file lists: binary/, multiclass/ ({train,val,test,removed}.csv), cross_dataset/
├── results/                 # figures, tables, logs (eda/, gradcam/, app/, multiclass/, cross_dataset/, logs/)
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

# 5. App: fit the non-chest-image gate for the served model, then open http://127.0.0.1:8000
.venv/Scripts/python -m src.gate
.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 6. Integration test (starts its own server on a free port)
.venv/Scripts/python tests/test_integration.py
```

### Image quality and patient history

The upload page first calls `POST /quality`; `/predict` repeats the quality checks before running the existing chest-image gate, ResNet50 prediction, and Grad-CAM path. Any detected quality issue—including blur, very dark/bright exposure, low resolution, or unusual framing—stops prediction and shows upload suggestions. Defaults are configurable with `XRAY_QUALITY_MIN_RESOLUTION` (128 px), `XRAY_QUALITY_MIN_SHARPNESS` (Laplacian variance 5), `XRAY_QUALITY_MIN_BRIGHTNESS` (20), `XRAY_QUALITY_MAX_BRIGHTNESS` (235), `XRAY_QUALITY_MIN_ASPECT_RATIO` (0.5), and `XRAY_QUALITY_MAX_ASPECT_RATIO` (2.0). These are technical checks only; passing does not establish medical reliability or diagnostic accuracy. The uploaded original is stored unchanged; quality checks do not replace or alter model preprocessing.

The UI asks for a Patient ID and sends it with the image to `POST /predict`. Existing API clients may omit it to use the original prediction-only workflow. With an ID, predictions and original / Grad-CAM / heatmap files are stored in SQLite at `data/patient_history/history.sqlite3` and under `data/patient_history/records/` (both within the git-ignored `data/` directory). Set `PATIENT_HISTORY_DIR` to move this local store. Patient IDs are limited to letters, numbers, dots, underscores and hyphens. Re-uploading the same image for a patient is allowed and creates a separate timestamped history record.

`GET /patients/{patient_id}/history` returns that patient's records in upload order. `GET /patients/{patient_id}/compare?record_a_id=…&record_b_id=…` only resolves records under the requested patient ID and returns an OpenCV feature-aligned difference map where registration succeeds. A report automatically compares a new record with the most recent prior record; the history page also supports selecting any two records. The plain-language summary explicitly compares model-predicted categories: a shift from a disease category to Normal is labeled a **possible improvement signal**, the reverse is a **possible new or worsening signal**, and same-category or disease-to-disease changes are explained without assigning a direction. It also notes that confidence is not disease severity. These are model-output interpretations, not confirmed clinical outcomes; a clinician must assess the patient.

### Optional patient accounts and doctor review

The sign-in and registration screen is a responsive, clinic-inspired portal with clear patient/doctor account options; it continues to send patients into the existing analysis workspace.

Open `http://127.0.0.1:8000/auth.html` to sign in or register. After patient sign-in, the app opens the existing main analysis interface at `/index.html`—the same image-quality checks, prediction, Grad-CAM report and patient-history experience. The account's name and Patient ID are populated in that interface. A signed-in patient also sees their saved X-ray reports on this main page, including each AI prediction and review status. The list refreshes automatically, so a patient who signs in after a doctor has submitted a recommendation sees the doctor's decision, observations, comments, name, review date and protected X-ray/Grad-CAM images. The report offers an optional doctor-review request; the AI-only result is available immediately and remains unchanged by a later doctor decision. Doctor sign-in opens the review queue. Patient and doctor passwords must be at least 6 characters; because six-character passwords are weaker, users should choose longer, unique passwords. Passwords are hashed with Argon2id. Doctors register with a four-digit `DOCTOR_REGISTRATION_CODE` supplied by the server operator; it should be chosen privately and is not a substitute for real-world credential verification. Signed HS256 JWTs expire after `AUTH_TOKEN_LIFETIME_SECONDS` (default 3600 seconds). For deployment, configure a persistent secret of at least 32 characters using `AUTH_JWT_SECRET` and protect it as a credential. If omitted, the app creates a local key in the configured data directory; tokens signed with that key remain valid across server restarts but are not portable across instances.

The authentication workflow uses the same SQLite database under `PATIENT_HISTORY_DIR`. Its account/review tables store users, patient and verified-doctor profiles, AI prediction metadata, review requests, and separate doctor decisions. An AI prediction is never overwritten by a doctor's approval or rejection. Patients can only access their own predictions/history; doctors only see unclaimed requests or requests assigned to them, and must claim a request before reading its protected images or submitting a review. Review status and completed doctor findings are shown on the existing analysis report. Access is appropriate for local educational testing only: it is not a production health-record system and has no clinical-grade identity verification, encryption-at-rest, audit/compliance controls, or notification delivery service.

Authentication and review API:

| Method | Endpoint | Access |
|---|---|---|
| `POST` | `/auth/patient/register` | Public patient registration |
| `POST` | `/auth/doctor/register` | Public, requires `DOCTOR_REGISTRATION_CODE` |
| `POST` | `/auth/login` | Public login; optional `role` check |
| `GET` | `/auth/me` | Any authenticated account |
| `POST` | `/auth/patient/predict` | Patient; runs existing quality + prediction + Grad-CAM flow |
| `GET` | `/auth/patient/predictions` | Patient's own AI reports and review status |
| `GET` | `/auth/patient/predictions/{prediction_id}/report` | Patient's own AI-only or completed doctor-reviewed report |
| `POST` | `/auth/patient/predictions/{prediction_id}/review-request` | Patient; opens one review request for their prediction |
| `GET` | `/auth/patient/predictions/{prediction_id}/images/{kind}` | Owning patient |
| `GET` | `/auth/doctor/reviews` | Doctor; available queue |
| `POST` | `/auth/doctor/reviews/{request_id}/claim` | Doctor; claims a request |
| `POST` | `/auth/doctor/reviews/{request_id}/submit` | Assigned doctor; records APPROVE/REJECT and observations |
| `GET` | `/auth/doctor/reviews/{request_id}/images/{kind}` | Doctor assigned to the request |

`kind` is `original`, `gradcam`, or `heatmap`. Doctor registration is intentionally code-gated. In PowerShell, choose a private four-digit code before starting the server and enter the same code in the portal's doctor-registration form:

```powershell
$env:DOCTOR_REGISTRATION_CODE = Read-Host "Choose a private 4-digit doctor code"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The code is available only to that server process; set it again whenever you restart the server. Do not commit the code or share it with patients. A four-digit code is suitable only for this local educational prototype; for production, use a real doctor verification and approval process rather than a shared registration code.

Phase 3 (Kaggle credentials only via the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables, or pass folders you downloaded yourself):

```bash
.venv/Scripts/python -m src.data.multiclass --covidqu-dir data/raw/covidqu --tb-dir data/raw/tb_database   # or no flags to download via Kaggle
.venv/Scripts/python -m src.train --task multiclass  # also makes it the model the app serves
.venv/Scripts/python -m src.gradcam --task multiclass
.venv/Scripts/python -m src.cross_dataset --reverse  # both directions
```

The app serves whichever task was trained last; it currently serves the **4-class ResNet50**. To serve the binary model instead, copy `models/binary/resnet50.keras` to `models/best_model.keras` and `models/binary/class_names.json` to `models/class_names.json`.

**Non-chest-image gate.** The classifier alone gives a disease for any image, including a face photo. `python -m src.gate` fits a check for the served model: an upload is compared with 400 training X-rays in the model's own feature space (cosine similarity, 5 nearest), and the threshold is set so about 99% of validation X-rays pass. Uploads below it get "Not a chest X-ray" and no disease, probabilities or heatmap. With the 4-class model the threshold is 0.768. It rejected 3 of 300 held-out test X-rays (1%), 3 of 240 held-out COVID-QU-Ex test X-rays (all pneumonia) and all 21 non-chest photos tried, among them the screenshot of a face photo that the app had labelled COVID-19 at 90.9%. The gate is saved in `models/chest_gate.npz` with the model file's hash, and is switched off if the served model changes; re-run `python -m src.gate` after retraining. It is a similarity check, not a guarantee: unusual images, such as other X-ray body parts, may still pass.

`QUICK_TRAIN=1` runs a few batches per stage to check the pipeline; it does not produce usable models. On a GPU, use [notebooks/ResNet50_Training_Colab.ipynb](notebooks/ResNet50_Training_Colab.ipynb).

## Environment

Python 3.12.10, TensorFlow 2.21 / Keras 3.15 (CPU), Windows 11, Intel i5-13420H, 16 GB RAM. Seed 42 for Python, NumPy, TensorFlow and every split.
