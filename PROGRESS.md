# PROGRESS

**Resume rule:** a new session reads this file and continues from the first unticked step. Ticked steps are finished; do not redo them.

## Setup

- [x] S1. Create PROGRESS.md: this checklist.
- [x] S2. Create CLAUDE.md: title, target folder structure, dataset rules, run commands.
- [x] S3. git repo initialised; `.gitignore` excludes data, models, kaggle.json, venvs; initial commit of the pre-spec code.

## REVIEW-I · Phase 1: Data and core ML

- [x] 1. Environment: `.venv` (Python 3.12, `--system-site-packages`), pinned requirements.txt installed; no GPU (TF on native Windows is CPU-only). Kept ResNet50 instead of the MobileNetV2 fallback because measured CPU speed (~3.5 min/epoch) fits the 5+3 epoch schedule; Colab notebook generated (see step 1b note in README).
- [x] 2. Data: Kermany downloaded via `hf_hub_download`, tree + counts logged (5856 images); train/val/test folders merged and re-split 80/10/10 by patient (StratifiedGroupKFold, seed 42, 3117 patients, 0 unparsed ids); file lists in `splits/binary/`; counts in `results/split_counts.csv`.
- [x] 3. EDA + QC: figures in `results/eda/`; 0 corrupt files, 32 exact duplicates removed (SHA-256, listed in `splits/binary/removed.csv`); 2,760 bacterial vs 1,485 viral pneumonia images.
- [x] 4. Preprocessing: grayscale -> 224x224 (INTER_AREA) -> CLAHE (clip 2.0, 8x8) -> RGB in `src/data/preprocessing.py`; normalization inside each model; train-only augmentation layers; before/after + augmentation figures in `results/`; balanced class weights (Normal 1.844, Pneumonia 0.686).
- [x] 5. Baseline CNN: 4 conv blocks (422,881 params), 8 epochs; test acc 94.2%, F1 0.961, AUC 0.986.
- [x] 6. ResNet50: ImageNet, frozen phase early-stopped after 4 of 5 epochs (best = epoch 1), fine-tuned last 20 layers (BatchNorm frozen) 3 epochs at 1e-5, class weights; test acc 95.7%, F1 0.971, AUC 0.993.
- [x] 7. Test-only evaluation in `results/`: comparison table, confusion matrices, ROC curves, training curves, subtype accuracy (ResNet50 bacterial 98.2%, viral 94.6%), 95% CIs + McNemar (`statistical_comparison.json`), size-shortcut probe (AUC 0.912).
- [x] 8. ResNet50 selected on validation F1 (0.962); saved to `models/best_model.keras` + `models/class_names.json` (per-task copies in `models/binary/`).
- [x] 9. REVIEW1_SUMMARY.md written; Review-I committed and tagged `review-1`.

## REVIEW-II · Phase 2: Explainability and app

- [x] 10. Grad-CAM (`src/gradcam.py`): layer found programmatically (`conv5_block3_out` for ResNet50, `block4_out` baseline, `out_relu` MobileNetV2); 10 test images incl. 4 wrong predictions in `results/gradcam/` (+ grid); shows reliance on image borders for some cases.
- [x] 11. FastAPI backend (`backend/main.py`): CORS, `GET /health`, `POST /predict` (class, confidence, all probabilities, base64 Grad-CAM overlay + heatmap + model input), `GET /samples/{class}`; shared preprocessing; classes from `models/class_names.json`.
- [x] 12. Frontend (`frontend/`): upload/drag-drop, real test-image samples, prediction + probability bars, Grad-CAM overlay with opacity and Original / Model input / Overlay / Compare views, disclaimer; checked in the browser at desktop and phone widths.
- [x] 13. Integration test (`tests/test_integration.py`) passes: starts uvicorn, real test images give the same class and probabilities as the offline evaluation, non-image -> 400, samples and frontend served. Fixed during testing: stale cached UI files (no-cache header + versioned assets), viewer sizing.

## REVIEW-II · Phase 3: Multi-class extension

- [x] 14. Datasets: COVID-QU-Ex (`anasmohammedtahir/covidqu`, CC BY-SA 4.0; 33,920 images verified) + TB Chest X-ray Database (`tawsifurrahman/tuberculosis-tb-chest-xray-dataset`, authors' copyright, cite; 3,500 Normal + 700 TB verified); Non-COVID infection merged into PNEUMONIA -> 4 classes NORMAL / PNEUMONIA / COVID / TB.
- [x] 15. Shortcut controls: identical CLAHE preprocessing; 3 exact + 8,644 near-duplicate copies removed (5,172 COVID-QU-Ex = Kermany images, 3,472 TB-db normals = COVID-QU-Ex images; thumbnail r >= 0.98, visually confirmed); Normal and Pneumonia balanced 1,200 per source; split stratified by class x source, grouped by patient / near-duplicate cluster; hash + cluster overlap asserted. 7,925 images, 6,339/793/793.
- [x] 16. Multi-class retrain with class weights (test n=793): baseline CNN acc 72.6%, macro F1 0.734, AUC 0.934; ResNet50 acc 82.4% (CI 79.5-84.8), macro F1 0.844, macro OvR AUC 0.962 (McNemar p=7e-8); per-class + per-source tables, CM, ROC in `results/multiclass/`.
- [x] 17. Cross-dataset (Normal vs Pneumonia, ResNet50): Kermany-trained 95.7% -> 73.9% on unseen COVID-QU-Ex (drop 21.8 pts); COVID-QU-Ex-trained 87.5% -> 74.3% on Kermany (drop 13.2 pts); specificity collapses on the unseen population. `results/cross_dataset/`.
- [x] 18. 4-class ResNet50 + class_names.json installed in `models/`; backend/frontend pick up the 4 classes; integration test re-run and passing; UI checked in the browser.
- [x] 19. REVIEW2_SUMMARY.md and final README written; committed and tagged `review-2`.
