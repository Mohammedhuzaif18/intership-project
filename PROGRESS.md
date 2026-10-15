# PROGRESS

**Resume rule:** a new session reads this file and continues from the first unticked step. Ticked steps are finished; do not redo them.

## Setup

- [x] S1. Create PROGRESS.md: this checklist.
- [x] S2. Create CLAUDE.md: title, target folder structure, dataset rules, run commands.
- [x] S3. git repo initialised; `.gitignore` excludes data, models, kaggle.json, venvs; initial commit of the pre-spec code.

## REVIEW-I · Phase 1: Data and core ML

- [ ] 1. Environment: venv, requirements.txt, dependencies installed, GPU detection; MobileNetV2 fallback rule applied or justified; Colab-ready notebook.
- [ ] 2. Data: download Kermany via `hf_hub_download`, verify, print tree + counts, merge folders, patient-level stratified 80/10/10 split (seed 42), file lists in `./splits`, class counts per split.
- [ ] 3. EDA + quality checks: class-count bar chart, 8 samples per class, image-size stats, intensity histograms, corrupt + exact-duplicate removal (file hash) with counts, bacterial vs viral counts.
- [ ] 4. Preprocessing: CLAHE, RGB, 224x224, normalization; train-only augmentation (rotation 15, shift 0.1, zoom 0.1, h-flip, brightness/contrast); before/after examples; class weights.
- [ ] 5. Baseline CNN (3-4 conv blocks, up to 8 epochs).
- [ ] 6. ResNet50 transfer learning (ImageNet; 5 frozen epochs, then 3 fine-tune epochs on last ~20 layers at low LR; class weights).
- [ ] 7. Test-set-only evaluation: accuracy, precision, recall, F1, confusion matrix, ROC-AUC, bacterial vs viral accuracy; confusion matrices, ROC curves, training curves, comparison table in `./results`.
- [ ] 8. Best model saved as `.keras` + `class_names.json`.
- [ ] 9. REVIEW1_SUMMARY.md; commit; tag `review-1`.

## REVIEW-II · Phase 2: Explainability and app

- [ ] 10. Grad-CAM with programmatic target layer (`conv5_block3_out` for ResNet50); 10 test heatmaps incl. wrong predictions in `./results/gradcam`.
- [ ] 11. FastAPI backend: CORS, `GET /health`, `POST /predict` (class, confidence, all probabilities, base64 Grad-CAM overlay); training preprocessing; class names from `class_names.json`.
- [ ] 12. Frontend: upload, prediction, probability bars, Grad-CAM image, disclaimer.
- [ ] 13. Integration test: start backend, send a real test image to `/predict`, confirm response; fix breakages.

## REVIEW-II · Phase 3: Multi-class extension

- [ ] 14. Merge COVID-QU-Ex Non-COVID infection with Kermany Pneumonia; classes Normal, Pneumonia/Non-COVID infection, COVID-19 (+ TB if a verified dataset is available).
- [ ] 15. Shortcut mitigation: identical preprocessing, per-class source balancing, cross-split duplicate check by file hash.
- [ ] 16. Retrain baseline CNN + ResNet50 multi-class with class weights; test metrics incl. per-class P/R/F1, confusion matrix, OvR ROC-AUC in `./results/multiclass`.
- [ ] 17. Cross-dataset test: train on some sources, test on an unseen source, report accuracy drop.
- [ ] 18. Save multi-class model + updated class_names.json; backend + frontend use it.
- [ ] 19. REVIEW2_SUMMARY.md; README.md final update; commit; tag `review-2`.
