# Project Guide for the Examiner

**Interpretable Deep Learning for Respiratory Disorder Detection from Chest X-Rays**

> This is a student project for learning. It is **not** a medical tool and must not be used to diagnose anyone.

This guide explains the whole project in simple words: what was built, how, why each choice was made, what the results are, and what the weaknesses are. The full technical write-ups are [README.md](README.md), [REVIEW1_SUMMARY.md](REVIEW1_SUMMARY.md) and [REVIEW2_SUMMARY.md](REVIEW2_SUMMARY.md). A list of technical words is in [section 21](#21-glossary).

## Contents

1. [The problem](#1-the-problem)
2. [Results at a glance](#2-results-at-a-glance)
3. [How the system works](#3-how-the-system-works)
4. [Datasets](#4-datasets)
5. [Cleaning the data](#5-cleaning-the-data)
6. [Train, validation and test split](#6-train-validation-and-test-split)
7. [Preparing the images](#7-preparing-the-images)
8. [The two models](#8-the-two-models)
9. [How the models are measured](#9-how-the-models-are-measured)
10. [Results: Normal vs Pneumonia](#10-results-normal-vs-pneumonia)
11. [Results: 4 classes](#11-results-4-classes)
12. [Test on a different dataset](#12-test-on-a-different-dataset)
13. [Grad-CAM: explaining the decisions](#13-grad-cam-explaining-the-decisions)
14. [The web app](#14-the-web-app)
15. [Testing the app](#15-testing-the-app)
16. [Extra checks: did the model learn shortcuts?](#16-extra-checks-did-the-model-learn-shortcuts)
17. [Important decisions and why](#17-important-decisions-and-why)
18. [Limitations](#18-limitations)
19. [Future work](#19-future-work)
20. [How to run it](#20-how-to-run-it)
21. [Glossary](#21-glossary)
22. [Questions the examiner may ask](#22-questions-the-examiner-may-ask)
23. [Where to find everything](#23-where-to-find-everything)

---

## 1. The problem

- **Pneumonia** is a lung infection and one of the main causes of death in children under five.
- Doctors usually check for it with a **chest X-ray**, but reading X-rays needs a trained radiologist, and there are not enough of them in many places.
- A computer model can help by giving a first opinion. But a doctor should not trust a model that only says "Pneumonia, 95%". They need to see **why**.

So this project does two things:

1. **Classify** chest X-rays:
   - first Normal vs Pneumonia (Review-I);
   - then Normal, Pneumonia, COVID-19 or Tuberculosis (Review-II).
2. **Explain** each decision with a heatmap (**Grad-CAM**) that shows which parts of the X-ray the model used.

A big part of the work is **honest testing**: making sure the results are not inflated by mistakes such as the same patient appearing in both training and testing.

## 2. Results at a glance

All numbers come from the **test set**, which the models never saw during training.

| Task | Best model | Test accuracy | ROC-AUC |
|---|---|---|---|
| Normal vs Pneumonia (583 test images) | ResNet50 | **95.7%** | 0.993 |
| 4 classes: Normal, Pneumonia, COVID-19, TB (793 test images) | ResNet50 | **82.3%** | 0.962 |
| Normal vs Pneumonia on a **different, unseen dataset** (2,400 images) | ResNet50 | 73.9% | 0.873 |

What else was delivered:

- Grad-CAM heatmaps for every prediction.
- A working web app: upload an X-ray, get a printable report.
- An automatic integration test that passes.
- Several extra experiments that check whether the model really learned the disease (section 16).

## 3. How the system works

```
                TRAINING (offline)                                  USING THE APP
 ┌───────────────────────────────────────────┐       ┌──────────────────────────────────────────┐
 │ 1. Download datasets                      │       │ 1. User fills name, age, uploads X-ray   │
 │ 2. Remove broken files and duplicates     │       │ 2. Browser sends ONLY the image          │
 │ 3. Split by patient: 80 / 10 / 10         │       │ 3. Server: same preprocessing as training│
 │ 4. Preprocess: gray, 224x224, CLAHE, RGB  │  ──►  │ 4. Check: is this a chest X-ray?         │
 │ 5. Train baseline CNN and ResNet50        │ model │ 5. Model gives class + probabilities     │
 │ 6. Pick best model on validation set      │       │ 6. Grad-CAM makes the heatmap            │
 │ 7. Measure on test set only               │       │ 7. Report page: result, heatmap, print   │
 └───────────────────────────────────────────┘       └──────────────────────────────────────────┘
```

The **same preprocessing function** (`src/data/preprocessing.py`) is used for training, for Grad-CAM and inside the app. This guarantees that the app sees images exactly as the model saw them during training.

## 4. Datasets

| Dataset | What it contains | Used for | Licence |
|---|---|---|---|
| **Kermany et al. 2018** (Chest X-Ray Images, Pneumonia) | 5,856 X-rays of children aged 1–5 from one hospital in Guangzhou, China; Normal or Pneumonia (bacterial or viral) | Main dataset (Review-I) and part of the 4-class set | CC BY 4.0 |
| **COVID-QU-Ex** (Tahir et al. 2021) | 33,920 adult X-rays: COVID-19, other infections ("Non-COVID"), Normal, with lung masks | 4-class set and the different-dataset test | CC BY-SA 4.0 |
| **TB Chest X-ray Database** (Rahman et al. 2020) | 3,500 Normal + 700 Tuberculosis | 4-class set (TB class) | Authors' copyright, research use with citation |
| **TBX11K** (Liu et al. 2020) | About 8,400 X-rays: healthy, sick (not TB), TB | External test only, never used for training | CC BY 4.0 |

- Kermany was downloaded from Hugging Face (`hf_hub_download`). The others came from Kaggle.
- Kaggle passwords were only read from environment variables, never written in the code.
- The images are **not** stored in Git (the `data/` folder is ignored); only the lists of which file goes where are committed.

## 5. Cleaning the data

### Kermany

| Check | Result |
|---|---|
| Images found | 5,856 |
| Broken (corrupt) files | 0 |
| Exact duplicates (same file fingerprint, SHA-256) | 32 removed |
| **Images kept** | **5,824**: Normal 1,579 (27%), Pneumonia 4,245 (73%) |
| Pneumonia types (from the file names) | Bacterial 2,760, viral 1,485 |
| Patients | 3,117 |

### Copies hidden across datasets (important discovery)

When the three training datasets were combined, many images turned out to be **copies of each other**:

- **5,172 COVID-QU-Ex images are Kermany images** (resized to 256x256).
- **3,472 of the 3,500 TB-database normal images are COVID-QU-Ex images.**

**Why this matters.** If a copy of a training image lands in the test set, the model has already seen the answer, and the score is falsely high.

**Why a normal duplicate check misses them.** The copies were resized and saved again, so their file fingerprints (SHA-256) are different.

**How they were found:**

1. Each image was shrunk to a 32x32 thumbnail.
2. Every pair of thumbnails was compared (correlation). A score of r ≥ 0.98 means "same image".
3. The threshold was tested first: resized copies score 1.000, and different patients score at most 0.972.
4. Every match had the same label in both datasets, and sample pairs were checked by eye (`results/multiclass/near_duplicate_examples.png`).

All copies were removed before splitting.

## 6. Train, validation and test split

| Part | Share | Used for |
|---|---|---|
| **Train** | 80% | Teaching the model |
| **Validation** | 10% | Choosing the best model and the best epoch |
| **Test** | 10% | The final score. Used only once, at the end |

**Split by patient, not by image.** One child can have several X-rays. If one X-ray is in training and another in testing, the model may recognise the *child* rather than the disease. In an early version that split by image, 74% of the pneumonia test patients also had X-rays in training. So:

- All three original Kermany folders were merged and split again, grouped by patient ID (read from the file name), with the same mix of normal / bacterial / viral in each part (`StratifiedGroupKFold`, seed 42).
- The code checks automatically that **no patient and no file appears in two parts**, before every training run.

| Part | Normal | Pneumonia (bacterial / viral) | Total | Patients |
|---|---|---|---|---|
| Train | 1,263 | 3,395 (2,208 / 1,187) | 4,658 | 2,490 |
| Validation | 158 | 425 (276 / 149) | 583 | 313 |
| Test | 158 | 425 (276 / 149) | 583 | 314 |

**4-class set** (7,925 images: 6,339 train / 793 validation / 793 test):

| Class | Train | Val | Test | Where the images come from |
|---|---|---|---|---|
| Normal | 1,942 | 243 | 243 | Kermany + COVID-QU-Ex (equal amounts) + 28 TB-database images |
| Pneumonia / Non-COVID infection | 1,920 | 240 | 240 | Kermany + COVID-QU-Ex (equal amounts) |
| COVID-19 | 1,920 | 240 | 240 | COVID-QU-Ex |
| Tuberculosis | 557 | 70 | 70 | TB database |

Normal and Pneumonia take the **same number of images from each dataset**, so the model cannot guess the class just from which dataset an image looks like.

## 7. Preparing the images

| Step | What happens | Why |
|---|---|---|
| 1. Grayscale | Convert to one 8-bit gray channel | X-rays are gray; some files were saved as colour |
| 2. Resize | 224x224 pixels | The input size ResNet50 expects |
| 3. **CLAHE** | Local contrast enhancement (clip 2.0, 8x8 tiles) | Makes lung details clearer. Done *after* resizing, so it works the same for every dataset |
| 4. RGB | Copy the gray channel 3 times | ResNet50 was pre-trained on colour images |
| 5. Normalise | Scale pixel values, inside the model itself | The app cannot forget this step, because it is part of the model |

Figure: `results/preprocessing_before_after.png`.

**Augmentation** (training only). Each training image is randomly changed a little every time, so the model learns the disease and not exact pixels:

- rotation up to 15°
- shift up to 10%
- zoom up to 10%
- horizontal flip
- brightness and contrast up to ±10%

Validation, test and the app never see augmented images. Figure: `results/augmentation_examples.png`.

**Class imbalance.** There are about 2.7 times more pneumonia images than normal ones. Without a fix, the model could just say "Pneumonia" most of the time. **Class weights** make a mistake on a normal image count more:

- Normal 1.844
- Pneumonia 0.686

## 8. The two models

| | **Baseline CNN** (built from scratch) | **ResNet50** (transfer learning) |
|---|---|---|
| Idea | A small network that learns everything from the X-rays | A large network already trained on 1.2 million everyday photos (ImageNet), then adapted to X-rays |
| Layers | 4 blocks of: 3x3 convolution → batch normalisation → ReLU → max-pool (32, 64, 128, 256 filters); then global average pooling, dense 128, dropout 0.4, output | ResNet50 body; then global average pooling, dense 128, dropout 0.4, output |
| Parameters | 422,881 | 23.85 million |
| Training | Adam optimiser, learning rate 0.001, up to 8 epochs | **Step 1:** body frozen, only the new top trains (lr 0.001, up to 5 epochs). **Step 2:** the last 20 layers are unfrozen and fine-tuned slowly (lr 0.00001, 3 epochs) |

Rules used for both models:

- batch size 32, with class weights;
- **early stopping**: stop when the validation loss stops improving and go back to the best epoch;
- the learning rate is lowered when progress stalls;
- seed 42.

**Why ResNet50 on a laptop without a GPU?**

- TensorFlow cannot use a GPU on native Windows.
- The fallback in the task was MobileNetV2, a smaller network.
- ResNet50 was timed at about 3.5 minutes per epoch on this CPU (Intel i5-13420H), so the full training took about 1 hour. That was acceptable, so ResNet50 was kept.
- A Google Colab notebook for GPU training is also included (`notebooks/`).

**Choosing the final model.** The model with the best **validation** F1 score was chosen, which was ResNet50. The test set was not used for this choice.

## 9. How the models are measured

Pneumonia is the "positive" class.

| Measure | Simple meaning |
|---|---|
| **Accuracy** | Share of all X-rays classified correctly |
| **Precision** | When the model says "Pneumonia", how often it is right |
| **Recall** (sensitivity) | Of all real pneumonia cases, how many the model finds |
| **Specificity** | Of all healthy X-rays, how many the model correctly calls Normal |
| **F1 score** | One number that balances precision and recall |
| **ROC-AUC** | How well the model ranks sick above healthy, at any threshold (1.0 = perfect, 0.5 = guessing) |
| **Confusion matrix** | A table of true class vs predicted class: shows exactly which mistakes happen |
| **95% confidence interval** | The range where the true accuracy probably lies, given the test-set size |
| **McNemar test** | A statistical test of whether two models really differ on the same test images (p < 0.05 means a real difference) |

## 10. Results: Normal vs Pneumonia

Test set, 583 images:

| Model | Accuracy (95% CI) | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline CNN | 94.2% (92.0–95.8) | 0.935 | **0.988** | 0.817 | 0.961 | 0.986 |
| **ResNet50** | **95.7%** (93.8–97.1) | **0.972** | 0.969 | **0.924** | **0.971** | **0.993** |

**Confusion matrices** (rows = true class, columns = model's answer):

| | Baseline says Normal | Baseline says Pneumonia | ResNet50 says Normal | ResNet50 says Pneumonia |
|---|---|---|---|---|
| Truly Normal (158) | 129 | 29 | 146 | 12 |
| Truly Pneumonia (425) | 5 | 420 | 13 | 412 |

**Accuracy by type (ResNet50):**

| Type | Test images | Accuracy |
|---|---|---|
| Bacterial pneumonia | 276 | 98.2% |
| Viral pneumonia | 149 | 94.6% |
| Normal | 158 | 92.4% |

What this shows:

- ResNet50 gives **far fewer false alarms** on healthy children: 12 vs 29. It misses 8 more pneumonia cases (13 vs 5).
- The accuracy difference between the two models is **not statistically significant** (McNemar p = 0.18). ResNet50 is better balanced, but not clearly more accurate.
- **Viral pneumonia is harder** than bacterial, because its signs are subtler.

Figures: `results/confusion_matrix_resnet50.png`, `results/roc_curves.png`, `results/training_curves_resnet50.png`.

## 11. Results: 4 classes

Test set, 793 images:

| Model | Accuracy (95% CI) | Macro precision | Macro recall | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN | 72.6% (69.4–75.6) | 0.813 | 0.712 | 0.734 | 0.934 |
| **ResNet50** | **82.3%** (79.5–84.8) | **0.847** | **0.845** | **0.844** | **0.962** |

Here ResNet50 is **clearly better** (McNemar p = 7e-8). The baseline calls too many images COVID-19.

**ResNet50 per class:**

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Normal | 0.761 | 0.864 | 0.809 | 243 |
| Pneumonia / Non-COVID infection | 0.801 | 0.788 | 0.794 | 240 |
| COVID-19 | 0.895 | 0.783 | 0.836 | 240 |
| Tuberculosis | 0.930 | 0.943 | 0.936 | 70 |

- Most mistakes are between Pneumonia and Normal, and COVID-19 being called Pneumonia or Normal. These are hard cases for doctors too.
- **Same class, different dataset, different accuracy:** pneumonia accuracy is 87.5% on Kermany images but only 70.0% on COVID-QU-Ex images. The model depends on where an image comes from.
- **TB looks almost perfect (ROC-AUC 0.999), but should not be trusted.** All TB images come from one dataset, so the model may recognise the dataset's look rather than the disease. Section 16 confirms this.

The app uses this 4-class ResNet50.

## 12. Test on a different dataset

Only Normal vs Pneumonia exists in two independent datasets, so this is the task that can be tested on a "new hospital":

| Trained on | Tested on | Seen in training? | Images | Accuracy | Recall | Specificity |
|---|---|---|---|---|---|---|
| Kermany (children) | Kermany | yes | 583 | 95.7% | 0.969 | 0.924 |
| Kermany (children) | COVID-QU-Ex (adults) | **no** | 2,400 | **73.9%** | 0.897 | **0.581** |
| COVID-QU-Ex (adults) | COVID-QU-Ex | yes | 240 | 87.5% | 0.825 | 0.925 |
| COVID-QU-Ex (adults) | Kermany (children) | **no** | 583 | **74.3%** | 0.918 | **0.272** |

- Accuracy drops by **21.8 points** and **13.2 points** on the unseen dataset.
- Both models still find most pneumonia cases (recall about 0.9). But they call many **healthy** X-rays from the other population "sick": specificity falls to 0.58 and 0.27.
- **Meaning:** what each model learned as "normal" is tied to its own population (children vs adults, different machines). A high score on one dataset does not guarantee a high score in another hospital.

## 13. Grad-CAM: explaining the decisions

**What it is.** Grad-CAM makes a heatmap that shows which regions of the X-ray pushed the model towards its answer. Red/yellow means "used a lot"; blue means "used little".

**How it works, in 4 steps:**

1. Take the last convolution layer of the network (for ResNet50: `conv5_block3_out`, a 7x7 grid).
2. Measure how much each feature map in that layer changes the predicted class's score (the gradients).
3. Add up the feature maps, weighted by those gradients, and keep only the positive part.
4. Stretch the 7x7 map to the image size and lay it over the X-ray.

The code finds this layer automatically, so it works for both models.

**What the 10 report images show** (`results/gradcam/`; 6 correct and the 4 most confident mistakes):

| Case | Truth → model's answer | Where the heat is |
|---|---|---|
| 1–3 | Normal → Normal | Centre of the chest, heart, diaphragm |
| 5 | Viral pneumonia → Pneumonia | **Both lung fields**: what a doctor would expect |
| 4, 6 | Bacterial pneumonia → Pneumonia | Image edges and soft tissue, partly outside the lungs |
| 7–8 | Normal → **Pneumonia** (wrong) | Chest wall, image border, near burned-in text |
| 9–10 | Viral pneumonia → **Normal** (wrong) | Windpipe and centre of the chest |

**Lesson.** Sometimes the model looks at the lungs, but sometimes at borders, text and markers. Grad-CAM revealed that the model partly learned **the dataset, not only the disease**. This finding led to the extra checks in section 16.

In the 4-class model (`results/multiclass/gradcam/`):

- a COVID-19 case lights up both lungs, and a TB case lights up the upper left lung, a typical TB location;
- but one mistake (a COVID-19 image called Pneumonia) has its heat on a burned-in side marker, the letter "D".

## 14. The web app

The app is called **PulmoScan**. It has two parts:

- a **FastAPI** backend (`backend/main.py`, Python);
- a frontend (`frontend/`, plain HTML, CSS and JavaScript).

**Page 1, New analysis (`index.html`):**

- Enter the patient's name and age (both checked: name 2–60 characters, age 0–120).
- Upload an X-ray (drag and drop, JPEG/PNG), or click a sample button to load a real test image of that class.
- Click **Analyze**.

**Page 2, Analysis report (`results.html`):**

- patient details and a report ID;
- the detected condition and the confidence;
- probability bars for all 4 classes;
- the Grad-CAM viewer with 4 views:
  - Original
  - Model input (what the network sees)
  - Overlay, with an opacity slider
  - Compare (side by side)
- five lines of general information about the condition;
- a disclaimer;
- a **Print / Save as PDF** button.

**Backend routes:**

| Route | What it does |
|---|---|
| `GET /health` | Says whether the model is loaded, and its classes and test metrics |
| `POST /predict` | Takes an image; returns the class, confidence, all probabilities and the Grad-CAM images |
| `GET /samples/{class}` | Returns a random test X-ray of that class, for the sample buttons |
| `/` | Serves the web pages |

**Safety and privacy features:**

- **Not-a-chest-X-ray check.** A classifier alone would name a disease for *any* picture, even a face photo. So the app first compares the upload with 400 real training X-rays, using the model's own internal features. If it is not similar enough, the app says "Not a chest X-ray" and shows no disease.
  - The threshold lets about 99% of real X-rays through.
  - It rejected all 21 non-chest photos that were tried.
  - It wrongly rejected 3 of 300 test X-rays (1%).
- **TB warning.** A TB result shows a warning that TB predictions are not reliable on new X-rays (see section 16).
- **Privacy.** Only the image is sent to the server. The name and age stay in the browser tab and are only printed on the report.
- **Input checks.**
  - Files over 20 MB are refused (error 413).
  - Files that are not images are refused (error 400).
  - Requests while no model is loaded get error 503.
- **Disclaimer** on every page: "Educational prototype, not a diagnostic tool".

Screenshots: `results/app/`.

## 15. Testing the app

`tests/test_integration.py` starts the real server and checks:

1. `/health` reports the right model, classes and Grad-CAM layer.
2. For real test X-rays (one per class, plus one the model gets wrong), `/predict` gives the **same class and probabilities as the offline evaluation** (difference under 0.002). This proves the app preprocesses images exactly like training.
3. The probabilities add up to 1, and all returned images are valid.
4. A non-image upload gets error 400, and an unknown sample class gets error 404.
5. The web page loads and shows the disclaimer.

**Result: all checks pass.** Two bugs found during testing were fixed:

- the browser showed an old, cached version of the page;
- the "Model input" view was drawn at the wrong size.

## 16. Extra checks: did the model learn shortcuts?

A **shortcut** is a pattern that predicts the label in this dataset but has nothing to do with the disease, such as image size or a text marker. These experiments test how much the models rely on them. None of them changed the app's model.

### 16.1 The image-size shortcut

In Kermany, pneumonia X-rays are naturally **smaller and wider** (median 1160x768 pixels) than normal ones (1654x1322).

A simple model that sees **only the width, height and shape**, and no pixels at all, already predicts the class with **ROC-AUC 0.912**. So a CNN could partly "cheat" by noticing size effects.

### 16.2 Experiment A: centre crop

**Change:** cut out the central square instead of squashing the whole image to 224x224 ([results/ablation_crop/REPORT.md](results/ablation_crop/REPORT.md)).

- ResNet50: 95.7% → **93.7%** (ROC-AUC 0.993 → 0.989).
- The shortcut did not disappear; it moved. Where the lungs sit in the cropped image still predicts the class (AUC 0.927).
- On the unseen dataset, ResNet50 stayed at about 74%. The baseline CNN improved from 55.5% to 75.3%.

### 16.3 Experiment B: crop to the lungs

**Change:** a lung-segmentation network (**U-Net**) finds the lungs, and only the lung box (plus a small margin) is kept ([results/ablation_lungbox/REPORT.md](results/ablation_lungbox/REPORT.md)).

- The U-Net was trained on COVID-QU-Ex lung masks. It never saw any of the test images. Its lung overlap score (Dice) was 0.969 on the COVID-QU-Ex test images and 0.956 on the Kermany test images.

| | Squash (default) | Centre crop | Lung box |
|---|---|---|---|
| ResNet50 test accuracy | 95.7% | 93.7% | **90.9%** |
| ResNet50 ROC-AUC | 0.993 | 0.989 | 0.965 |
| Lung position predicts class (AUC, 0.5 = no) | 0.937 | 0.927 | **0.666** |
| Share of Grad-CAM heat inside the lungs (share of the image the lungs cover) | 22.5% (22.9%) | 26.1% (29.0%) | **30.9% (23.0%)** |
| Accuracy on the unseen COVID-QU-Ex images | 73.9% | 74.0% | 67.5% |

What this shows:

- The **lung-position cue is mostly gone** (0.937 → 0.666).
- The model now looks **more at the lungs**: more heat than area, for the first time.
- But accuracy dropped about 5 points. So **part of the original 95.7% came from things outside the lungs**: borders, framing, text markers.
- One cue remains: the native resolution of the lung box still predicts the class (AUC 0.884).

### 16.4 External test on TBX11K

TBX11K is a TB dataset that no model was trained on ([results/external_tbx11k/REPORT.md](results/external_tbx11k/REPORT.md)).

**First, a copy check.** TBX11K was compared with all 43,976 training-source images, and 46 copies were removed. 8,353 images were used.

| | On its own test set | On TBX11K (new source) |
|---|---|---|
| TB images found by the 4-class model | 94.3% (AUC 0.999) | **5.8%** of 799 (AUC 0.838) |
| Healthy X-rays called Normal | | 92.9% |

- **The TB class does not transfer.** It mostly learned "this image comes from the TB database", not TB itself. This is why the app now shows a TB warning.
- The model does notice that something is abnormal: any-finding vs healthy has AUC about 0.85. But it often names the wrong disease.
- The not-a-chest-X-ray check passed 99.5–100% of these real X-rays.

## 17. Important decisions and why

| Decision | Reason |
|---|---|
| Split **by patient** after merging the original folders | Several X-rays per child; an image-level split leaked 74% of pneumonia test patients into training |
| Remove duplicates and copies **before** splitting | A copy in train and test lets the model "remember" the answer |
| Choose the model on **validation**, report on **test** only | The test score stays an honest estimate |
| **CLAHE after resizing** | Same effect for every dataset, whatever its original resolution |
| Normalisation and augmentation **inside the model** | The app cannot accidentally preprocess differently, and test images are never augmented |
| **Class weights** instead of copying images | Keeps every image, and the task asked for class weights |
| Kept **ResNet50** on CPU | Measured training time (~1 hour) was acceptable; ResNet50 was the stated objective |
| BatchNorm layers **frozen** while fine-tuning | Updating them with small batches often breaks a pre-trained network |
| Equal images **per dataset** in the shared classes | Stops "which dataset" from predicting "which class" |
| Seed **42** everywhere | Results can be reproduced exactly |
| Ablations write to **separate folders** | Experiments can never overwrite the app's model or the main results |

## 18. Limitations

1. **Shortcuts exist.** Image size, framing, borders and text markers carry label information, and the models use some of it. The reported accuracies are upper bounds, not clinical performance.
2. **Weak transfer to new hospitals.** Accuracy drops 13–22 points on an unseen dataset, and the TB class fails on an external TB dataset.
3. **TB comes from one source only**, so the TB class learned the source.
4. **No patient IDs in COVID-QU-Ex and the TB database.** Two different X-rays of the same adult could still be in different splits. Only near-identical images are grouped.
5. **Labels come from the datasets**, not from a new check by a radiologist.
6. **CPU only.** Short training (5 + 3 epochs) and no hyperparameter search.
7. **Grad-CAM is coarse** (a 7x7 grid for ResNet50). It shows influence, not proof of where the disease is.

## 19. Future work

- Resize every lung region to a common resolution before cropping, to remove the last size shortcut.
- Test on an external dataset of children's X-rays.
- Get TB images from several hospitals, so the TB class learns the disease.
- Train on several datasets and always test on a dataset left out of training.
- Calibrate the probabilities, and accept DICOM files (the hospital X-ray format) in the app.

## 20. How to run it

From the project folder, in Git Bash on Windows. The Python environment is `.venv` (Python 3.12).

**Start the app** (for the demo):

```bash
.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000**.

**Run the integration test:**

```bash
.venv/Scripts/python tests/test_integration.py
```

**Rebuild everything from the start** (about 1 hour of training per task on CPU):

```bash
.venv/Scripts/python -m src.data.prepare
```

```bash
.venv/Scripts/python -m src.train --task binary
```

```bash
.venv/Scripts/python -m src.gradcam
```

The 4-class steps (they need Kaggle data) are listed in [README.md](README.md) under "How to run".

**Suggested demo for the viva:**

1. Start the app and open it in the browser.
2. Enter a name and age, click a **Normal** sample, then **Analyze**. Show the report, the probability bars and the Grad-CAM views.
3. Go back and try a **Pneumonia** and a **COVID-19** sample. Point out where the heat falls.
4. Upload a normal photo, for example a face or a landscape. The app answers **"Not a chest X-ray"**.
5. Try a **TB** sample. Show the TB warning and explain why (section 16.4).
6. Click **Print / Save as PDF** to show the printable report.

## 21. Glossary

| Word | Meaning |
|---|---|
| **CNN** (convolutional neural network) | A neural network that learns image patterns with small sliding filters |
| **Transfer learning** | Starting from a network trained on another task (here, ImageNet photos) and adapting it |
| **Fine-tuning** | Training some pre-trained layers further, slowly, on the new data |
| **Epoch** | One pass through all the training images |
| **Augmentation** | Small random changes to training images, so the model generalises |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalisation: improves local contrast |
| **Class weights** | Make mistakes on the smaller class count more during training |
| **Validation set** | Data used to choose the model; not for the final score |
| **Test set** | Data used only once, for the final score |
| **Data leakage** | Test information reaching training (e.g. the same patient in both), which inflates scores |
| **Shortcut learning** | The model uses a pattern that predicts the label but is not the disease |
| **Ablation** | An experiment that removes or changes one part to see its effect |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping: the explanation heatmap |
| **U-Net** | A network that marks every pixel, here "lung" or "not lung" |
| **Dice score** | Overlap between predicted and true lung masks (1 = perfect) |
| **ROC-AUC** | Ranking quality from 0.5 (guessing) to 1.0 (perfect) |
| **McNemar test** | Tests whether two models really differ on the same images |
| **FastAPI** | A Python library for building web APIs |
| **SHA-256** | A file fingerprint; two identical files have the same one |

## 22. Questions the examiner may ask

**Q: Why not just use the dataset's own train/test folders?**
They are not separated by patient: 170 pneumonia patients appear in both. The validation folder has only 16 images. So everything was merged and split again by patient.

**Q: Your accuracy is 95.7%. Can a hospital use this?**
No.
- On X-rays from another dataset, accuracy falls to about 74%.
- The extra checks show the model partly uses shortcuts.
- It is an educational prototype, not a certified medical device.

**Q: Why ResNet50 and not a newer model?**
It is the model the objectives asked for, it is well understood, and it trained in about an hour on a CPU. Transfer learning works well when there are only a few thousand images.

**Q: Why class weights and not oversampling?**
Class weights keep every image exactly once and do not create repeated copies of normal images. The task also asked for them.

**Q: How do you know the app gives the same answer as your evaluation?**
The integration test sends real test images through the running app and compares the result with the offline evaluation. The difference is under 0.002.

**Q: What does Grad-CAM tell you, and what can it not tell you?**
It shows which regions influenced the answer. It is coarse (7x7) and shows influence, not proof of disease. Here it was useful because it **exposed** heat on borders and markers.

**Q: Why should we not trust the TB score of 0.999?**
All TB images came from one dataset. On a new TB dataset (TBX11K), the model found only 5.8% of TB cases.

**Q: What was the most important thing you learned?**
Careful evaluation matters as much as the model. These steps changed the conclusions:
- the patient-level split;
- the hidden copies across datasets;
- the different-dataset test;
- the shortcut checks.

A high test score alone would have been misleading.

**Q: What would you do with more time?**
See section 19. Most important: test on external data and remove the remaining resolution shortcut.

## 23. Where to find everything

**Folders:**

| Folder | Contents |
|---|---|
| `src/data/` | Download, cleaning, splitting, preprocessing, multi-dataset merge |
| `src/models.py`, `src/train.py`, `src/evaluate.py` | Models, training, evaluation |
| `src/gradcam.py` | Grad-CAM |
| `src/gate.py` | The not-a-chest-X-ray check |
| `src/lung_segmenter.py` | U-Net lung segmenter (extra checks) |
| `src/ablation_*.py`, `src/external_test.py` | Extra checks (section 16) |
| `backend/` | FastAPI server |
| `frontend/` | Web pages |
| `tests/` | Integration test |
| `splits/` | Which image is in train / validation / test |
| `results/` | All figures, tables and reports |
| `notebooks/` | Google Colab notebook for GPU training |

**Figures for slides:**

| Topic | File |
|---|---|
| Class balance | `results/eda/class_counts.png` |
| Split by patient | `results/eda/split_counts.png` |
| Example X-rays | `results/eda/samples_per_class.png` |
| Image sizes (the size shortcut) | `results/eda/image_sizes.png` |
| Preprocessing before / after | `results/preprocessing_before_after.png` |
| Augmentation | `results/augmentation_examples.png` |
| Training curves | `results/training_curves_resnet50.png` |
| Confusion matrix (binary) | `results/confusion_matrix_resnet50.png` |
| ROC curves (binary) | `results/roc_curves.png` |
| Grad-CAM, binary | `results/gradcam/gradcam_grid.png` |
| 4-class confusion matrix | `results/multiclass/confusion_matrix_resnet50.png` |
| 4-class Grad-CAM | `results/multiclass/gradcam/gradcam_grid.png` |
| Hidden copies across datasets | `results/multiclass/near_duplicate_examples.png` |
| Different-dataset test | `results/cross_dataset/confusion_matrix_kermany_on_covidquex.png` |
| Lung-box Grad-CAM comparison | `results/ablation_lungbox/comparison/gradcam_examples.png` |
| App screenshots | `results/app/` |

**Environment:**

- Python 3.12, TensorFlow 2.21 / Keras 3.15 (CPU only)
- Windows 11, Intel i5-13420H, 16 GB RAM
- Seed 42
