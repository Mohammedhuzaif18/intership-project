# Paired comparison on the same test images (crop minus squash; CIs from 2,000 patient-level bootstrap resamples)

| Model        |   Accuracy crop - squash | Accuracy diff 95% CI   |   Only squash right |   Only crop right |   McNemar exact p |   ROC-AUC crop - squash | AUC diff 95% CI   |   Prediction agreement |
|:-------------|-------------------------:|:-----------------------|--------------------:|------------------:|------------------:|------------------------:|:------------------|-----------------------:|
| ResNet50     |                  -0.0206 | -0.040 to -0.004       |                  20 |                 8 |            0.0357 |                 -0.0048 | -0.010 to -0.000  |                 0.9520 |
| Baseline CNN |                  -0.0446 | -0.076 to -0.012       |                  54 |                28 |            0.0054 |                  0.0036 | -0.001 to +0.009  |                 0.8593 |
