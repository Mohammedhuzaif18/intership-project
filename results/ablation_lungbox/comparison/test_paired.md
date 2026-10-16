# Paired comparison on the same test images (lungbox minus the other pipeline; CIs from 2,000 patient-level bootstrap resamples)

| Model        | Comparison       |   Accuracy change | Accuracy change 95% CI   |   Only the other pipeline right |   Only lungbox right |   McNemar exact p |   ROC-AUC change | AUC change 95% CI   |   Prediction agreement |
|:-------------|:-----------------|------------------:|:-------------------------|--------------------------------:|---------------------:|------------------:|-----------------:|:--------------------|-----------------------:|
| ResNet50     | lungbox - squash |           -0.0480 | -0.072 to -0.027         |                              38 |                   10 |            0.0001 |          -0.0279 | -0.042 to -0.017    |                 0.9177 |
| ResNet50     | lungbox - crop   |           -0.0274 | -0.052 to -0.004         |                              34 |                   18 |            0.0365 |          -0.0231 | -0.036 to -0.013    |                 0.9108 |
| Baseline CNN | lungbox - squash |           -0.0274 | -0.058 to +0.002         |                              43 |                   27 |            0.0722 |          -0.0230 | -0.037 to -0.010    |                 0.8799 |
| Baseline CNN | lungbox - crop   |            0.0172 | -0.005 to +0.038         |                              16 |                   26 |            0.1641 |          -0.0266 | -0.039 to -0.015    |                 0.9280 |
