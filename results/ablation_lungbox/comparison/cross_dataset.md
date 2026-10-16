# Unseen COVID-QU-Ex Normal / Non-COVID images (cross-dataset)

| Model        | Preprocessing   | Tested on          |    n |   Accuracy | Accuracy 95% CI   |   Precision |   Recall |   Specificity |     F1 |   ROC-AUC |
|:-------------|:----------------|:-------------------|-----:|-----------:|:------------------|------------:|---------:|--------------:|-------:|----------:|
| ResNet50     | squash          | COVID-QU-Ex unseen | 2400 |     0.7388 | 72.1-75.6%        |      0.6814 |   0.8967 |        0.5808 | 0.7744 |    0.8726 |
| ResNet50     | crop            | COVID-QU-Ex unseen | 2400 |     0.7400 | 72.2-75.7%        |      0.6910 |   0.8683 |        0.6117 | 0.7696 |    0.8499 |
| Baseline CNN | squash          | COVID-QU-Ex unseen | 2400 |     0.5546 | 53.5-57.4%        |      0.5297 |   0.9742 |        0.1350 | 0.6862 |    0.7837 |
| Baseline CNN | crop            | COVID-QU-Ex unseen | 2400 |     0.7529 | 73.5-77.0%        |      0.7336 |   0.7942 |        0.7117 | 0.7627 |    0.8331 |
| ResNet50     | lungbox         | COVID-QU-Ex unseen | 2400 |     0.6754 | 65.6-69.4%        |      0.6141 |   0.9442 |        0.4067 | 0.7442 |    0.8572 |
| Baseline CNN | lungbox         | COVID-QU-Ex unseen | 2400 |     0.7771 | 76.0-79.3%        |      0.7930 |   0.7500 |        0.8042 | 0.7709 |    0.8618 |
