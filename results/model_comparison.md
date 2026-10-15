# Model comparison, held-out test set (binary Normal vs Pneumonia, n=583)

| Model        |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |   Specificity |   TP |   TN |   FP |   FN |   Val F1 (selection) |   Epochs |
|:-------------|-----------:|------------:|---------:|-------:|----------:|--------------:|-----:|-----:|-----:|-----:|---------------------:|---------:|
| Baseline CNN |     0.9417 |      0.9354 |   0.9882 | 0.9611 |    0.9862 |        0.8165 |  420 |  129 |   29 |    5 |               0.9463 |        8 |
| ResNet50     |     0.9571 |      0.9717 |   0.9694 | 0.9706 |    0.9932 |        0.9241 |  412 |  146 |   12 |   13 |               0.9621 |        7 |
