# Model comparison, held-out test set (binary Normal vs Pneumonia, n=583)

| Model        |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |   Specificity |   TP |   TN |   FP |   FN |   Val F1 (selection) |   Epochs |
|:-------------|-----------:|------------:|---------:|-------:|----------:|--------------:|-----:|-----:|-----:|-----:|---------------------:|---------:|
| Baseline CNN |     0.9142 |      0.9845 |   0.8965 | 0.9384 |    0.9632 |        0.962  |  381 |  152 |    6 |   44 |               0.9253 |        8 |
| ResNet50     |     0.9091 |      0.945  |   0.9294 | 0.9371 |    0.9654 |        0.8544 |  395 |  135 |   23 |   30 |               0.9408 |        8 |
