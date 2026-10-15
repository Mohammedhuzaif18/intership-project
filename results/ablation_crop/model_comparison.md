# Model comparison, held-out test set (binary Normal vs Pneumonia, n=583)

| Model        |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |   Specificity |   TP |   TN |   FP |   FN |   Val F1 (selection) |   Epochs |
|:-------------|-----------:|------------:|---------:|-------:|----------:|--------------:|-----:|-----:|-----:|-----:|---------------------:|---------:|
| Baseline CNN |     0.8971 |      0.9973 |   0.8612 | 0.9242 |    0.9898 |        0.9937 |  366 |  157 |    1 |   59 |               0.9192 |        8 |
| ResNet50     |     0.9365 |      0.9686 |   0.9435 | 0.9559 |    0.9885 |        0.9177 |  401 |  145 |   13 |   24 |               0.9536 |        8 |
