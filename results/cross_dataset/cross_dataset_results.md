# Cross-dataset generalization (test images only)

| Model    | Trained on   | Tested on   | Source seen in training   |    n |   Accuracy |   Precision |   Recall |   Specificity |     F1 |   ROC-AUC |
|:---------|:-------------|:------------|:--------------------------|-----:|-----------:|------------:|---------:|--------------:|-------:|----------:|
| ResNet50 | Kermany      | Kermany     | yes                       |  583 |     0.9571 |      0.9717 |   0.9694 |        0.9241 | 0.9706 |    0.9932 |
| ResNet50 | Kermany      | COVID-QU-Ex | no                        | 2400 |     0.7388 |      0.6814 |   0.8967 |        0.5808 | 0.7744 |    0.8726 |
| ResNet50 | COVID-QU-Ex  | COVID-QU-Ex | yes                       |  240 |     0.8750 |      0.9167 |   0.8250 |        0.9250 | 0.8684 |    0.9329 |
| ResNet50 | COVID-QU-Ex  | Kermany     | no                        |  583 |     0.7427 |      0.7723 |   0.9176 |        0.2722 | 0.8387 |    0.6601 |

## Accuracy drop on the unseen source

| Trained on   |   Seen-source accuracy |   Unseen-source accuracy |   Accuracy drop (points) |   Seen-source F1 |   Unseen-source F1 |
|:-------------|-----------------------:|-------------------------:|-------------------------:|-----------------:|-------------------:|
| Kermany      |                 0.9571 |                   0.7388 |                  21.8368 |           0.9706 |             0.7744 |
| COVID-QU-Ex  |                 0.8750 |                   0.7427 |                  13.2290 |           0.8684 |             0.8387 |
