# Per-class test metrics (4-class, one-vs-rest ROC-AUC)

| Model        | Class                           |   precision |   recall |     f1 |   support |   roc_auc |
|:-------------|:--------------------------------|------------:|---------:|-------:|----------:|----------:|
| Baseline CNN | Normal                          |      0.8631 |   0.5967 | 0.7056 |       243 |    0.907  |
| Baseline CNN | Pneumonia / Non-COVID infection |      0.8168 |   0.65   | 0.7239 |       240 |    0.898  |
| Baseline CNN | COVID-19                        |      0.5928 |   0.9583 | 0.7325 |       240 |    0.9331 |
| Baseline CNN | Tuberculosis                    |      0.9783 |   0.6429 | 0.7759 |        70 |    0.9966 |
| ResNet50     | Normal                          |      0.7609 |   0.8642 | 0.8092 |       243 |    0.9449 |
| ResNet50     | Pneumonia / Non-COVID infection |      0.8008 |   0.7875 | 0.7941 |       240 |    0.9349 |
| ResNet50     | COVID-19                        |      0.8952 |   0.7833 | 0.8356 |       240 |    0.9677 |
| ResNet50     | Tuberculosis                    |      0.9296 |   0.9429 | 0.9362 |        70 |    0.9986 |
