# Grad-CAM heat share, crop minus squash on the same images (patient / cluster bootstrap CI, Wilcoxon signed-rank)

| Dataset            | Model        | Region   |    n |   Heat share, squash |   Heat share, crop |   Crop - squash | 95% CI           |   Wilcoxon p |
|:-------------------|:-------------|:---------|-----:|---------------------:|-------------------:|----------------:|:-----------------|-------------:|
| Kermany test       | ResNet50     | lung     |  517 |               0.2254 |             0.2606 |          0.0351 | +0.027 to +0.043 |       0.0000 |
| Kermany test       | ResNet50     | box      |  583 |               0.5015 |             0.6795 |          0.1780 | +0.156 to +0.196 |       0.0000 |
| Kermany test       | ResNet50     | border   |  583 |               0.3377 |             0.3053 |         -0.0324 | -0.045 to -0.018 |       0.0000 |
| Kermany test       | Baseline CNN | lung     |  517 |               0.2237 |             0.1586 |         -0.0651 | -0.084 to -0.047 |       0.0000 |
| Kermany test       | Baseline CNN | box      |  583 |               0.5430 |             0.6357 |          0.0927 | +0.068 to +0.113 |       0.0000 |
| Kermany test       | Baseline CNN | border   |  583 |               0.2830 |             0.3064 |          0.0234 | -0.002 to +0.053 |       0.0004 |
| COVID-QU-Ex unseen | ResNet50     | lung     | 2400 |               0.2817 |             0.2553 |         -0.0264 | -0.032 to -0.021 |       0.0000 |
| COVID-QU-Ex unseen | ResNet50     | border   | 2400 |               0.2693 |             0.2759 |          0.0067 | +0.001 to +0.012 |       0.8319 |
| COVID-QU-Ex unseen | Baseline CNN | lung     | 2400 |               0.1367 |             0.1522 |          0.0155 | +0.009 to +0.021 |       0.0000 |
| COVID-QU-Ex unseen | Baseline CNN | border   | 2400 |               0.2660 |             0.3571 |          0.0911 | +0.083 to +0.100 |       0.0000 |
