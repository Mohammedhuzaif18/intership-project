# Grad-CAM heat share (and unseen-set accuracy), lungbox minus the other pipeline on the same images (patient / cluster bootstrap CI; Wilcoxon signed-rank for heat, exact McNemar for accuracy)

| Dataset            | Model        | Comparison       | Region   |    n |   Heat share, squash |   Heat share, lungbox |   Change | 95% CI           |   Wilcoxon p |   Heat share, crop |
|:-------------------|:-------------|:-----------------|:---------|-----:|---------------------:|----------------------:|---------:|:-----------------|-------------:|-------------------:|
| Kermany test       | ResNet50     | lungbox - squash | lung     |  517 |               0.2254 |                0.3091 |   0.0837 | +0.075 to +0.092 |       0.0000 |           nan      |
| Kermany test       | ResNet50     | lungbox - squash | box      |  583 |               0.5015 |                0.6897 |   0.1883 | +0.175 to +0.200 |       0.0000 |           nan      |
| Kermany test       | ResNet50     | lungbox - squash | border   |  583 |               0.3377 |                0.2259 |  -0.1119 | -0.120 to -0.102 |       0.0000 |           nan      |
| Kermany test       | ResNet50     | lungbox - crop   | lung     |  517 |             nan      |                0.3091 |   0.0485 | +0.040 to +0.057 |       0.0000 |             0.2606 |
| Kermany test       | ResNet50     | lungbox - crop   | box      |  583 |             nan      |                0.6897 |   0.0102 | -0.004 to +0.027 |       0.2771 |             0.6795 |
| Kermany test       | ResNet50     | lungbox - crop   | border   |  583 |             nan      |                0.2259 |  -0.0794 | -0.090 to -0.071 |       0.0000 |             0.3053 |
| Kermany test       | Baseline CNN | lungbox - squash | lung     |  517 |               0.2237 |                0.1598 |  -0.0640 | -0.081 to -0.048 |       0.0000 |           nan      |
| Kermany test       | Baseline CNN | lungbox - squash | box      |  583 |               0.5430 |                0.6440 |   0.1010 | +0.089 to +0.113 |       0.0000 |           nan      |
| Kermany test       | Baseline CNN | lungbox - squash | border   |  583 |               0.2830 |                0.2702 |  -0.0127 | -0.024 to -0.002 |       0.0022 |           nan      |
| Kermany test       | Baseline CNN | lungbox - crop   | lung     |  517 |             nan      |                0.1598 |   0.0011 | -0.005 to +0.008 |       0.9668 |             0.1586 |
| Kermany test       | Baseline CNN | lungbox - crop   | box      |  583 |             nan      |                0.6440 |   0.0083 | -0.015 to +0.033 |       0.9477 |             0.6357 |
| Kermany test       | Baseline CNN | lungbox - crop   | border   |  583 |             nan      |                0.2702 |  -0.0361 | -0.068 to -0.007 |       0.0736 |             0.3064 |
| COVID-QU-Ex unseen | ResNet50     | lungbox - squash | lung     | 2398 |               0.2817 |                0.3850 |   0.1033 | +0.097 to +0.110 |       0.0000 |           nan      |
| COVID-QU-Ex unseen | ResNet50     | lungbox - squash | border   | 2398 |               0.2693 |                0.2639 |  -0.0054 | -0.012 to +0.001 |       0.0000 |           nan      |
| COVID-QU-Ex unseen | ResNet50     | lungbox - crop   | lung     | 2398 |             nan      |                0.3850 |   0.1298 | +0.126 to +0.134 |       0.0000 |             0.2552 |
| COVID-QU-Ex unseen | ResNet50     | lungbox - crop   | border   | 2398 |             nan      |                0.2639 |  -0.0121 | -0.018 to -0.006 |       0.0000 |             0.2759 |
| COVID-QU-Ex unseen | ResNet50     | lungbox - squash | accuracy | 2400 |               0.7388 |                0.6754 |  -0.0633 | -0.080 to -0.046 |       0.0000 |           nan      |
| COVID-QU-Ex unseen | ResNet50     | lungbox - crop   | accuracy | 2400 |             nan      |                0.6754 |  -0.0646 | -0.082 to -0.047 |       0.0000 |             0.7400 |
| COVID-QU-Ex unseen | Baseline CNN | lungbox - squash | lung     | 2400 |               0.1367 |                0.3061 |   0.1693 | +0.163 to +0.175 |       0.0000 |           nan      |
| COVID-QU-Ex unseen | Baseline CNN | lungbox - squash | border   | 2400 |               0.2660 |                0.3407 |   0.0747 | +0.067 to +0.082 |       0.0000 |           nan      |
| COVID-QU-Ex unseen | Baseline CNN | lungbox - crop   | lung     | 2400 |             nan      |                0.3061 |   0.1539 | +0.151 to +0.157 |       0.0000 |             0.1522 |
| COVID-QU-Ex unseen | Baseline CNN | lungbox - crop   | border   | 2400 |             nan      |                0.3407 |  -0.0164 | -0.024 to -0.008 |       0.0000 |             0.3571 |
| COVID-QU-Ex unseen | Baseline CNN | lungbox - squash | accuracy | 2400 |               0.5546 |                0.7771 |   0.2225 | +0.197 to +0.247 |       0.0000 |           nan      |
| COVID-QU-Ex unseen | Baseline CNN | lungbox - crop   | accuracy | 2400 |             nan      |                0.7771 |   0.0242 | +0.010 to +0.039 |       0.0021 |             0.7529 |
