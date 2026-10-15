# Any finding vs healthy on TBX11K (external; 95% CIs)

Flagged = predicted class is not Normal, whatever the disease name.

| Model                     | Healthy flagged (any disease)   | Sick-not-TB flagged   | TB flagged (any disease)   | AUC, 1 - P(Normal): sick or TB vs healthy   |
|:--------------------------|:--------------------------------|:----------------------|:---------------------------|:--------------------------------------------|
| 4-class ResNet50 (app)    | 7.1% (6.4%-8.0%)                | 63.0% (61.5%-64.5%)   | 40.2% (36.8%-43.6%)        | 0.856 (0.849-0.864)                         |
| 4-class baseline CNN      | 44.6% (43.0%-46.1%)             | 92.5% (91.6%-93.3%)   | 64.3% (60.9%-67.6%)        | 0.851 (0.843-0.859)                         |
| Binary ResNet50 (Kermany) | 12.3% (11.3%-13.4%)             | 74.1% (72.7%-75.5%)   | 45.6% (42.1%-49.0%)        | 0.848 (0.840-0.856)                         |
