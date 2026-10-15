# Predicted class by TBX11K image type

| Model                     | TBX11K images   |    n | -> NORMAL   | -> PNEUMONIA   | -> COVID   | -> TB   |
|:--------------------------|:----------------|-----:|:------------|:---------------|:-----------|:--------|
| 4-class ResNet50 (app)    | Healthy         | 3799 | 92.9%       | 0.3%           | 6.7%       | 0.2%    |
| 4-class ResNet50 (app)    | Sick, not TB    | 3755 | 37.0%       | 11.9%          | 49.1%      | 2.0%    |
| 4-class ResNet50 (app)    | Active TB       |  651 | 59.3%       | 4.3%           | 30.1%      | 6.3%    |
| 4-class ResNet50 (app)    | Latent TB       |  148 | 62.2%       | 4.1%           | 30.4%      | 3.4%    |
| 4-class baseline CNN      | Healthy         | 3799 | 55.4%       | 0.4%           | 44.1%      | 0.0%    |
| 4-class baseline CNN      | Sick, not TB    | 3755 | 7.5%        | 0.9%           | 91.5%      | 0.1%    |
| 4-class baseline CNN      | Active TB       |  651 | 37.0%       | 0.9%           | 61.8%      | 0.3%    |
| 4-class baseline CNN      | Latent TB       |  148 | 29.7%       | 1.4%           | 68.2%      | 0.7%    |
| Binary ResNet50 (Kermany) | Healthy         | 3799 | 87.7%       | 12.3%          | 0.0%       | 0.0%    |
| Binary ResNet50 (Kermany) | Sick, not TB    | 3755 | 25.9%       | 74.1%          | 0.0%       | 0.0%    |
| Binary ResNet50 (Kermany) | Active TB       |  651 | 55.3%       | 44.7%          | 0.0%       | 0.0%    |
| Binary ResNet50 (Kermany) | Latent TB       |  148 | 50.7%       | 49.3%          | 0.0%       | 0.0%    |
