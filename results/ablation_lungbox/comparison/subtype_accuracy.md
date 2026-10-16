# Test accuracy per subtype

| model        | subtype   |   n |   correct |   accuracy | preprocessing   |
|:-------------|:----------|----:|----------:|-----------:|:----------------|
| Baseline CNN | bacteria  | 276 |       275 |     0.9964 | squash          |
| Baseline CNN | normal    | 158 |       129 |     0.8165 | squash          |
| Baseline CNN | virus     | 149 |       145 |     0.9732 | squash          |
| ResNet50     | bacteria  | 276 |       271 |     0.9819 | squash          |
| ResNet50     | normal    | 158 |       146 |     0.9241 | squash          |
| ResNet50     | virus     | 149 |       141 |     0.9463 | squash          |
| Baseline CNN | bacteria  | 276 |       250 |     0.9058 | crop            |
| Baseline CNN | normal    | 158 |       157 |     0.9937 | crop            |
| Baseline CNN | virus     | 149 |       116 |     0.7785 | crop            |
| ResNet50     | bacteria  | 276 |       267 |     0.9674 | crop            |
| ResNet50     | normal    | 158 |       145 |     0.9177 | crop            |
| ResNet50     | virus     | 149 |       134 |     0.8993 | crop            |
| Baseline CNN | bacteria  | 276 |       253 |     0.9167 | lungbox         |
| Baseline CNN | normal    | 158 |       152 |     0.9620 | lungbox         |
| Baseline CNN | virus     | 149 |       128 |     0.8591 | lungbox         |
| ResNet50     | bacteria  | 276 |       266 |     0.9638 | lungbox         |
| ResNet50     | normal    | 158 |       135 |     0.8544 | lungbox         |
| ResNet50     | virus     | 149 |       129 |     0.8658 | lungbox         |
