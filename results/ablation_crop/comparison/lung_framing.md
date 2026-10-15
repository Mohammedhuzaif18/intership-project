# Lung framing in the model input (Kermany train + test images with a COVID-QU-Ex lung mask; x0/x1 = left/right, y0/y1 = top/bottom edge of the lungs)

| Feature (fraction of the model input side or area)   |   Median Normal |   Median Pneumonia |   Test ROC-AUC, this feature alone |
|:-----------------------------------------------------|----------------:|-------------------:|-----------------------------------:|
| lung_retained_by_crop                                |          1.0000 |             0.9347 |                             0.8284 |
| squash:lung_area                                     |          0.2161 |             0.2207 |                             0.6115 |
| crop:lung_area                                       |          0.2634 |             0.2924 |                             0.7090 |
| squash:lung_x0                                       |          0.1161 |             0.1161 |                             0.5591 |
| crop:lung_x0                                         |          0.0268 |             0.0000 |                             0.7774 |
| squash:lung_x1                                       |          0.8929 |             0.8705 |                             0.6074 |
| crop:lung_x1                                         |          0.9777 |             1.0000 |                             0.6983 |
| squash:lung_y0                                       |          0.1741 |             0.1071 |                             0.8871 |
| crop:lung_y0                                         |          0.1741 |             0.1071 |                             0.8871 |
| squash:lung_y1                                       |          0.8929 |             0.9375 |                             0.7342 |
| crop:lung_y1                                         |          0.8839 |             0.8884 |                             0.5266 |
