# Size-only probes (logistic regression; train split -> test split)

| Probe features                                                                            |   n train |   n test |   test_roc_auc |   test_balanced_accuracy |
|:------------------------------------------------------------------------------------------|----------:|---------:|---------------:|-------------------------:|
| Native width + height + aspect ratio (Review-I probe)                                     |      4658 |      583 |         0.9120 |                   0.8391 |
| Aspect ratio only (kept by squash as distorted anatomy, removed by crop)                  |      4658 |      583 |         0.8608 |                   0.7887 |
| Crop side only (native resolution of the centre square: all the size left after cropping) |      4658 |      583 |         0.9117 |                   0.8391 |
| Lung area + lung bounding box in the squash model input (no pixels)                       |      4130 |      517 |         0.9369 |                   0.8611 |
| Lung area + lung bounding box in the crop model input (no pixels)                         |      4130 |      517 |         0.9273 |                   0.8505 |
