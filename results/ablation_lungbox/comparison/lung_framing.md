# Lung framing in the model input (Kermany train + test images with a COVID-QU-Ex lung mask; x0/x1 = left/right, y0/y1 = top/bottom edge of the lungs, as a fraction of the input side)

| Feature           |   Median Normal |   Median Pneumonia |   Test ROC-AUC, this feature alone |
|:------------------|----------------:|-------------------:|-----------------------------------:|
| squash:lung_area  |          0.2161 |             0.2207 |                             0.6115 |
| crop:lung_area    |          0.2634 |             0.2924 |                             0.7090 |
| lungbox:lung_area |          0.2413 |             0.2137 |                             0.5963 |
| squash:lung_x0    |          0.1161 |             0.1161 |                             0.5591 |
| crop:lung_x0      |          0.0268 |             0.0000 |                             0.7774 |
| lungbox:lung_x0   |          0.0446 |             0.0491 |                             0.5965 |
| squash:lung_x1    |          0.8929 |             0.8705 |                             0.6074 |
| crop:lung_x1      |          0.9777 |             1.0000 |                             0.6983 |
| lungbox:lung_x1   |          0.9554 |             0.9554 |                             0.6155 |
| squash:lung_y0    |          0.1741 |             0.1071 |                             0.8871 |
| crop:lung_y0      |          0.1741 |             0.1071 |                             0.8871 |
| lungbox:lung_y0   |          0.1652 |             0.1696 |                             0.5189 |
| squash:lung_y1    |          0.8929 |             0.9375 |                             0.7342 |
| crop:lung_y1      |          0.8839 |             0.8884 |                             0.5266 |
| lungbox:lung_y1   |          0.8438 |             0.8348 |                             0.5393 |
| log_input_side    |          7.2442 |             6.8732 |                             0.8825 |
| box_aspect        |          1.3413 |             1.3549 |                             0.5104 |
| black_share       |          0.2553 |             0.2732 |                             0.5468 |
| letterbox_share   |          0.2545 |             0.2619 |                             0.5105 |
| outside_left      |          0.0000 |             0.0000 |                             0.5277 |
| outside_right     |          0.0000 |             0.0000 |                             0.5184 |
| outside_top       |          0.0000 |             0.0000 |                             0.6482 |
| outside_bottom    |          0.0000 |             0.0000 |                             0.6535 |
| gt_lungs_in_rect  |          1.0000 |             1.0000 |                             0.5100 |
| box_iou_vs_mask   |          0.9711 |             0.9648 |                             0.5709 |
