# Lung segmenter (U-Net, 256x256) on COVID-QU-Ex lung masks

| Images                                                  |    n |   Dice (mean) |   Dice (median) |   Dice after cleaning |   Dice < 0.80 |   Lung bounding-box IoU |   Empty predictions |
|:--------------------------------------------------------|-----:|--------------:|----------------:|----------------------:|--------------:|------------------------:|--------------------:|
| Test folder, all                                        | 6788 |        0.9695 |          0.9776 |                0.9697 |            20 |                  0.9622 |                   0 |
| Test folder, COVID-19                                   | 2395 |        0.9711 |          0.9789 |                0.9713 |            11 |                  0.9664 |                   0 |
| Test folder, Non-COVID                                  | 2253 |        0.9557 |          0.9632 |                0.9560 |             9 |                  0.9456 |                   0 |
| Test folder, Normal                                     | 2140 |        0.9822 |          0.9852 |                0.9823 |             0 |                  0.9751 |                   0 |
| Held out from training: cross dataset test (any folder) | 2400 |        0.9718 |          0.9812 |                0.9720 |             4 |                  0.9606 |                   0 |
| Held out from training: kermany test copy (any folder)  |  517 |        0.9558 |          0.9635 |                0.9561 |             2 |                  0.9605 |                   0 |
| Held out from training: kermany val copy (any folder)   |  525 |        0.9550 |          0.9653 |                0.9554 |             2 |                  0.9576 |                   0 |
