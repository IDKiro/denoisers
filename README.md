# Denoisers

Personal implementation of denoisers by PyTorch.

Release trained model and dataset later.

Dataset: ENOIR (train on Mi3 and S90, test on T3i)

## Result

### Baseline

| Method | PSNR  | SSIM  | FLOPs / G | Params / M | Run-Time / ms |
| ------ | :---: | :---: | :-------: | :--------: | :-----------: |
| DnCNN  | 25.96 | 0.766 |   147.1   |    0.6     |     57.04     |
| UNet   | 38.39 | 0.916 |   60.0    |    7.8     |     22.73     |
| RDN    | 39.41 | 0.914 |   334.1   |    1.1     |    140.40     |
| N3Net  | 39.24 | 0.912 |     \     |     \      |     46.30     |
| CBDNet | 39.26 | 0.913 |   172.0   |    4.3     |     54.73     |

### Self

| Method      | PSNR  | SSIM  | FLOPs / G | Params / M | Run-Time / ms |
| ----------- | :---: | :---: | :-------: | :--------: | :-----------: |
| SSnbt-UNet  | 37.53 | 0.902 |   35.8    |    9.7     |     28.77     |
| SE-UNet     | 38.27 | 0.915 |   60.0    |    7.8     |     63.03     |
| GC-UNet     | 38.13 | 0.914 |   60.1    |    8.6     |     34.01     |
| N3-UNet     |       |       |           |            |               |
| Mobile-UNet |       |       |           |            |               |