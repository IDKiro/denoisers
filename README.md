# Denoisers

Personal implementation of denoisers by PyTorch.

Pretrained model and dataset: [[OneDrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/3140103306_zju_edu_cn/Ep1DF1eJ6gpEq43hxffWY7oBFUdtWFKuiDUYwd6QVtD9jA?e=9DYujR)]

## Results

### Baseline

- DnCNN: [[Paper](https://ieeexplore.ieee.org/document/7839189/)]
- RDN: [[GitHub](https://github.com/yulunzhang/RDN)][[Paper](https://arxiv.org/abs/1812.10477)]
- N3Net: [[GitHub](https://github.com/visinf/n3net)][[Paper](https://arxiv.org/abs/1810.12575)]
- CBDNet: [[Paper](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-CBDNet.pdf)]
- UNet: [[GitHub](https://github.com/milesial/Pytorch-UNet)][[Paper](https://arxiv.org/abs/1505.04597)]
- REDNet: [[Paper](https://arxiv.org/abs/1603.09056)]
- CARN & CARNM: [[GitHub](https://github.com/nmhkahn/CARN-pytorch)][[Paper](https://arxiv.org/abs/1810.05052v1)]

| Method   | PSNR  | SSIM  | FLOPs / G | Params / M |  FPS  |
| -------- | :---: | :---: | :-------: | :--------: | :---: |
| DnCNN    | 39.31 | 0.913 |   146.2   |    0.6     | 24.2  |
| UNet     | 39.31 | 0.915 |   239.2   |    31.0    | 16.6  |
| UNet-S   | 38.39 | 0.916 |   60.0    |    7.8     | 46.6  |
| REDNet   | 39.49 | 0.915 |   184.9   |    0.7     | 14.2  |
| MemNet-6 |       |       |   768.8   |    2.9     |       |
| RDN-1    | 39.56 | 0.916 |   334.1   |    1.1     |  7.3  |
| N3Net-2  |       |       |  *292.4*  |   *1.2*    |       |
| CBDNet   | 39.26 | 0.913 |   172.0   |    4.3     | 19.7  |
| CARN     | 39.24 | 0.913 |   272.0   |    0.8     |  7.6  |
| CARNM    | 39.23 | 0.913 |   149.5   |    0.4     |  6.6  |

### Bad Results

- DuRB: [[Paper](https://arxiv.org/abs/1903.08817)]
- MemNet: [[GitHub](https://github.com/Vandermode/pytorch-MemNet)][[Paper](http://openaccess.thecvf.com/content_iccv_2017/html/Tai_MemNet_A_Persistent_ICCV_2017_paper.html)]

| Method      | PSNR  | SSIM  | FLOPs / G | Params / M |  FPS  |
| ----------- | :---: | :---: | :-------: | :--------: | :---: |
| DuRB-P      | 32.77 | 0.691 |   220.7   |    0.8     |  7.4  |
| DuRB-P-ReLu | 34.27 | 0.749 |   220.9   |    0.8     |  7.0  |
| MemNet-3    | 34.70 | 0.775 |   193.8   |    0.7     |  9.0  |

### Expansion

| Method      | PSNR  | SSIM  | FLOPs / G | Params / M |  FPS  |
| ----------- | :---: | :---: | :-------: | :--------: | :---: |
| SSnbt-UNet  | 37.53 | 0.902 |   35.8    |    9.7     | 37.4  |
| SE-UNet     | 38.27 | 0.915 |   60.0    |    7.8     | 16.6  |
| GC-UNet     | 38.13 | 0.914 |   60.1    |    8.6     | 30.8  |
| Mobile-UNet | 30.44 | 0.630 |    3.5    |    0.4     | 54.9  |
| SN-UNet     | 36.95 | 0.883 |   46.3    |    3.5     | 38.9  |