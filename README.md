# Denoisers

Personal implementation of denoisers by PyTorch.

Pretrained model and dataset: [[OneDrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/3140103306_zju_edu_cn/Ep1DF1eJ6gpEq43hxffWY7oBFUdtWFKuiDUYwd6QVtD9jA?e=9DYujR)]

## Model

- DnCNN: [[GitHub](https://github.com/SaoYan/DnCNN-PyTorch)][[Paper](https://ieeexplore.ieee.org/document/7839189/)]
- UNet: [[GitHub](https://github.com/milesial/Pytorch-UNet)][[Paper](https://arxiv.org/pdf/1505.04597.pdf)]
- RDN: [[GitHub](https://github.com/yulunzhang/RDN)][[Paper](https://arxiv.org/abs/1812.10477)]
- N3Net: [[GitHub](https://github.com/visinf/n3net)][[Paper](https://arxiv.org/abs/1810.12575)]
- CBDNet: [[Paper](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-CBDNet.pdf)]
- DuRB: [[Paper](https://arxiv.org/abs/1903.08817)]
- MemNet: [[GitHub](https://github.com/Vandermode/pytorch-MemNet)][[Paper](http://openaccess.thecvf.com/content_iccv_2017/html/Tai_MemNet_A_Persistent_ICCV_2017_paper.html)]

## Result

### Baseline

| Method   | PSNR  | SSIM  | FLOPs / G | Params / M | Latency / ms |  FPS  |
| -------- | :---: | :---: | :-------: | :--------: | :----------: | :---: |
| DnCNN    | 25.96 | 0.766 |   147.1   |    0.6     |    57.04     | 19.0  |
| UNet     | 38.39 | 0.916 |   60.0    |    7.8     |    22.73     | 46.6  |
| RDN-1    | 39.41 | 0.914 |   334.1   |    1.1     |      \       |   \   |
| N3Net-1  | 39.24 | 0.912 |     \     |     \      |    46.30     | 24.2  |
| CBDNet   | 39.26 | 0.913 |   172.0   |    4.3     |    54.73     | 19.7  |
| DuRB-P   |       |       |   220.7   |    0.8     |              |       |
| MemNet-3 |       |       |   193.8   |    0.7     |              |       |

### Other

| Method      | PSNR  | SSIM  | FLOPs / G | Params / M | Latency / ms |  FPS  |
| ----------- | :---: | :---: | :-------: | :--------: | :----------: | :---: |
| SSnbt-UNet  | 37.53 | 0.902 |   35.8    |    9.7     |    28.77     | 37.4  |
| SE-UNet     | 38.27 | 0.915 |   60.0    |    7.8     |    63.03     | 16.6  |
| GC-UNet     | 38.13 | 0.914 |   60.1    |    8.6     |    34.01     | 30.8  |
| Mobile-UNet | 30.44 | 0.630 |    3.5    |    0.4     |    20.75     | 54.9  |