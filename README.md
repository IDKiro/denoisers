# Denoisers

Personal implementation of denoisers by PyTorch. 

Pretrained model and dataset: [[OneDrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/3140103306_zju_edu_cn/Ep1DF1eJ6gpEq43hxffWY7oBFUdtWFKuiDUYwd6QVtD9jA?e=9DYujR)]

## Results

### Baseline

- DnCNN: [[Code](https://github.com/cszn/DnCNN)][[Paper](https://ieeexplore.ieee.org/document/7839189/)]
- UNet: [[Code](https://github.com/milesial/Pytorch-UNet)][[Paper](https://arxiv.org/abs/1505.04597)]
- REDNet: [[Code](https://github.com/JindongJiang/RedNet)][[Paper](https://arxiv.org/abs/1603.09056)]
- MWCNN: [[Code](https://github.com/lpj0/MWCNN)][[Paper](https://arxiv.org/abs/1805.07071)]
- RDN+: [[Code](https://github.com/yulunzhang/RDN)][[Paper](https://arxiv.org/abs/1812.10477)]
- N3Net: [[Code](https://github.com/visinf/n3net)][[Paper](https://arxiv.org/abs/1810.12575)]
- CBDNet: [[Code](https://github.com/GuoShi28/CBDNet)][[Paper](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-CBDNet.pdf)]

| Method | PSNR  | SSIM  | FLOPs / G | Params / M |  FPS  |
| ------ | :---: | :---: | :-------: | :--------: | :---: |
| DnCNN  | 37.81 | 0.898 |   146.2   |    0.6     | 24.2  |
| UNet   | 37.79 | 0.901 |   239.2   |    31.0    | 16.6  |
| REDNet | 37.95 | 0.900 |   184.9   |    0.7     | 14.2  |
| MWCNN  | 38.17 | 0.901 |   259.0   |    16.2    | 11.1  |
| RDN+   | 38.19 | 0.903 |  1687.8   |    5.5     |  1.4  |
| N3Net  | 38.17 | 0.902 |    N/A    |    0.5     |  N/A  |
| CBDNet | 38.15 | 0.900 |   172.0   |    4.3     | 19.7  |

### Bad Result

- DuRB: [[Paper](https://arxiv.org/abs/1903.08817)]
- MemNet: [[Code](https://github.com/wutianyiRosun/MemNet)][[Paper](http://openaccess.thecvf.com/content_iccv_2017/html/Tai_MemNet_A_Persistent_ICCV_2017_paper.html)]

| Method      | PSNR  | SSIM  | FLOPs / G | Params / M |  FPS  |
| ----------- | :---: | :---: | :-------: | :--------: | :---: |
| DuRB-P      | 31.77 | 0.691 |   220.7   |    0.8     |  7.4  |
| DuRB-P-ReLu | 33.27 | 0.749 |   220.9   |    0.8     |  7.0  |
| MemNet-3    | 33.70 | 0.775 |   193.8   |    0.7     |  9.0  |
| MemNet-6    | 32.56 | 0.718 |   768.8   |    2.9     |  2.2  |

### Expansion

**All the models have been move to the `/@addtional/` folder.**

- RDN: [[Code](https://github.com/yulunzhang/RDN)][[Paper](https://arxiv.org/abs/1802.08797)]
- CARN & CARNM: [[Code](https://github.com/nmhkahn/CARN-pytorch)][[Paper](https://arxiv.org/abs/1810.05052v1)]

| Method        | PSNR  | SSIM  | FLOPs / G | Params / M |  FPS  |
| ------------- | :---: | :---: | :-------: | :--------: | :---: |
| RDN-4         | 38.02 | 0.901 |   334.1   |    1.1     |  7.3  |
| RDN+-4        | 38.02 | 0.901 |   355.5   |    1.2     |  6.9  |
| N3Net-2-17    | 38.05 | 0.900 |   294.5   |    1.1     |  0.1  |
| CARN          | 37.73 | 0.898 |   272.0   |    0.8     |  7.6  |
| CARNM         | 37.71 | 0.898 |   149.5   |    0.4     |  6.6  |
| ResNet        | 37.87 | 0.899 |   271.4   |    1.0     | 11.4  |
| UNet-S        | 36.62 | 0.901 |   60.0    |    7.8     | 46.6  |
| SSnbt-UNet-S  | 35.58 | 0.884 |   35.8    |    9.7     | 37.4  |
| SE-UNet-S     | 36.50 | 0.900 |   60.0    |    7.8     | 16.6  |
| GC-UNet-S     | 36.38 | 0.899 |   60.1    |    8.6     | 30.8  |
| Mobile-UNet-S | 29.33 | 0.588 |    3.5    |    0.4     | 54.9  |
| SN-UNet-S     | 35.22 | 0.863 |   46.3    |    3.5     | 38.9  |