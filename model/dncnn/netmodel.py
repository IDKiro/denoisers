import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def convnxn(in_planes, out_planes, kernelsize, stride=1, bias=False):
    padding = kernelsize//2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding, bias=bias)


def dncnn_batchnorm_init(m, kernelsize=3, b_min=0.025):
    r"""
    Reproduces batchnorm initialization from DnCNN
    https://github.com/cszn/DnCNN/blob/master/TrainingCodes/DnCNN_TrainingCodes_v1.1/DnCNN_init_model_64_25_Res_Bnorm_Adam.m
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.weight.data[(m.weight.data > 0) & (m.weight.data <= b_min)] = b_min
    m.weight.data[(m.weight.data < 0) & (m.weight.data >= -b_min)] = -b_min
    m.weight.data = m.weight.data.abs()
    m.bias.data.zero_()
    m.momentum = 0.001


class Network(nn.Module):
    r"""
    Implements a DnCNN network
    """
    def __init__(self,  nplanes_in=3, nplanes_out=3, features=64, kernel=3, depth=17, residual=True, bn=False):
        r"""
        :param nplanes_in: number of of input feature channels
        :param nplanes_out: number of of output feature channels
        :param features: number of of hidden layer feature channels
        :param kernel: kernel size of convolution layers
        :param depth: number of convolution layers (minimum 2)
        :param residual: whether to add a residual connection from input to output
        :param bn:  whether to add batchnorm layers
        """
        super(Network, self).__init__()

        self.residual = residual
        self.nplanes_out = nplanes_out
        self.nplanes_in = nplanes_in
        self.kernelsize = kernel
        self.nplanes_residual = None

        self.conv1 = convnxn(nplanes_in, features, kernelsize=kernel, bias=True)
        self.bn1 = nn.BatchNorm2d(features) if bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        layers = []
        for i in range(depth-2):
            layers += [convnxn(features, features, kernel),
                       nn.BatchNorm2d(features)  if bn else nn.Sequential(),
                       nn.ReLU(inplace=True)]
        self.layer1 = nn.Sequential(*layers)
        self.conv2 = convnxn(features , nplanes_out, kernelsize=kernel, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / (n)))
            elif isinstance(m, nn.BatchNorm2d):
                dncnn_batchnorm_init(m, kernelsize=self.kernelsize, b_min=0.025)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.conv2(x)

        nplanes_residual = self.nplanes_residual or self.nplanes_in
        if self.residual:
            nshortcut = min(self.nplanes_in, self.nplanes_out, nplanes_residual)
            x[:,:nshortcut,:,:] = x[:,:nshortcut,:,:] + shortcut[:,:nshortcut,:,:]

        return x