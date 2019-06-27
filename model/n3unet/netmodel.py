'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import math

import torch.nn as nn

from . import non_local
from .unet import UNet

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

def cnn_from_def(cnn_opt):
    kernel = cnn_opt.get("kernel",3)
    padding = (kernel-1)//2
    cnn_bn = cnn_opt.get("bn",True)
    cnn_depth = cnn_opt.get("depth",0)
    cnn_channels = cnn_opt.get("features")
    cnn_outchannels = cnn_opt.get("nplanes_out",)
    chan_in = cnn_opt.get("nplanes_in")

    if cnn_depth == 0:
        cnn_outchannels=chan_in

    cnn_layers = []
    relu = nn.ReLU(inplace=True)

    for i in range(cnn_depth-1):
        cnn_layers.extend([
            nn.Conv2d(chan_in,cnn_channels,kernel, 1, padding, bias=not cnn_bn),
            nn.BatchNorm2d(cnn_channels) if cnn_bn else nn.Sequential(),
            relu
        ])
        chan_in = cnn_channels

    if cnn_depth > 0:
        cnn_layers.append(
            nn.Conv2d(chan_in,cnn_outchannels,kernel, 1, padding, bias=True)
        )

    net = nn.Sequential(*cnn_layers)
    net.nplanes_out = cnn_outchannels
    net.nplanes_in = cnn_opt.get("nplanes_in")
    return net


class N3Block(nn.Module):
    r"""
    N3Block operating on a 2D images
    """
    def __init__(self, nplanes_in, k, patchsize=10, stride=5,
                nl_match_window=15,
                temp_opt={}, embedcnn_opt={}):
        r"""
        :param nplanes_in: number of input features
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param nl_match_window: size of matching window around each patch,
            i.e. the nl_match_window x nl_match_window patches around a query patch
            are used for matching
        :param temp_opt: options for handling the the temperature parameter
        :param embedcnn_opt: options for the embedding cnn, also shared by temperature cnn
        """
        super(N3Block, self).__init__()
        self.patchsize = patchsize
        self.stride = stride

        # patch embedding
        embedcnn_opt["nplanes_in"] = nplanes_in
        self.embedcnn = cnn_from_def(embedcnn_opt)

        # temperature cnn
        with_temp = temp_opt.get("external_temp")
        if with_temp:
            tempcnn_opt = dict(**embedcnn_opt)
            tempcnn_opt["nplanes_out"] = 1
            self.tempcnn = cnn_from_def(tempcnn_opt)
        else:
            self.tempcnn = None

        self.nplanes_in = nplanes_in
        self.nplanes_out = (k+1) * nplanes_in

        indexer = lambda xe_patch,ye_patch: non_local.index_neighbours(xe_patch, ye_patch, nl_match_window, exclude_self=True)
        self.n3aggregation = non_local.N3Aggregation2D(indexing=indexer, k=k,
                patchsize=patchsize, stride=stride, temp_opt=temp_opt)
        self.k = k

        self.reset_parameters()

    def forward(self, x):
        if self.k <= 0:
            return x

        xe = self.embedcnn(x)
        ye = xe

        xg = x
        if self.tempcnn is not None:
            log_temp = self.tempcnn(x)
        else:
            log_temp = None

        x = self.n3aggregation(xg,xe,ye,log_temp=log_temp)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d)):
                dncnn_batchnorm_init(m, kernelsize=3, b_min=0.025)

class N3UNet(nn.Module):
    def __init__(self):
        super(N3UNet, self).__init__()
        nl_opt = {'k':4}

        cnns = []
        nls = []

        cnns.append(UNet(3, 3))
        nl = N3Block(3, **nl_opt)
        nls.append(nl)

        cnns.append(UNet(15, 3))

        self.nls = nn.Sequential(*nls)
        self.blocks = nn.Sequential(*cnns)

    def forward(self, x):
        x = self.blocks[0](x)
        x = self.nls(x)
        x = self.blocks[1](x)
        return x
