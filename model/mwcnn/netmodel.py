import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import common

class Network(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(Network, self).__init__()
        in_channel = 3
        n_feats = 64
        self.scale_idx = 0

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        n = 3
        m_head = [common.BBlock(conv, in_channel * 4, 160, 3)]
        d_l1 = []
        for _ in range(n):
            d_l1.append(common.BBlock(conv, 160, 160, 3))

        d_l2 = [common.BBlock(conv, 640, n_feats * 4, 3)]
        for _ in range(n):
            d_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 4, 3))

        pro_l3 = [common.BBlock(conv, n_feats * 16, n_feats * 4, 3)]
        for _ in range(n*2):
            pro_l3.append(common.BBlock(conv, n_feats * 4, n_feats * 4, 3))
        pro_l3.append(common.BBlock(conv, n_feats * 4, n_feats * 16, 3))

        i_l2 = []
        for _ in range(n):
            i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 4, 3))
        i_l2.append(common.BBlock(conv, n_feats * 4,640, 3))

        i_l1 = []
        for _ in range(n):
            i_l1.append((common.BBlock(conv, 160, 160, 3)))

        m_tail = [conv(160, in_channel * 4, 3)]

        self.head = nn.Sequential(*m_head)

        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)

        self.pro_l3 = nn.Sequential(*pro_l3)

        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)

        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x1 = self.d_l1(self.head(self.DWT(x)))
        x2 = self.d_l2(self.DWT(x1))
        
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2

        x_ = self.IWT(self.i_l2(x_)) + x1
        x = self.IWT(self.tail(self.i_l1(x_))) + x

        return x
