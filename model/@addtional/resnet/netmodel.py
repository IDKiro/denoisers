import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += residual
        out = self.relu(out)
        return out


class Network(nn.Module):
    def __init__(self, block=BasicBlock, embed_dim=10, hidden_dim=10):
        super(Network, self).__init__()

        self.groups = 1
        self.inplanes = 64
        self.num_layers = 14
        self.conv1 = conv3x3(3, self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self._make_group(block, self.inplanes)

        self.outc = nn.Conv2d(self.inplanes, 3, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_group(self, block, planes, group_id=1):
        """ Create the whole group"""
        for i in range(self.num_layers):
            meta = self._make_layer_v2(block, planes)

            setattr(self, 'group{}_layer{}'.format(group_id, i), meta)

    def _make_layer_v2(self, block, planes):
        """ create one block and optional a gate module """
        layer = block(self.inplanes, planes)
        
        return layer

    def forward(self, x, pretrain=False):
        x = self.conv1(x)
        x = self.relu(x)

        for g in range(self.groups):
            for i in range(self.num_layers):
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)

        x = self.outc(x)

        return x
        