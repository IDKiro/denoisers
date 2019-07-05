import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, nplanes, kernel1, kernel2, dilation1, dilation2):
        super(Block, self).__init__()

        padding1 = (kernel1 - 1) * dilation1 // 2
        padding2 = (kernel2 - 1) * dilation2 // 2

        self.doubleconv = nn.Sequential(
            nn.Conv2d(nplanes, nplanes, 3, 1, 1),
            nn.ReLU(inplace=True),                                          # no relu lead to a bad result
            nn.Conv2d(nplanes, nplanes, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.ct1 = nn.Sequential(
            nn.Conv2d(nplanes, nplanes, kernel1, 1, padding1, dilation1),
            nn.ReLU(inplace=True)
        )

        self.ct2 = nn.Sequential(
            nn.Conv2d(nplanes, nplanes, kernel2, 1, padding2, dilation2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, mainin, bypassin):
        convout = mainin + self.doubleconv(mainin)
        bypassout = self.ct1(convout) + bypassin
        mainout = self.ct2(bypassout)

        return mainout, bypassout
        