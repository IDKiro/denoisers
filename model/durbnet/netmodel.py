
import torch
import torch.nn as nn
import torch.nn.functional as F

from .durb import DuRB

class DuRBNet(nn.Module):
    def __init__(self):
        super(DuRBNet, self).__init__()
        self.inc = inconv(3, 32)

        self.durb1 = DuRB(32, 5, 3, 1, 1)
        self.durb2 = DuRB(32, 7, 5, 1, 1)
        self.durb3 = DuRB(32, 7, 5, 2, 1)
        self.durb4 = DuRB(32, 11, 7, 2, 1)
        self.durb5 = DuRB(32, 11, 5, 1, 1)
        self.durb6 = DuRB(32, 11, 7, 3, 1)


        self.outc = outconv(32, 3)

    def forward(self, x):
        inx = self.inc(x)

        md1, bd1 = self.durb1(inx, inx)
        md2, bd2 = self.durb2(md1, bd1)
        md3, bd3 = self.durb3(md2, bd2)
        md4, bd4 = self.durb4(md3, bd3)
        md5, bd5 = self.durb5(md4, bd4)
        md6, _ = self.durb6(md5, bd5)

        outx = self.outc(md6)

        out = outx + x

        return out


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    )

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x

