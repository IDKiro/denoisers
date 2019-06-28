
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobile import MobileBottleneck, Hswish

class MobileUNet(nn.Module):
    def __init__(self):
        super(MobileUNet, self).__init__()
        self.inc = inconv(3, 12)

        self.down1 = nn.Sequential(
            MobileBottleneck(12, 16, 3, 2, 32, 'RE')
        )

        self.down2 = nn.Sequential(
            MobileBottleneck(16, 16, 3, 1, 16, 'RE'),
            MobileBottleneck(16, 24, 3, 2, 64, 'RE')
        )

        self.down3 = nn.Sequential(
            MobileBottleneck(24, 24, 3, 1, 72, 'RE'),
            MobileBottleneck(24, 40, 5, 2, 72, 'RE')
        )

        self.down4 = nn.Sequential(
            MobileBottleneck(40, 40, 5, 1, 120, 'RE'),
            MobileBottleneck(40, 40, 5, 1, 120, 'RE'),
            MobileBottleneck(40, 80, 3, 2, 240, 'HS')
        )

        self.bneck5 = nn.Sequential(
            MobileBottleneck(80, 80, 3, 1, 200, 'HS'),
            MobileBottleneck(80, 80, 3, 1, 184, 'HS'),
            MobileBottleneck(80, 80, 3, 1, 184, 'HS'),
            MobileBottleneck(80, 80, 3, 1, 200, 'HS')
        )

        self.up1 = up(80, 40)
        self.bneck6 = nn.Sequential(
            MobileBottleneck(40, 40, 5, 1, 120, 'RE'),
            MobileBottleneck(40, 40, 5, 1, 120, 'RE')
        )

        self.up2 = up(40, 24)
        self.bneck7 = nn.Sequential(
            MobileBottleneck(24, 24, 3, 1, 72, 'RE')
        )

        self.up3 = up(24, 16)
        self.bneck8 = nn.Sequential(
            MobileBottleneck(16, 16, 3, 1, 16, 'RE')
        )

        self.up4 = up(16, 12)
        self.outc = outconv(12, 3)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        bneck5 = self.bneck5(down4)

        up1 = self.up1(bneck5, down3)
        bneck6 = self.bneck6(up1)

        up2 = self.up2(bneck6, down2)
        bneck7 = self.bneck7(up2)

        up3 = self.up3(bneck7, down1)
        bneck8 = self.bneck8(up3)
        
        up4 = self.up4(bneck8, inx)
        outx = self.outc(up4)

        out = outx + x

        return out


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        Hswish()
                    )

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

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



