
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEUNet(nn.Module):
    def __init__(self):
        super(SEUNet, self).__init__()
        self.inc = inconv(3, 32)
        self.se1 = SELayer(32)
        self.down1 = down(32, 64)
        self.se2 = SELayer(64)
        self.down2 = down(64, 128)
        self.se3 = SELayer(128)
        self.down3 = down(128, 256)
        self.se4 = SELayer(256)
        self.down4 = down(256, 512)
        self.se5 = SELayer(512)
        self.up1 = up(512, 256)
        self.se6 = SELayer(256)
        self.up2 = up(256, 128)
        self.se7 = SELayer(128)
        self.up3 = up(128, 64)
        self.se8 = SELayer(64)
        self.up4 = up(64, 32)
        self.se9 = SELayer(32)
        self.outc = outconv(32, 3)

    def forward(self, x):
        down1 = self.inc(x)
        se1 = self.se1(down1)
        down2 = self.down1(se1)
        se2 = self.se2(down2)
        down3 = self.down2(se2)
        se3 = self.se3(down3)
        down4 = self.down3(se3)
        se4 = self.se4(down4)
        down5 = self.down4(se4)
        se5 = self.se5(down5)
        up1 = self.up1(se5, se4)
        se6 = self.se6(up1)
        up2 = self.up2(se6, se3)
        se7 = self.se7(up2)
        up3 = self.up3(se7, se2)
        se8 = self.se8(up3)
        up4 = self.up4(se8, se1)
        se9 = self.se9(up4)
        up5 = self.outc(se9)
        return up5


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.AvgPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
