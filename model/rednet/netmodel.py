import math
from torch import nn


class REDNet(nn.Module):
    def __init__(self, num_features=64):
        super(REDNet, self).__init__()
        self.inconv = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )

        conv_layers = []
        deconv_layers = []

        for _ in range(9):
            conv_layers.append(nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
                ))

        for _ in range(9):
            deconv_layers.append(nn.Sequential(
                nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                ))

        self.outconv = nn.ConvTranspose2d(num_features, 3, kernel_size=3, padding=1)

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.inconv(x)
        conv2 = self.conv_layers[0](conv1)
        conv3 = self.conv_layers[1](conv2)
        conv4 = self.conv_layers[2](conv3)
        conv5 = self.conv_layers[3](conv4)
        conv6 = self.conv_layers[4](conv5)
        conv7 = self.conv_layers[5](conv6)
        conv8 = self.conv_layers[6](conv7)
        conv9 = self.conv_layers[7](conv8)
        conv10 = self.conv_layers[8](conv9)

        deconv1 = self.relu(self.deconv_layers[0](conv10) + conv9)
        deconv2 = self.relu(self.deconv_layers[1](deconv1) + conv8)
        deconv3 = self.relu(self.deconv_layers[2](deconv2) + conv7)
        deconv4 = self.relu(self.deconv_layers[3](deconv3) + conv6)
        deconv5 = self.relu(self.deconv_layers[4](deconv4) + conv5)
        deconv6 = self.relu(self.deconv_layers[5](deconv5) + conv4)
        deconv7 = self.relu(self.deconv_layers[6](deconv6) + conv3)
        deconv8 = self.relu(self.deconv_layers[7](deconv7) + conv2)
        deconv9 = self.relu(self.deconv_layers[8](deconv8) + conv1)
        deconv10 = self.relu(self.outconv(deconv9) + x)

        return deconv10
