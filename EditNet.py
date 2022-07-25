import torch
import torch.nn as nn
import torchvision
import numpy as np

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        self.skip = nn.Conv2d(out_channel,+in_channel, out_channel, 1, 1, 0)
    def forward(self, x, x_f):
        x = self.upsample(x)
        out = self.conv(torch.cat([x, x_f], dim=1))
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        self.downsample = nn.MaxPool2d(2)
    def forward(self, x):
        skip_out = self.conv(x)
        out = self.downsample(skip_out)
        return out, skip_out


class MiddleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
        )
        self.downsample = nn.MaxPool2d(2)
        self.encoder = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.conv(x)
        residual = x
        x = self.encoder(x)
        return x + residual

class UNet(nn.Module):
    def __init__(self, input_channel=3, output_channel=3):
        super(UNet, self).__init__()
        nb_filter = [16, 32, 64, 128, 256]
        self.conv_init = nn.Conv2d(input_channel, nb_filter[0], 1, 1, 0)
        self.middle_1 = MiddleBlock(nb_filter[0], nb_filter[0])
        self.down1 = DownBlock(nb_filter[0], nb_filter[1])
        self.down2 = DownBlock(nb_filter[1], nb_filter[2])
        self.down3 = DownBlock(nb_filter[2], nb_filter[3])
        self.middle_2 = MiddleBlock(nb_filter[3], nb_filter[4])
        self.up4 = UpBlock(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.up3 = UpBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.up2 = UpBlock(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv_fin = nn.Sequential(
            nn.Conv2d(nb_filter[1], output_channel, 1, 1, 0),
            )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv_init(x)
        x = self.middle_1(x)
        x1_1, x1_f = self.down1(x)
        x2_1, x2_f = self.down2(x1_1)
        x3_1, x3_f = self.down3(x2_1)
        x_middle = self.middle_2(x3_1)
        x3_2 = self.up4(x_middle, x3_f)
        x2_2 = self.up3(x3_2, x2_f)
        x1_2 = self.up2(x2_2, x1_f)
        x_fin = self.conv_fin(x1_2)
        return self.sigmoid(x_fin)