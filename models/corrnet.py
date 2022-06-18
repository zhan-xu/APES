import torch
from torch.nn import Sequential, Parameter
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['corrnet']


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.InstanceNorm2d(mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv_gated(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv2d_1 = nn.Conv2d(in_channels, mid_channels, 3, 1, padding=1, dilation=1)
        self.mask_conv2d_1 = nn.Conv2d(in_channels, mid_channels, 3, 1, padding=1, dilation=1)
        self.norm_1 = nn.BatchNorm2d(mid_channels)

        self.conv2d_2 = nn.Conv2d(mid_channels, out_channels, 3, 1, padding=1, dilation=1)
        self.mask_conv2d_2 = nn.Conv2d(mid_channels, out_channels, 3, 1, padding=1, dilation=1)
        self.norm_2 = nn.BatchNorm2d(out_channels)

        self.sigmoid = torch.nn.Sigmoid()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        conv_1 = self.activation(self.conv2d_1(x))
        mask_1 = self.sigmoid(self.mask_conv2d_1(x))
        x_1 = conv_1 * mask_1
        x_1 = self.norm_1(x_1)

        conv_2 = self.activation(self.conv2d_2(x_1))
        mask_2 = self.sigmoid(self.mask_conv2d_2(x_1))
        x_2 = conv_2 * mask_2
        x_2 = self.norm_2(x_2)
        return x_2


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, gated=False):
        super().__init__()
        if gated:
            self.maxpool_conv = Sequential(
                nn.MaxPool2d(2),
                DoubleConv_gated(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, gated=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            if gated:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv_gated(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_out = self.conv(x)
        return x_out


class UNet_gated(nn.Module):
    def __init__(self, chn_in, chn_mid, chn_out, downsample_time, bilinear=True):
        assert len(chn_mid) == (downsample_time + 1)
        super(UNet_gated, self).__init__()
        self.downsample_time = downsample_time
        self.inc = DoubleConv_gated(chn_in, chn_mid[0])
        factor = 2 if bilinear else 1

        # down
        downlayers = []
        for t in range(downsample_time):
            if t == downsample_time - 1:
                downlayers += [Down(chn_mid[t], chn_mid[t+1] // factor, gated=True)]
            else:
                downlayers += [Down(chn_mid[t], chn_mid[t+1], gated=True)]
        self.downlayers = Sequential(*downlayers)

        # up
        uplayers = []
        for t in range(downsample_time):
            if t == downsample_time - 1:
                uplayers += [Up(chn_mid[-1 - t], chn_mid[-2 - t], bilinear, gated=False)]
            else:
                uplayers += [Up(chn_mid[-1 - t], chn_mid[-2 - t] // factor, bilinear, gated=False)]
        self.uplayers = Sequential(*uplayers)

        self.out = OutConv(chn_mid[0], chn_out)

    def forward(self, x_in, mask_in):
        x_cat = torch.cat((x_in, mask_in), dim=1)
        x_down = [self.inc(x_cat)]
        for t in range(self.downsample_time):
            x_down.append(self.downlayers[t](x_down[-1]))

        x_up = []
        for t in range(self.downsample_time):
            if t == 0:
                x_up.append(self.uplayers[t](x_down[-1], x_down[-2]))
            else:
                x_up.append(self.uplayers[t](x_up[-1], x_down[-2-t]))

        x_out = self.out(x_up[-1])
        return x_out


class CorrNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, tau=0.07):
        super(CorrNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.tau = Parameter(torch.Tensor([tau]))
        self.unet_shared = UNet_gated(chn_in=self.n_channels, chn_mid=[32, 64, 128, 256, 512],
                                      chn_out=self.n_classes, downsample_time=4, bilinear=bilinear)

    def forward(self, img1, img2, mask_1, mask_2):
        img1_out = self.unet_shared(img1, mask_1)
        img1_out = F.normalize(img1_out, dim=1)

        img2_out = self.unet_shared(img2, mask_2)
        img2_out = F.normalize(img2_out, dim=1)

        return img1_out, img2_out, self.tau


def corrnet(**kwargs):
    model = CorrNet(n_channels=kwargs["n_channels"], n_classes=kwargs["n_classes"], bilinear=kwargs["bilinear"])
    return model