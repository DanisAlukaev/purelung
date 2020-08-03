import torch
import torch.nn as nn
import torch.nn.functional as F

activations = {
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'softmax': F.softmax
}


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='relu'):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.activation != 'linear':
            x = activations[self.activation](x)
        return x


class ConvTransposeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='relu'):
        super(ConvTransposeBlock, self).__init__()
        self.activation = activation
        self.conv_tr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                          padding, bias=bias)

    def forward(self, x):
        x = self.conv_tr(x)
        if self.activation != 'linear':
            x = activations[self.activation](x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, isDownsample=False,
                 isEqDecoder=False):
        super(ResBlock, self).__init__()

        self.isDownsample = isDownsample
        self.isEqDecoder = isEqDecoder
        if self.isDownsample:
            self.downsample = nn.AvgPool2d(2, 2, 0)
            self.equalize = nn.Conv2d(in_channels, out_channels, 2, 2, 0, bias=bias)

        if self.isEqDecoder:
            self.eq_dec = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        if self.isDownsample:
            out = self.downsample(x)
        else:
            out = x
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        if self.isDownsample:
            out = self.equalize(identity) + out
        elif self.isEqDecoder:
            out = self.eq_dec(identity) + out
        else:
            out = identity + out
        out = F.relu(out)
        return out
