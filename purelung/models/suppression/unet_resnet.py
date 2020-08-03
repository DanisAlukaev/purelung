'''
    A simple plain U-net model that improves the autoencoder model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .common_blocks import ConvBlock, ConvTransposeBlock, ResBlock


class DecoderBlock(nn.Module):

    def __init__(self, in_features, out_features, collaps_rate, kernel_sizes=[2, 3],
                 strides=[2, 1], paddings=[0, 1], bias=False):
        super(DecoderBlock, self).__init__()

        self.conv_transpose = ConvTransposeBlock(in_features, out_features, kernel_sizes[0], strides[0], paddings[0],
                                                 bias)
        self.res_block = ResBlock(in_features, out_features, kernel_sizes[1], strides[1],
                                  paddings[1], bias, False, True)

    def forward(self, lower_dim, higher_dim):
        upsample = self.conv_transpose(lower_dim)
        res = self.res_block(torch.cat((upsample, higher_dim), 1))
        return res


class UnetResnetEncoder(nn.Module):

    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5, 3, 3, 3, 3, 3],
                 strides=[1, 1, 1, 1, 1, 1], paddings=[2, 1, 1, 1, 1, 1], bias=False):
        super(UnetResnetEncoder, self).__init__()
        self.encoder_blocks = []
        features_num = start_features_num
        start_in_channels = in_channels
        alias = "_".join(["Conv2d", str(kernel_sizes[0]), str(features_num)])
        self.encoder_blocks.append(ConvBlock(in_channels, features_num, kernel_sizes[0], strides[0],
                                             paddings[0], bias))
        for i in range(1, len(kernel_sizes)):
            alias = "_".join(["ResBlock", str(i), str(kernel_sizes[i]), str(features_num)])
            in_channels = features_num
            if i % 2 == 0:
                isDownsample = True
                features_num *= expand_rate
            else:
                isDownsample = False
            self.encoder_blocks.append(ResBlock(in_channels, features_num, kernel_sizes[i], strides[i],
                                                paddings[i], bias, isDownsample))
        self.encoder_blocks = nn.Sequential(*self.encoder_blocks)

    def forward(self, x):
        encoder_passes = []
        for idx, encoder_block in enumerate(self.encoder_blocks.children()):
            x = encoder_block(x)
            if idx % 2 == 1:
                encoder_passes.append(x)
        return encoder_passes


class UnetDecoder(nn.Module):

    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5, 2, 5, 2, 5],
                 strides=[1, 2, 1, 2, 1], paddings=[2, 0, 2, 0, 2], bias=False):
        super(UnetDecoder, self).__init__()
        self.decoder_blocks = []

        self.num_downsampling = len(kernel_sizes) // 2 - 1
        max_features = start_features_num * (expand_rate ** self.num_downsampling)

        for i in range(self.num_downsampling):
            decoder_block = DecoderBlock(max_features, max_features // expand_rate, expand_rate, [2, 3],
                                         [2, 1], [0, 1], bias)
            max_features = max_features // expand_rate
            self.decoder_blocks.append(decoder_block)
        self.decoder_blocks = nn.Sequential(*self.decoder_blocks)

    def forward(self, encoder_passes):
        x = encoder_passes[-1]
        for idx, decoder_block in enumerate(self.decoder_blocks.children()):
            x = decoder_block(x, encoder_passes[-(idx + 2)])
        return x


class MyResnetUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, start_features_num=32, expand_rate=2,
                 kernel_sizes=[5, 3, 3, 3, 3, 3, 3, 3],
                 strides=[1, 1, 1, 1, 1, 1, 1, 1], paddings=[2, 1, 1, 1, 1, 1, 1, 1], bias=False,
                 final_activation='relu'):
        super(MyResnetUnet, self).__init__()

        self.encoder = []
        self.decoder_blocks = []
        features_num = start_features_num
        start_in_channels = in_channels

        self.encoder = UnetResnetEncoder(in_channels, start_features_num, expand_rate, kernel_sizes,
                                         strides, paddings, bias)

        self.decoder = UnetDecoder(in_channels, start_features_num, expand_rate, kernel_sizes,
                                   strides, paddings, bias)

        self.final_conv = ResBlock(start_features_num, start_features_num, kernel_sizes[-1],
                                   strides[-1], paddings[-1], bias, False, False)
        self.final_conv2 = ConvBlock(start_features_num, out_channels, kernel_sizes[0],
                                     strides[0], paddings[0], bias, final_activation)

    def forward(self, x):
        encoder_passes = self.encoder(x)

        x = self.decoder(encoder_passes)
        x = self.final_conv(x)
        x = self.final_conv2(x)
        return x


def myResnetUnet(in_channels, out_channels, final_activation):
    return MyResnetUnet(in_channels, out_channels, 16, 2, final_activation=final_activation)
