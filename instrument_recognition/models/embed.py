from functools import reduce
from operator import __add__

import torch
from torch import nn

class ConvBlock(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, 
                 padding = 'same'):
        super().__init__()
        assert isinstance(kernel_size, tuple)

        if padding == 'same' and stride == 1:
            padding = reduce(__add__,
                                [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        else:
            raise ValueError(' implemented anything other than same padding and stride 1')

        self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_channels=in_channels,  out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x.shape)
        x = self.pad(x)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.batchnorm(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        return x


class CQT2DFTEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.batchnorm0 = nn.BatchNorm2d(1)
        self.conv1 = ConvBlock(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.conv2 = ConvBlock(in_channels=32, out_channels=32, kernel_size=(3, 3))

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = ConvBlock(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv4 = ConvBlock(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.conv5 = ConvBlock(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.conv6 = ConvBlock(in_channels=128, out_channels=128, kernel_size=(3, 3))

        self.conv7 = ConvBlock(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.conv8 = ConvBlock(in_channels=128, out_channels=128, kernel_size=(3, 3))

        self.pooooool = nn.MaxPool2d(kernel_size=(15, 4), stride=(15, 4))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    
    def forward(self, x):
        # print(x.shape)
        x = self.batchnorm0(x)

        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.conv5(x)
        # print(x.shape)
        x = self.conv6(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.conv7(x)
        # print(x.shape)
        x = self.conv8(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.pooooool(x)
        x = self.flatten(x)
        # print(x.shape)



        return x
