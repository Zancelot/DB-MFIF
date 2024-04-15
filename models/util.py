import torch.nn as nn


def conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size//2), bias=bias)


def ResBlock(in_channels: object, out_channels: object, kernel_size: object, bias: object = True, batchNorm: object = False) -> object:
    if batchNorm:
        return nn.Sequential(
            conv(in_channels, in_channels, kernel_size, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            conv(in_channels, out_channels, kernel_size, bias=True),
            nn.BatchNorm2d(out_channels)
        )
    else:
        return nn.Sequential(
            conv(in_channels, in_channels, kernel_size, bias=True),
            nn.ReLU(True),
            conv(in_channels, out_channels, kernel_size, bias=True)
        )
