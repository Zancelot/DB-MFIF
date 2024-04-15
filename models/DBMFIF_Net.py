import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util import ResBlock, conv


__all__ = [
    'DBMFIF', 'DBMFIF_bn'
]

class DBMFIFNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(DBMFIFNet, self).__init__()

        n_feats_b = 64
        kernel_size = 3
        defocus_in = 6
        deblur_out = 3

        self.conv01a = conv(defocus_in, 64,   3)
        self.conv02a = conv(64, n_feats_b, 3)
        self.conv03a = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.conv04a = ResBlock(2*n_feats_b, 2*n_feats_b, kernel_size)
        self.conv04a1 = nn.Conv2d(2*n_feats_b, n_feats_b, 1)
        self.conv05a = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.conv06a = ResBlock(2*n_feats_b, 2*n_feats_b, kernel_size)
        self.conv06a1 = nn.Conv2d(2*n_feats_b, n_feats_b, 1)
        self.conv07a = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.conv08a = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.conv09a = conv(n_feats_b, 1, kernel_size)
        self.conv01b = conv(defocus_in, n_feats_b, kernel_size)
        self.convRes01b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes02b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes03b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes04b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes05b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes06b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes07b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes08b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes09b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes10b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes11b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes12b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes13b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes14b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes15b = ResBlock(2 *n_feats_b, 2 *n_feats_b, kernel_size)
        self.convRes15c1 = nn.Conv2d(2 * n_feats_b, n_feats_b, 1)
        self.convRes16b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes17c = ResBlock(2*n_feats_b, 2*n_feats_b, kernel_size)
        self.convRes17c1 = nn.Conv2d(2 * n_feats_b, n_feats_b, 1)
        self.conv02c = conv(n_feats_b, n_feats_b, kernel_size)
        self.conv03c = conv(n_feats_b, deblur_out, kernel_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)


    def forward(self, input_image):

        b, _, h, w = input_image.size()

        out_conv01b = self.conv01b(input_image)
        out_convRes01b = out_conv01b + self.convRes01b(out_conv01b)
        out_convRes02b = out_convRes01b + self.convRes02b(out_convRes01b)
        out_convRes03b = out_convRes02b + self.convRes03b(out_convRes02b)
        out_convRes04b = out_convRes03b + self.convRes04b(out_convRes03b)
        out_convRes05b = out_convRes04b + self.convRes05b(out_convRes04b)
        out_convRes06b = out_convRes05b + self.convRes06b(out_convRes05b)
        out_convRes07b = out_convRes06b + self.convRes07b(out_convRes06b)
        out_convRes08b = out_convRes07b + self.convRes08b(out_convRes07b)
        out_convRes09b = out_convRes08b + self.convRes09b(out_convRes08b)
        out_convRes10b = out_convRes09b + self.convRes10b(out_convRes09b)
        out_convRes11b = out_convRes10b + self.convRes11b(out_convRes10b)
        out_convRes12b = out_convRes11b + self.convRes12b(out_convRes11b)
        out_convRes13b = out_convRes12b + self.convRes13b(out_convRes12b)
        out_convRes14b = out_convRes13b + self.convRes14b(out_convRes13b)
        out_conv01a = self.conv01a(input_image)
        out_conv02a = self.conv02a(out_conv01a)
        out_conv03a = self.conv03a(out_conv02a) + out_conv02a
        out_conv03a = torch.cat([out_conv03a, out_convRes05b], 1)
        out_conv04a = self.conv04a(out_conv03a) + out_conv03a
        out_conv04a1 = self.conv04a1(out_conv04a)
        out_conv05a = self.conv05a(out_conv04a1) + out_conv04a1
        out_conv05a = torch.cat([out_conv05a, out_convRes10b], 1)
        out_conv06a = self.conv06a(out_conv05a) + out_conv05a
        out_conv06a1 = self.conv06a1(out_conv06a)
        out_conv07a = self.conv07a(out_conv06a1) + out_conv06a1
        out_conv08a = self.conv08a(out_conv07a) + out_conv07a
        out_conv09a = self.conv09a(out_conv08a)
        out_convRes14b = torch.cat([out_convRes14b, out_conv07a], 1)
        out_convRes15b = out_convRes14b + self.convRes15b(out_convRes14b)
        out_convRes15b1 = self.convRes15c1(out_convRes15b)
        out_convRes16b = out_convRes15b1 + self.convRes16b(out_convRes15b1)
        out_convRes16b = torch.cat([out_convRes16b, out_conv08a], 1)
        out_convRes17b = out_convRes16b + self.convRes17c(out_convRes16b)
        out_convRes17b1 = self.convRes17c1(out_convRes17b)
        out_conv02b = self.conv02c(out_convRes17b1)
        out_conv03b = self.conv03c(out_conv02b)
        output = torch.cat((out_conv09a, out_conv03b), 1)

        return output

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def DBMFIF(data=None):
    model = DBMFIFNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def DBMFIF_bn(data=None):
    model = DBMFIFNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
