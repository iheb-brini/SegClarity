# TODO clean code
# %%
import torch
import torch.nn as nn

#################
#  LUNET MODEL  #
#################


def create_conv_bn_relu(in_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1):
    """
    Create conv+BN+Relu layers
    """
    return (nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),)


def create_dil_block(in_c, out_c):
    """
    Create encoder block
    in_c (int): input channel
    out_c (int): output channel
    """
    conv = nn.Sequential(
        *create_conv_bn_relu(in_c, out_c),
        *create_conv_bn_relu(out_c, out_c),
        *create_conv_bn_relu(out_c, out_c, padding=2, dilation=2),
        *create_conv_bn_relu(out_c, out_c, padding=2, dilation=2),
    )
    return conv


def create_encoding_block(in_c, out_c):
    """
    Create encoding block with single conv
    """
    conv = nn.Sequential(
        *create_conv_bn_relu(in_c, out_c),
    )
    return conv


def create_encoding_block_double_conv(in_c, out_c):
    """
    Create encoding block with double conv
    """
    conv = nn.Sequential(
        *create_conv_bn_relu(in_c, out_c),
        *create_conv_bn_relu(out_c, out_c),
    )
    return conv


class lunet_model(nn.Module):
    def __init__(self, out_channels=4, features=[16, 32]):
        super(lunet_model, self).__init__()

        self.dil1 = create_dil_block(3, features[0])

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.dil2 = create_dil_block(features[0], features[0])

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.dil3 = create_dil_block(features[0], features[0])

        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.dil4 = create_dil_block(features[0], features[0])

        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bott = create_encoding_block_double_conv(features[0], features[0])

        self.tconv1 = nn.ConvTranspose2d(
            features[0], features[0], kernel_size=2, stride=2)

        self.conv1 = create_encoding_block(features[1], features[0])

        self.tconv2 = nn.ConvTranspose2d(
            features[0], features[0], kernel_size=2, stride=2)

        self.conv2 = create_encoding_block(features[1], features[0])

        self.tconv3 = nn.ConvTranspose2d(
            features[0], features[0], kernel_size=2, stride=2)

        self.conv3 = create_encoding_block(features[1], features[0])

        self.tconv4 = nn.ConvTranspose2d(
            features[0], features[0], kernel_size=2, stride=2)

        self.conv4 = create_encoding_block_double_conv(features[1], features[0])

        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        dil_1 = self.dil1(x)

        pool_1 = self.pool1(dil_1)

        dil_2 = self.dil2(pool_1)

        pool_2 = self.pool2(dil_2)

        dil_3 = self.dil3(pool_2)

        pool_3 = self.pool3(dil_3)

        dil_4 = self.dil4(pool_3)

        pool_4 = self.pool4(dil_4)

        bott = self.bott(pool_4)

        tconv_1 = self.tconv1(bott)

        concat1 = torch.cat((tconv_1, dil_4), dim=1)

        conv_1 = self.conv1(concat1)

        tconv_2 = self.tconv2(conv_1)

        concat2 = torch.cat((tconv_2, dil_3), dim=1)

        conv_2 = self.conv2(concat2)

        tconv_3 = self.tconv3(conv_2)

        concat3 = torch.cat((tconv_3, dil_2), dim=1)

        conv_3 = self.conv3(concat3)

        tconv_4 = self.tconv4(conv_3)

        concat4 = torch.cat((tconv_4, dil_1), dim=1)

        conv_4 = self.conv4(concat4)

        x = self.final_layer(conv_4)

        return x
