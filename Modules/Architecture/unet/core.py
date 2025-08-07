# TODO clean code
# %%
import torch
import torch.nn as nn

#################
#  UNET MODEL  #
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

def block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
        )
    return conv


class unet_model(nn.Module):
    def __init__(self, out_channels=4,features=[64, 128, 256, 512]):
        super(unet_model, self).__init__()
        self.conv1 = create_encoding_block_double_conv(3,features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        
        self.conv2 = create_encoding_block_double_conv(features[0],features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        
        self.conv3 = create_encoding_block_double_conv(features[1],features[2])
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        
        self.conv4 = create_encoding_block_double_conv(features[2],features[3])
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        
        self.bottleneck = create_encoding_block_double_conv(features[3],features[3]*2)
        
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.conv5 = create_encoding_block_double_conv(features[3]*2,features[3])
        
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.conv6 = create_encoding_block_double_conv(features[3],features[2])
        
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.conv7 = create_encoding_block_double_conv(features[2],features[1])
        
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.conv8 = create_encoding_block_double_conv(features[1],features[0])  
        
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        
        #encoder
        x_1 = self.conv1(x) #to_concat
        x_2 = self.pool1(x_1)
        x_3 = self.conv2(x_2) #to_concat
        x_4 = self.pool2(x_3)
        x_5 = self.conv3(x_4) #to_concat
        x_6 = self.pool3(x_5)
        x_7 = self.conv4(x_6) #to_concat
        x_8 = self.pool4(x_7)
        x_9 = self.bottleneck(x_8) 
        
        #decoder
        x_10 = self.tconv1(x_9)
        x_11 = torch.cat((x_7, x_10), dim=1)
        x_12 = self.conv5(x_11) 
        x_13 = self.tconv2(x_12)
        x_14 = torch.cat((x_5, x_13), dim=1)
        x_15 = self.conv6(x_14)
        x_16 = self.tconv3(x_15)
        x_17 = torch.cat((x_3, x_16), dim=1)
        x_18 = self.conv7(x_17) 
        x_19 = self.tconv4(x_18)
        x_20 = torch.cat((x_1, x_19), dim=1)
        x_21 = self.conv8(x_20)
        x = self.final_layer(x_21)
        
        return x
