import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')

from common.blocks import Conv2dReLU, SCSEModule
from base.model import Model


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)

        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.block(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None, upsample=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        return self.block(x)
        
class FeaturePyramidAttention(nn.Module):
    def __init__(self, channels):
        """
        Feature Pyramid Attention
        https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch/blob/master/networks.py
        :type channels: int
        """
        super(FeaturePyramidAttention, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Master branch
        h, w = x.size(2), x.size(3)
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)
        x1_2 = self.relu(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)
        x2_2 = self.relu(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)
        x3_2 = self.relu(x3_2)

        # Merge branch 1 and 
        x3_upsample = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3_2)
        x2_merge = x2_2 + x3_upsample
        x2_upsample = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x2_2)
        x1_merge = x1_2 + x2_upsample
        x_master = x_master * nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x1_merge)
        
        out = x_master + x_gpb

        return out

class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=None,
            attention_type=None
    ):
        super().__init__()

        if center == 'normal':
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        elif center == 'fpa':
            channels = encoder_channels[0]
            self.center = FeaturePyramidAttention(channels)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)
        return x
    
    
class HyperColumnsDecoder(Model):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=None,
            attention_type=None
    ):
        super().__init__()

        if center == 'normal':
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        elif center == 'fpa':
            channels = encoder_channels[0]
            self.center = FeaturePyramidAttention(channels)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        
        self.connect_conv1_2 = nn.Sequential(
            nn.Conv2d(out_channels[0], final_channels, kernel_size=(1, 1)), 
            nn.BatchNorm2d(final_channels))
        
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.connect_conv2_3 = nn.Sequential(
            nn.Conv2d(out_channels[1], final_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(final_channels))
        
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.connect_conv3_4 = nn.Sequential(
            nn.Conv2d(out_channels[2], final_channels, kernel_size=(1, 1)), 
            nn.BatchNorm2d(final_channels))
        
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.connect_conv4_5 = nn.Sequential(
            nn.Conv2d(out_channels[3], final_channels, kernel_size=(1, 1)), 
            nn.BatchNorm2d(final_channels))
        
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.connect_conv5_6 = nn.Sequential(
            nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(final_channels))
        
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))
        
        self.hc_conv = nn.Sequential(
            Conv2dReLU(6*final_channels, out_channels[4], kernel_size=3, padding=1, stride=1, use_batchnorm=use_batchnorm),
            nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1)))
        
        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)
        
        x = self.layer1([encoder_head, skips[0]])
        xc = self.connect_conv1_2(F.interpolate(x, scale_factor=16, mode='nearest'))
        x = self.layer2([x, skips[1]])
        xc = torch.cat((xc, self.connect_conv2_3(F.interpolate(x, scale_factor=8, mode='nearest'))), 1)
        x = self.layer3([x, skips[2]])
        xc = torch.cat((xc,  self.connect_conv3_4(F.interpolate(x, scale_factor=4, mode='nearest'))), 1)
        x = self.layer4([x, skips[3]])
        xc = torch.cat((xc,  self.connect_conv4_5(F.interpolate(x, scale_factor=2, mode='nearest'))), 1)
        x = self.layer5([x, None])
        xc = torch.cat((xc,  self.connect_conv5_6(x)), 1)
        x = self.final_conv(x)
        x = torch.cat((x, xc), 1)
        x = self.hc_conv(x)

        return x
