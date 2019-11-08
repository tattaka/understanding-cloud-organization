import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')

from common.blocks import Conv2dReLU, SCSEModule, CBAMModule, AdaptiveConcatPool2d
from base.model import Model
from .center_block import *

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)
        elif attention_type == 'cbam':
            self.attention1 = CBAMModule(in_channels)
            self.attention2 = CBAMModule(out_channels)

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

class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=None,
            attention_type=None,
            classification = False,
    ):
        super().__init__()

        if center == 'normal':
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        elif center == 'fpa':
            channels = encoder_channels[0]
            self.center = FPA(channels)
        elif center == 'aspp':
            channels = encoder_channels[0]
            self.center = ASPP(channels, channels, dilations=[1, (1, 6), (2, 12), (3, 18)])
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
        
        self.classification = classification
        if self.classification:
            self.linear_feature = nn.Sequential(
                nn.Conv2d(encoder_channels[0], 64, kernel_size=1),
                AdaptiveConcatPool2d(1),
                Flatten(),
                nn.Dropout(),
                nn.Linear(128, final_channels)
            )
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
        if self.classification:
            class_refine = self.linear_feature(encoder_head)[:, :, None, None]
#             print(x.size(), class_refine.size())
            x = x * class_refine
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
            self.center = FPA(channels)
        elif center == 'aspp':
            channels = encoder_channels[0]
            self.center = ASPP(inplanes=channels, mid_c=channels/2, dilations=[1, 6, 12, 18])
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
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)