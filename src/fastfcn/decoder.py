import torch.nn as nn
import torchvision.models as models
import sys
sys.path.append('../')
from common.blocks import *
from base.model import Model
from .blocks import JPU, ASPP


class FastFCNDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            final_channels=1,
            use_batchnorm=True,
            mid_channel = 128, 
    ):
        super().__init__()
        self.jpu = JPU([encoder_channels[0],encoder_channels[1],encoder_channels[2]], mid_channel)
        
        self.aspp = ASPP(mid_channel*4, mid_channel, dilations=[1, (1, 4), (2, 8), (3, 12)])
        

        self.final_conv = nn.Conv2d(mid_channel, final_channels, kernel_size=(1, 1))
        
        self.initialize()

    def forward(self, x):
        x = self.jpu(x[0], x[1], x[2])
        x = self.aspp(x)
        x = nn.functional.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        x = self.final_conv(x)
        return x