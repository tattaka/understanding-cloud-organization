import torch.nn as nn
import torchvision.models as models
from .blocks import (RefineNetBlock, ResidualConvUnit,
                      RefineNetBlockImprovedPooling)
import sys
sys.path.append('../')
from common.blocks import *
from base.model import Model

class BaseRefineNetDecoder(Model):
    def __init__(self,
                 input_shape,
                 encoder_channels,
                 refinenet_block,
                 num_classes=1,
                 features=256,):
        super().__init__()
        input_channel, input_size = input_shape

        if input_size % 32 != 0:
            raise ValueError("{} not divisble by 32".format(input_shape))
        self.layer1_rn = nn.Conv2d(
            encoder_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            encoder_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            encoder_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            encoder_channels[0], 2 * features, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.refinenet4 = RefineNetBlock(2 * features,
                                         (2 * features, input_size // 32))
        self.refinenet3 = RefineNetBlock(features,
                                         (2 * features, input_size // 32),
                                         (features, input_size // 16))
        self.refinenet2 = RefineNetBlock(features,
                                         (features, input_size // 16),
                                         (features, input_size // 8))
        self.refinenet1 = RefineNetBlock(features, (features, input_size // 8),
                                         (features, input_size // 4))
        self.output_conv = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))
        
    def forward(self, x):
        layer_1_rn = self.layer1_rn(x[3])
        layer_2_rn = self.layer2_rn(x[2])
        layer_3_rn = self.layer3_rn(x[1])
        layer_4_rn = self.layer4_rn(x[0])
        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        out = nn.functional.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        return out
    
class RefineNetPoolingImproveDecoder(BaseRefineNetDecoder):
    def __init__(self,
                 input_shape,
                 encoder_channels,
                 num_classes=1,
                 features=256):
        """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling

        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_shape,
            encoder_channels,
            RefineNetBlockImprovedPooling,
            num_classes=num_classes,
            features=features)
        
class RefineNetDecoder(BaseRefineNetDecoder):
    def __init__(self,
                 input_shape,
                 encoder_channels,
                 num_classes=1,
                 features=256):
        """Multi-path 4-Cascaded RefineNet for image segmentation

        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_shape,
            encoder_channels,
            RefineNetBlock,
            num_classes,
            features=features)

