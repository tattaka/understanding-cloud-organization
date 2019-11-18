import pretrainedmodels
import segmentation_models_pytorch as smp
import torch.nn as nn
import torchvision
import torch
from typing import Optional, Type

from cloud_utils import * 
from dataset import * 
from optimizers import * 
import encoders

import unet.model
import fpn.model
import linknet.model
import pspnet.model
import refinenet.model
import deeplab.model
import fastfcn.model

def get_model(model_type: str = 'Unet',
              encoder: str = 'resnet18',
              encoder_weights: str = 'imagenet',
              activation: str = None,
              n_classes: int = 4,
              task: str = 'segmentation',
              attention_type: str = None, 
              center: str = None,
              source: str = 'pretrainedmodels',
              head: str = 'simple', 
              tta:bool = False,
              classification:bool = False
             ):
    
    if task == 'segmentation':
        if model_type == 'UNet':
            model = unet.model.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                attention_type=attention_type,
                activation=activation,
                center=center,
                tta=tta, 
            )
        elif model_type == 'HyperColumns':
            model = unet.model.HyperColumns(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                attention_type=attention_type,
                activation=activation, 
                center=center,
                tta=tta,
            )

        elif model_type == 'LinkNet':
            model = linknet.model.Linknet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation,
                tta=tta,
            )

        elif model_type == 'FPN':
            model = fpn.model.FPN(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation,
                tta=tta,
            )
        elif model_type == 'PSPNet':
            model = pspnet.model.PSPNet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation,
                tta=tta
            )
        elif model_type == 'RefineNetPoolingImprove':
            model = refinenet.model.RefineNetPoolingImprove(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation,
                tta=tta,
            )
        elif model_type == 'RefineNet':
            model = refinenet.model.RefineNet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation,
                tta=tta,
            )
        elif model_type == 'DeepLab':
            model = deeplab.model.DeepLab(
                encoder_name=encoder,
                num_classes=n_classes,
                activation=activation,
                tta=tta,
            )
        elif model_type == 'FastFCN':
            model = fastfcn.model.FastFCN(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation,
                tta=tta, 
            )
        elif model_type == 'FastFCNImprove':
            model = fastfcn.model.FastFCNImprove(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation,
                tta=tta, 
            )

#         elif model_type == 'resnet34_fpn':
#             model = resnet34_fpn(num_classes=n_classes, fpn_features=128)

#         elif model_type == 'effnetB4_fpn':
#             model = effnetB4_fpn(num_classes=n_classes, fpn_features=128)

        else:
            model = None

    elif task == 'classification':
        if source == 'pretrainedmodels':
            model_fn = pretrainedmodels.__dict__[encoder]
            model = model_fn(num_classes=1000, pretrained=encoder_weights)
        elif source == 'torchvision':
            model = torchvision.models.__dict__[encoder](pretrained=encoder_weights)

        if head == 'simple':
            model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)
        else:
            model = Net(net=model)

    return model

def get_ref_model(
               infer_model,
              encoder: str = 'resnet18',
              encoder_weights: str = 'imagenet',
              activation: str = None,
              n_classes: int = 4,
              center:str = None,
              attention_type:str=None,
              source: str = 'pretrainedmodels',
              tta:bool = False,
              preprocess = True
             ):
    
    model = unet.model.Ref_Unet(
        infer_model = infer_model, 
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=n_classes,
        activation=activation,
        center=center,
        attention_type=attention_type,
        tta=tta,
        adapt_input = n_classes+3,
        preprocess = preprocess
    )

    return model


class Flatten(nn.Module):
    """
    Simple class for flattening layer.
    """
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Net(nn.Module):
    def __init__(
            self,
            num_classes: int = 4,
            p: float = 0.2,
            net = None) -> None:
        """
        Custom head architecture
        Args:
            num_classes: number of classes
            p: dropout probability
            net: original model
        """
        super().__init__()
        modules = list(net.children())[:-1]
        n_feats = list(net.children())[-1].in_features
        # add custom head
        modules += [nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(81536),
            nn.Dropout(p),
            nn.Linear(81536, n_feats),
            nn.Linear(n_feats, num_classes),
            nn.Sigmoid()
        )]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return logits

if __name__ == '__main__':
    x = np.zeros((3, 3, 384, 576), dtype="f")
    x = torch.from_numpy(x)
    print("input shape:", x.size())
    model = get_model(model_type = 'FastFCN',
              encoder= 'resnet50',
              encoder_weights= 'imagenet',
              activation = None,
              n_classes = 4,
              task = 'segmentation')

    y = model(x)
    print("out shape:", y.size())