import torch
import torch.nn as nn
from .model import Model
import cv2
import numpy as np
import sys
sys.path.append('../')
from common.blocks import Conv2dReLU, Swish

class EncoderDecoder(Model):
    
    def __init__(self, encoder, decoder, activation, tta=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tta = tta

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif actication == 'Swith':
            self.activation = Swish()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/"Swith"/callable/None')

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
#         x = self.encoder(x)
#         x = self.decoder(x)
        if self.tta: 
            y1 =  self.decoder(self.encoder(x))
            y2 = self.decoder(self.encoder(x.flip(3))).flip(3)
            y3 = self.decoder(self.encoder(x.flip(2))).flip(2)
            y4 = self.decoder(self.encoder(x.flip((2, 3)))).flip((2, 3))
            y = (y1 + y2 + y3 + y4) * 0.25
        else:
            y = self.decoder(self.encoder(x))
#         print(x.size(), y.size())
        return y

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x
sigmoid = lambda x: 1 / (1 + torch.exp(-x))
class RefineEncoderDecoder(Model):
    
    def __init__(self, infer_model, encoder, decoder, activation, tta=False, adapt_input=None, preprocess=True):
        super().__init__()
        self.infer_model = infer_model
        self.encoder = encoder
        self.decoder = decoder
        self.tta = tta
        self.adapt_input = adapt_input
        self.preprocess = preprocess
        if self.adapt_input is not None:
            self.first_conv = Conv2dReLU(adapt_input, 3, kernel_size=1, use_batchnorm=True)

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif actication == 'Swish':
            self.activation = Swith()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/"Swith"/callable/None')

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
#         x = self.encoder(x)
#         x = self.decoder(x)
        
        x_pred = sigmoid(self.infer_model.predict(x))
        if self.preprocess:
            x_pred = torch.where(x_pred<0.7, torch.zeros(x_pred.shape).cuda(), torch.ones(x_pred.shape).cuda())
        x = torch.cat([x, x_pred], dim=1)
        if self.adapt_input is not None:
            x = self.first_conv(x)
        if self.tta: 
            y1 = self.decoder(self.encoder(x))
            y2 = self.decoder(self.encoder(x.flip(3))).flip(3)
            y3 = self.decoder(self.encoder(x.flip(2))).flip(2)
            y4 = self.decoder(self.encoder(x.flip((2, 3)))).flip((2, 3))
            y = (y1 + y2 + y3 + y4) * 0.25
        else:
            y = self.decoder(self.encoder(x))
#         print(x.size(), y.size())
        return y

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x
    