import torch
import torch.nn as nn
from .model import Model


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
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
#         x = self.encoder(x)
#         x = self.decoder(x)
        if self.tta: # WIP
            y1 =  self.decoder(self.encoder(x))
            y2 = self.decoder(self.encoder(x.flip(3)))
            y = (y1 + y2.flip(3)) * 0.5
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
