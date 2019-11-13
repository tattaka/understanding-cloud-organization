import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import build_decoder
from .backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, encoder_name='resnet50', output_stride=16, num_classes=21,
                 freeze_bn=False, activation=None, tta=False):
        super(DeepLab, self).__init__()
        if encoder_name.split()[0] == 'drn':
            output_stride = 8
        
        BatchNorm = nn.BatchNorm2d

        self.encoder = build_backbone(encoder_name, output_stride)
        self.decoder = build_decoder(num_classes, encoder_name, output_stride)
        self.tta = tta
        if freeze_bn:
            self.freeze_bn()
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

    def forward(self, input):
        if self.tta: 
            x, low_level_feat = self.encoder(input)
            x = self.decoder(x, low_level_feat)
            y1 = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
            
            x, low_level_feat = self.encoder(input.flip(3))
            x = self.decoder(x, low_level_feat)
            y2 = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True).flip(3)
            
            x, low_level_feat = self.encoder(input.flip(2))
            x = self.decoder(x, low_level_feat)
            y3 = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True).flip(2)
            
            x, low_level_feat = self.encoder(input.flip((2, 3)))
            x = self.decoder(x, low_level_feat)
            y4 = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True).flip((2, 3))
            
            y = (y1 + y2 + y3 + y4) * 0.25
        else:
            x, low_level_feat = self.encoder(input)
            x = self.decoder(x, low_level_feat)
            y = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
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
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


