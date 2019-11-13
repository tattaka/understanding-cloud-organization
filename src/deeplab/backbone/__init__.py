from . import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride)
    elif backbone == 'drn_d_54':
        return drn.drn_d_54()
    elif backbone == 'mobilenet_v2':
        return mobilenet.MobileNetV2(output_stride)
    else:
        raise NotImplementedError
