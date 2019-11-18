import sys
from functools import partial

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import segmentation_models_pytorch as smp
sys.path.append('../')
import utils

class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.ensemble_filestems = ["segmentation_UNet_densenet121_aspp_384x576_tta", "segmentation_Linknet_densenet121_384x576_tta", "segmentation_Linknet_efficientnet-b3_384x576_tta", "segmentation_UNet_densenet121_fpa_256x512_tta", "segmentation_FPN_efficientnet-b3_384x576_tta", "segmentation_RefineNet_densenet121_384x576_tta", "segmentation_RefineNet_efficientnet-b3_384x576_tta", "segmentation_RefineNetPoolingImprove_densenet121_384x576_tta","segmentation_FastFCN_densenet121_384x576_tta", "segmentation_FastFCNImprove_densenet121_384x576_tta", "segmentation_DeepLab_resnet50_384x576_tta", "segmentation_FPN_efficientnet-b1_384x576_tta", "segmentation_FastFCN_efficientnet-b3_384x576_tta", "segmentation_FastFCNImprove_efficientnet-b3_384x576_tta", "segmentation_FPN_efficientnet-b3_small_256x384_tta", "segmentation_FPN_efficientnet-b3_big_512x768_tta", "segmentation_LinkNet_densenet121_small_256x384_tta", "segmentation_LinkNet_densenet121_big_256x768_tta", "segmentation_LinkNet_efficientnet-b3_small_256x384_tta", "segmentation_LinkNet_efficientnet-b3_big_512x768_tta"]
#         self.ensemble_filestems = ["segmentation_UNet_densenet121_aspp_384x576_tta", "segmentation_Linknet_densenet121_384x576_tta", "segmentation_Linknet_efficientnet-b3_384x576_tta", "segmentation_UNet_densenet121_fpa_256x512_tta", "segmentation_UNet_efficientnet-b3_fpa_384x576_tta", "segmentation_FPN_efficientnet-b3_384x576_tta", "segmentation_FPN_densenet121_384x576_tta", "segmentation_RefineNet_densenet121_384x576_tta", "segmentation_RefineNet_efficientnet-b3_384x576_tta", "segmentation_RefineNetPoolingImprove_densenet121_384x576_tta","segmentation_FastFCN_densenet121_384x576_tta", "segmentation_FastFCNImprove_densenet121_384x576_tta", "segmentation_DeepLab_resnet50_384x576_tta", "segmentation_FPN_efficientnet-b1_384x576_tta", "segmentation_FastFCN_efficientnet-b3_384x576_tta", "segmentation_FastFCNImprove_efficientnet-b3_384x576_tta", "segmentation_FPN_efficientnet-b3_small_256x384_tta", "segmentation_FPN_efficientnet-b3_big_512x768_tta", "segmentation_LinkNet_densenet121_small_256x384_tta", "segmentation_LinkNet_densenet121_big_256x768_tta", "segmentation_LinkNet_efficientnet-b3_small_256x384_tta", "segmentation_LinkNet_efficientnet-b3_big_512x768_tta"]
        self.temp = 0
        self.ensemble_weight = None
        self.class_num = 4
        self.convex_mode = None