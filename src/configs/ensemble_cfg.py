import sys
from functools import partial

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import segmentation_models_pytorch as smp
sys.path.append('../')
import utils

class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.ensemble_filestems = ["segmentation_Unet_densenet121_aspp_tta", "segmentation_Linknet_densenet121_tta", "segmentation_Linknet_efficientnet-b3_tta", "segmentation_Unet_densenet121_fpa_tta", "segmentation_FPN_efficientnet-b3_tta"]
#         self.unsemble_filestems = ["segmentation_Unet_densenet121_aspp_tta"]
        self.temp = 0.5
        self.ensemble_weight = None
        self.class_num = 4
        self.convex_mode = None