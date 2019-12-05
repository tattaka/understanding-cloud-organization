import sys
from functools import partial

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import segmentation_models_pytorch as smp
sys.path.append('../')
import utils

class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.ensemble_filestems = ["segmentation_FPN_efficientnet-b3_384x576_tta", "segmentation_FPN_efficientnet-b3_big_512x768_tta", "segmentation_FPN_efficientnet-b3_small_256x384_tta", "segmentation_LinkNet_densenet121_384x576_tta", "segmentation_LinkNet_densenet121_big_256x768_tta", "segmentation_LinkNet_densenet121_small_256x384_tta"]
        self.temp = 0
        self.ensemble_weight = None
        self.class_num = 4
        self.convex_mode = None