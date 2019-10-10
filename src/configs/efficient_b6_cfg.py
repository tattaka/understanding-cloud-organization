import sys
from functools import partial

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import segmentation_models_pytorch as smp

from losses import * 

class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.task = "segmentation"
#         self.model_type = "HyperColumns"
        self.model_type = "Unet"
        self.logdir = "../logs/segmentation_Unet_efficientb6"
        self.fold_max = 4
#         self.backborn = "resnet18" # 0.650
#         self.backborn = "efficientnet-b4" # 0.652
        self.backborn = "efficientnet-b6"
        self.attention_type = "scse"
        
        self.img_size = (320, 640)
        
        
        self.batchsize = 16
        self.class_num = 4
        
        self.num_workers = 8
        self.max_epoch = 50
        
#         self.scheduler = partial(ReduceLROnPlateau, factor=0.5, patience=2)
        self.scheduler = partial(ReduceLROnPlateau, factor=0.1, patience=2)
        self.criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        self.accumeration = 4
        self.convex_mode = None # 0.659
        self.tta = False
#         self.tta = True # 0.652
#         self.convex_mode = "convex" # 0.659
        