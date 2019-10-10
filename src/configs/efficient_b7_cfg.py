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
        self.logdir = "../logs/segmentation_Unet_efficientb7"
        self.fold_max = 4
        self.backborn = "efficientnet-b7"
        self.attention_type = "scse"
        
        self.img_size = (320, 640)
        
        
        self.batchsize = 8
        self.class_num = 4
        
        self.num_workers = 8
        self.max_epoch = 50
        
#         self.scheduler = partial(ReduceLROnPlateau, factor=0.5, patience=2)
        self.scheduler = partial(ReduceLROnPlateau, factor=0.1, patience=2)
        self.criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        self.accumeration = 8
        self.convex_mode = None
        
#         self.convex_mode = "convex"
        