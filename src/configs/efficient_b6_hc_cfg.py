import sys
from functools import partial

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import segmentation_models_pytorch as smp

from losses import * 

class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.task = "segmentation"
        self.model_type = "HyperColumns"
        self.fold_max = 4
        self.backborn = "efficientnet-b6"
        self.attention_type = "scse"
        self.logdir = "../logs/"+self.task+"_"+self.model_type+"_"+self.backborn
        
        self.img_size = (320, 640)
        
        
        self.batchsize = 12
        self.class_num = 4
        
        self.num_workers = 8
        self.max_epoch = 50
        
        self.scheduler = partial(ReduceLROnPlateau, factor=0.1, patience=2)
        self.criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        self.accumeration = 8
        self.convex_mode = None
        self.tta = False # 0.654
#         self.tta = True # 0.653
#         self.convex_mode = "convex"
        