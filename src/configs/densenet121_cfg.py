import sys
from functools import partial

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import segmentation_models_pytorch as smp

from losses import * 

class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.task = "segmentation"
        self.model_type = "Unet"
        self.fold_max = 4
        self.backborn = "densenet121"
        self.logdir = "../logs/"+self.task+"_"+self.model_type+"_"+self.backborn
        print("logdir: ", self.logdir)
        self.attention_type = "scse"
        
        self.img_size = (320, 640)
        
        
        self.batchsize = 32
        self.class_num = 4
        
        self.num_workers = 8
        self.max_epoch = 50
        
#         self.scheduler = partial(ReduceLROnPlateau, factor=0.5, patience=2)
        self.early_stop = True
        self.scheduler = partial(ReduceLROnPlateau, factor=0.1, patience=2)
        self.criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        self.accumeration = 2
        self.convex_mode = None # 0.659
        self.tta = False
#         self.tta = True # 0.652
#         self.convex_mode = "convex" # 0.659
        