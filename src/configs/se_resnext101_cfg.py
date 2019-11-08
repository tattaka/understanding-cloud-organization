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

        self.per_image_norm = False #0.659
        self.optimizer = "RAdam"
        self.backborn = "se_resnext101_32x4d"
        self.encoder_weights = "imagenet"
        self.attention_type = "scse"
#         self.attention_type = None
        self.logdir = "../logs/"+self.task+"_"+self.model_type+"_"+self.backborn
        
        self.img_size = (320, 640)
#         self.img_size = (480, 480)
        
        
        self.batchsize = 16
        self.class_num = 4
        
        self.num_workers = 8
        self.max_epoch = 50
        
        self.optimizer = "RAdam"
        self.lr = 1e-2
        self.lr_e = 1e-3
        self.lookahead = False
        
        self.early_stop = True
#         self.scheduler = partial(ReduceLROnPlateau, factor=0.5, patience=2)
        self.scheduler = partial(ReduceLROnPlateau, factor=0.1, patience=2)
        self.criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        self.accumeration = 4
#         self.tta = False
        self.tta = True # 0.659
        self.convex_mode = None # 0.659
#         self.convex_mode = "convex" # 0.659
        