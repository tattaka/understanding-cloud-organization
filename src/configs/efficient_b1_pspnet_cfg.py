import sys
from functools import partial

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import segmentation_models_pytorch as smp

from losses import * 

class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.task = "segmentation"
        self.model_type = "PSPNet"
        self.fold_max = 5

        self.per_image_norm = False 
        self.optimizer = "RAdam"
        # b1-0.658, b2-0.651 b3_fold10-0.657 b4_fold5-0.655 b6_fold5-0.656
        self.backborn = "efficientnet-b1"
        self.attention_type = None
        self.encoder_weights = "imagenet"
        self.center = None
        self.logdir = "../logs/"+self.task+"_"+self.model_type+"_"+self.backborn 
        
        self.img_size = (384, 576)
        
        
        self.batchsize = 32
        self.class_num = 4
        
        self.num_workers = 8
        self.max_epoch = 50
        
        self.optimizer = "RAdam"
        self.lr = 1e-2 # b2 NaN
        self.lr_e = 1e-3
        self.lookahead = False
        
        self.early_stop = True
#         self.scheduler = partial(ReduceLROnPlateau, factor=0.5, patience=2)
        self.scheduler = partial(ReduceLROnPlateau, factor=0.1, patience=2)
        self.criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        self.accumeration = 5

        self.tta = True 
        self.convex_mode = None 
#         self.convex_mode = "convex" 
        