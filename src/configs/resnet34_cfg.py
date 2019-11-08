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
        self.center = 'fpa'
        self.encoder_weights = "imagenet"
        self.fold_max = 4

        self.per_image_norm = False 
        self.optimizer = "RAdam"
        self.backborn = "resnet34"
        self.attention_type = "scse"
#         self.attention_type = None
        self.logdir = "../logs/"+self.task+"_"+self.model_type+"_"+self.backborn+"_"+ self.center
        
        self.img_size = (384, 768)
#         self.img_size = (480, 480)
        
        
        self.batchsize = 32
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
        self.accumeration = 2
#         self.tta = False
        self.tta = True
        self.convex_mode = None
#         self.convex_mode = "convex" # 0.659
        