import sys
from functools import partial

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import segmentation_models_pytorch as smp
sys.path.append('../')
import utils

class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.task = "segmentation"
        self.model_type = "Unet"
        self.fold_max = 5

        self.per_image_norm = False 
        self.optimizer = "RAdam"
        self.backborn = "efficientnet-b1"
        self.attention_type = "cbam"
        self.encoder_weights = "imagenet"
        self.center = "aspp"
        self.logdir = "../logs/"+self.task+"_"+self.model_type+"_"+self.backborn+"_"+self.center
        
        self.img_size = (384, 576)
        
        
        self.batchsize = 15
        self.class_num = 4
        
        self.num_workers = 8
        self.max_epoch = 40
        
        self.optimizer = "RAdam"
        self.lr = 1e-2 # b2 NaN
        self.lr_e = 1e-3
        self.lookahead = False
        
        self.early_stop =True
        
        self.scheduler = partial(ReduceLROnPlateau, factor=0.1, patience=2)
#         self.scheduler = partial(CosineAnnealingLR, T_max=self.max_epoch, eta_min=1e-6)

        self.criterion = utils.losses.BCEDiceLoss(dice_bce_ratio=(0.7, 0.3))
        self.accumeration = 4
        
        self.tta = True 
        self.convex_mode = None 
        
        self.mixup = False 
        self.label_smoothing_eps = 0 
        
        self.classification = False
        self.refine = False
        self.ref_backborn = "efficientnet-b3"
        self.ref_attention_type = "cbam"
        self.ref_center = "aspp"
        self.preprocess = False
        self.ref_lr = 1e-2
        self.ref_lr_e = 1e-3
        self.ref_criterion = utils.losses.BCEDiceLoss(dice_bce_ratio=(0.7, 0.3)) 
        
        self.ref_max_epoch = 20
        