import sys
from functools import partial

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import segmentation_models_pytorch as smp
sys.path.append('../')
import utils
# CV:0.6669, LB:
class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.task = "segmentation"
        self.model_type = "RefineNet"
        self.fold_max = 5

        self.per_image_norm = False 
        self.optimizer = "RAdam"
        self.backborn = "densenet121"
        self.attention_type = None
        self.encoder_weights = "imagenet"
        self.center = None
        self.logdir = "../logs/"+self.task+"_"+self.model_type+"_"+self.backborn + "_classification"
        self.img_size = (384, 576)
        
        self.resume = None
        
        self.batchsize = 12
        self.class_num = 4
        
        self.num_workers = 8
        self.max_epoch = 40
        
        self.optimizer = "RAdam"
        self.lr = 5e-4
        self.lr_e = 1e-4
        self.lookahead = False
        
        self.classification = True
        
        self.early_stop = False
        self.scheduler = partial(ReduceLROnPlateau, factor=0.1, patience=2)
        self.criterion = utils.losses.BCEDiceLoss(dice_bce_ratio=(0.7, 0.3), classification=self.classification)
        self.accumeration = 5
#         self.tta = False
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