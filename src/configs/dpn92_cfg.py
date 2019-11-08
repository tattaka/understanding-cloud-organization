import sys
from functools import partial

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import segmentation_models_pytorch as smp
sys.path.append('../')
import utils
# CV:0.665, LB:0.656
class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.task = "segmentation"
        self.model_type = "Unet"
        self.fold_max = 5

        self.per_image_norm = False 
        self.optimizer = "RAdam"
        self.backborn = "dpn92"
        self.attention_type = "scse"
        self.encoder_weights = "imagenet+5k"
        self.center = None
        self.logdir = "../logs/"+self.task+"_"+self.model_type+"_"+self.backborn
        self.img_size = (384, 576)
        
        
        self.batchsize = 15
        self.class_num = 4
        
        self.num_workers = 8
        self.max_epoch = 40
        
        self.optimizer = "RAdam"
        self.lr = 1e-2
        self.lr_e = 1e-3
        self.lookahead = False
        
        self.early_stop = True
        self.scheduler = partial(ReduceLROnPlateau, factor=0.1, patience=2)
        self.criterion = utils.losses.BCEDiceLoss(dice_bce_ratio=(0.7, 0.3))
        self.accumeration = 4
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