import os
import cv2
import collections
import time 
import tqdm
from PIL import Image
from functools import partial
import argparse
from importlib import import_module
train_on_gpu = True
import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu
from albumentations import pytorch as AT

from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, CriterionCallback, OptimizerCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback

import segmentation_models_pytorch as smp
import apex
from apex import amp


from utils import * 
from losses import * 
from dataset import * 
from optimizers import * 
from models import * 
import encoders
import unet.model

def stratified_groups_kfold(df, target, n_splits=5, random_state=0):
    all_groups = pd.Series(df[target])
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idx_tr, idx_val in folds.split(all_groups, all_groups):
        idx_tr_new = df.iloc[idx_tr]
        idx_val_new = df.iloc[idx_val]
        print(len(idx_tr_new),  len(idx_val_new))
        yield idx_tr_new, idx_val_new
    
def main(config):
    opts = config()
    path = opts.path
    train = pd.read_csv(f'{path}/train.csv')
    
    n_train = len(os.listdir(f'{path}/train_images'))
    n_test = len(os.listdir(f'{path}/test_images'))
    print(f'There are {n_train} images in train dataset')
    print(f'There are {n_test} images in test dataset')
    
    train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()
    train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts()
    
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})

    if not os.path.exists("csvs/train_all.csv"):
        train_ids, valid_ids = train_test_split(id_mask_count, random_state=39, stratify=id_mask_count['count'], test_size=0.1)
        valid_ids.to_csv("csvs/valid_threshold.csv", index=False)
        train_ids.to_csv("csvs/train_all.csv", index=False)
    else:
        train_ids = pd.read_csv("csvs/train_all.csv")
        valid_ids = pd.read_csv("csvs/valid_threshold.csv")
        
    for fold, (train_ids_new, valid_ids_new) in enumerate(stratified_groups_kfold(train_ids, target='count', n_splits=opts.fold_max, random_state=0)):
        train_ids_new.to_csv(f'csvs/train_fold{fold}.csv')
        valid_ids_new.to_csv(f'csvs/valid_fold{fold}.csv')
        train_ids_new = train_ids_new['img_id'].values
        valid_ids_new = valid_ids_new['img_id'].values
        
        ENCODER = opts.backborn
        ENCODER_WEIGHTS = 'imagenet'
        DEVICE = 'cuda'

        ACTIVATION = None
        model = get_model(model_type=opts.model_type,
              encoder = ENCODER,
              encoder_weights = ENCODER_WEIGHTS,
              activation = ACTIVATION,
              n_classes = opts.class_num,
              task = opts.task,
              attention_type = opts.attention_type, 
              head = 'simple'
         )

        preprocessing_fn = encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        
        num_workers = opts.num_workers
        bs = opts.batchsize
        train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids_new, transforms = get_training_augmentation(opts.img_size), preprocessing=get_preprocessing(preprocessing_fn))
        valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids_new, transforms = get_validation_augmentation(opts.img_size), preprocessing=get_preprocessing(preprocessing_fn))

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

        loaders = {
            "train": train_loader,
            "valid": valid_loader
        }
        num_epochs = opts.max_epoch
        logdir = f"{opts.logdir}/fold{fold}" 
        # TODO: get_optimizer()を作ってconfigでmodel管理する
        optimizer = RAdam([
            {'params': model.decoder.parameters(), 'lr': 1e-2}, 
            {'params': model.encoder.parameters(), 'lr': 1e-3},  
        ])
        opt_level = 'O1'
        model.cuda()
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        scheduler = opts.scheduler(optimizer)
#         scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        criterion = opts.criterion
        runner = SupervisedRunner()
        callbacks = [DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)]
        if opts.accumeration is not None:
            callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001), CriterionCallback(), OptimizerCallback(accumulation_steps=opts.accumeration)]
        print(f"############################## Start learning of fold{fold}! ##############################")
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=callbacks,
            logdir=logdir,
            num_epochs=num_epochs,
            verbose=True
        )
        del model
        del loaders
        del runner
        gc.collect()
        print(f"############################## Finish learning of fold{fold}! ##############################")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='default_cfg')
    args = parser.parse_args()
    config = import_module("configs."+ args.config)
    main(config.Config)
        
    
    
    
    