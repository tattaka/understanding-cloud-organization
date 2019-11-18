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


from cloud_utils import * 
from dataset import * 
from optimizers import * 
from models import * 
from sync_batchnorm import convert_model
import encoders


def main(config):
    opts = config()
    path = opts.path
    train = pd.read_csv(f'{path}/train.csv')
    sub = pd.read_csv(f'{path}/sample_submission.csv')
    
    n_train = len(os.listdir(f'{path}/train_images'))
    n_test = len(os.listdir(f'{path}/test_images'))
    
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
    train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()
    train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts()
    
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    
    valid_ids = pd.read_csv("csvs/valid_threshold.csv")["img_id"].values
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values
#     print(valid_ids)
    ENCODER = opts.backborn
    ENCODER_WEIGHTS = opts.encoder_weights
    DEVICE = 'cuda'

    ACTIVATION = None
    model = get_model(model_type=opts.model_type,
              encoder = ENCODER,
              encoder_weights = ENCODER_WEIGHTS,
              activation = ACTIVATION,
              n_classes = opts.class_num,
              task = opts.task,
              attention_type = opts.attention_type, 
              head = 'simple',
              center = opts.center,
              tta = opts.tta
     )
    if opts.refine:
        model = get_ref_model(infer_model = model,
                  encoder = opts.ref_backborn,
                  encoder_weights = ENCODER_WEIGHTS,
                  activation = ACTIVATION,
                  n_classes = opts.class_num,
                  preprocess = opts.preprocess,
                  tta = opts.tta
         )
    model = convert_model(model)
    preprocessing_fn = encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    encoded_pixels = []
    runner = SupervisedRunner()
    probabilities = np.zeros((2220, 350, 525))
    
    for i in range(opts.fold_max):
        if opts.refine:
            logdir = f"{opts.logdir}_refine/fold{i}" 
        else:
            logdir = f"{opts.logdir}/fold{i}" 
        valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms = get_validation_augmentation(opts.img_size), preprocessing=get_preprocessing(preprocessing_fn))
        valid_loader = DataLoader(valid_dataset, batch_size=opts.batchsize, shuffle=False, num_workers=opts.num_workers)
        loaders = {"infer": valid_loader}
        runner.infer(
            model=model,
            loaders=loaders,
            callbacks=[
                CheckpointCallback(
                    resume=f"{logdir}/checkpoints/best.pth"),
                InferCallback()
            ],
        )
        valid_masks = []
        for i, (batch, output) in enumerate(tqdm.tqdm(zip(
                valid_dataset, runner.callbacks[0].predictions["logits"]))):
            image, mask = batch
            for m in mask:
                if m.shape != (350, 525):
                    m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                valid_masks.append(m)

            for j, probability in enumerate(output):
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                probabilities[i * 4 + j, :, :] += probability
                
    probabilities /= opts.fold_max
    if opts.tta:
        np.save(f'probabilities/{opts.logdir.split("/")[-1]}_{opts.img_size[0]}x{opts.img_size[1]}_tta_valid.npy', probabilities)
    else:
        np.save(f'probabilities/{opts.logdir.split("/")[-1]}_{opts.img_size[0]}x{opts.img_size[1]}_valid.npy', probabilities)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    class_params = {}
    cv_d = []
    for class_id in tqdm.trange(opts.class_num, desc='class_id', leave=False):
#         print(class_id)
        attempts = []
        for t in tqdm.trange(0, 100, 10, desc='threshold', leave=False):
            t /= 100
            for ms in tqdm.tqdm([0, 100, 1000, 5000, 10000, 11000, 14000, 15000, 16000, 18000, 19000, 20000, 21000, 23000, 25000, 27000, 30000, 50000], desc='min_size', leave=False):
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(sigmoid(probability), t, ms, convex_mode=opts.convex_mode) 

                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
#                     print(i.shape, j.shape)
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))
                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
        

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        cv_d.append(attempts_df['dice'].values[0])
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]

        class_params[class_id] = (best_threshold, best_size)
    cv_d = np.array(cv_d)
    print("CV Dice:", np.mean(cv_d))
    pathlist = ["../input/test_images/" + i.split("_")[0] for i in sub['Image_Label']]
    
    del masks
    del valid_masks
    del probabilities
    gc.collect()
    
    ############# predict ###################
    probabilities = np.zeros((n_test, 4, 350, 525))
    for fold in tqdm.trange(opts.fold_max, desc='fold loop'):
        if opts.refine:
            logdir = f"{opts.logdir}_refine/fold{fold}" 
        else:
            logdir = f"{opts.logdir}/fold{fold}"
#         loaders = {"test": test_loader}
        test_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids, transforms = get_validation_augmentation(opts.img_size), preprocessing=get_preprocessing(preprocessing_fn))
        test_loader = DataLoader(test_dataset, batch_size=opts.batchsize, shuffle=False, num_workers=opts.num_workers)
        runner_out = runner.predict_loader(model, test_loader, resume=f"{logdir}/checkpoints/best.pth", verbose=True)
        for i, batch in enumerate(tqdm.tqdm(runner_out, desc='probability loop')):
            for j, probability in enumerate(batch):
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                probabilities[i, j, :, :] += probability
        gc.collect()
    probabilities /= opts.fold_max
    if opts.tta:
        np.save(f'probabilities/{opts.logdir.split("/")[-1]}_{opts.img_size[0]}x{opts.img_size[1]}_tta_test.npy', probabilities)
    else:
        np.save(f'probabilities/{opts.logdir.split("/")[-1]}_{opts.img_size[0]}x{opts.img_size[1]}_test.npy', probabilities)
    image_id = 0
    print("##################### start post_process #####################")
    for i in tqdm.trange(n_test, desc='post porocess loop'):
        for probability in probabilities[i]:
            predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1], convex_mode=opts.convex_mode)
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                black_mask = get_black_mask(pathlist[image_id])
                predict = np.multiply(predict, black_mask)
                r = mask2rle(predict)
                encoded_pixels.append(r)
            image_id += 1
        gc.collect()
    print("##################### Finish post_process #####################")
    #######################################
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(f'submissions/submission_{opts.logdir.split("/")[-1]}_{opts.img_size[0]}x{opts.img_size[1]}.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='default_cfg')
    args = parser.parse_args()
    config = import_module("configs."+ args.config)
    main(config.Config)