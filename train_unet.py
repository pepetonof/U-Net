# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 01:29:28 2021

@author: josef
"""

from data import get_data
from utils import (get_loaders, save_model, check_dice, 
                   save_predictions_as_imgs2, saveTrainDetails, 
                   save_graphtv)
from train_model import trainAndvalid
from model import UNET

import torch
import os
from torchsummary import summary
import albumentations as A
from albumentations.augmentations.transforms import ElasticTransform

import torch.optim as optim
from torchgeometry.losses import dice_loss
import torch.nn as nn

import numpy as np

from loss_f import combo_loss

"""Get data for train and validation, data of filenames (list)"""
def loaders(size_train, height, width, NUM_WORKERS):
    TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR = get_data(size_train)
    BATCH_SIZE = 1
    IMAGE_HEIGHT = height #384  # 768 originally
    IMAGE_WIDTH = width #512   # 1024 originally
    PIN_MEMORY = True
    train_transform = A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    A.HorizontalFlip(p=0.5),
                    ElasticTransform(alpha=1, sigma=10, alpha_affine=20, interpolation=2, 
                                     border_mode= 0, approximate=True, p=0.8),
                    A.Normalize(
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0,),
                ],)
    val_transform = A.Compose(
                    [
                        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                        A.Normalize(
                            mean=[0.0, 0.0, 0.0],
                            std=[1.0, 1.0, 1.0],
                            max_pixel_value=255.0,),
                    ],)
    train_loader, val_loader =get_loaders(
              TRAIN_IMG_DIR,
              TRAIN_MASK_DIR,
              VAL_IMG_DIR,
              VAL_MASK_DIR,
              BATCH_SIZE,
              train_transform,
              val_transform,
              NUM_WORKERS,
              PIN_MEMORY,
              )
    return train_loader, val_loader

"""Create folder to storage"""
corrida='unet_ce2_100'
path=os.getcwd()
# path='/content/gdrive/MyDrive/unet'
ruta=path+"/corrida"+str(corrida)
if not os.path.exists(ruta):
    os.makedirs(ruta)
    
NUM_WORKERS = 0 if torch.cuda.is_available() else 0 #Also used for dataloaders
#EPOCHS = 0 if torch.cuda.is_available() else 0
IMAGE_HEIGHT = 384 
IMAGE_WIDTH = 512
"""Get train and valid loader"""
train_loader, val_loader = loaders(size_train=0.7, height=IMAGE_HEIGHT, 
                                   width=IMAGE_WIDTH, NUM_WORKERS=NUM_WORKERS)

"""Create UNET model"""
model = UNET(in_channels=3, out_channels=2)

"""Hyperparameters for train"""
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 100
LOAD_MODEL = False

"""Optimizer"""
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
"""Loss function"""
LOSS_FN = nn.CrossEntropyLoss()#dice_loss
# LOSS_FN = combo_loss

"""Train Model"""
dices, train_loss, valid_loss = trainAndvalid(
    model,
    train_loader,
    val_loader,
    NUM_EPOCHS,
    
    LOSS_FN,
    LEARNING_RATE,
    optimizer,
    
    DEVICE,
    NUM_WORKERS,
    LOAD_MODEL,
    ruta, #Ruta donde se encuentra localizado el modelo si se desea cargar
    )

print('\n Dice Index =', np.mean(dices))

"""Save train and valid losss"""
save_graphtv(train_loss, valid_loss, ruta, 'TrainValidationLoss')

"""Save model"""
save_model(model, optimizer, ruta)

"""Determina número de parámetros"""
model = model.to(DEVICE)
model_stats = summary(model, (3, 384, 512), verbose=0)
summary_model = str(model_stats)

"""Save train details"""
saveTrainDetails(
    len(train_loader), len(val_loader),
    LEARNING_RATE, NUM_EPOCHS,
    IMAGE_HEIGHT, IMAGE_WIDTH,
    LOSS_FN, optimizer,
    np.mean(dices), np.std(dices),
    np.max(dices), np.min(dices),
    summary_model,
    ruta+'/UNET_Train_details.txt'
    )

"""Free GPU memory"""
# print(torch.cuda.memory_allocated())
# del model, optimizer, model_stats
# torch.cuda.empty_cache()
# print(torch.cuda.memory_allocated())