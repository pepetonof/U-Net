# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:09:39 2022

@author: josef
"""
import os
import torch
# import torch.nn as nn
import numpy as np

import torch.optim as optim
from torchinfo import summary


from models import UNET, UNetPlusPlus, AttentionUNet, UNet_3Plus
from model.loss_f import ComboLoss
from model.train_valid import train_and_validate
from model.predict import test
from model.utils import save_graphtv, save_graphtvd

from data.loader import loaders
import data.dataSplit as dataSplit
import data.dataStatic as dataStatic

from utils.save_utils import saveTrainDetails, save_execution
from utils.visualization_utils import visualize_augmentations

class model_out:
    def __init__(self, model, 
                 # dice, params, 
                 # train_loss, valid_loss,
                 # train_dice, valid_dice,
                 # dices, ious, hds,
                 ):
        self.model=model
        # self.dice=dice
        # self.params=params
        # self.train_loss=train_loss
        # self.valid_loss=valid_loss
        # self.train_dice=train_dice
        # self.valid_dice=valid_dice,
        # self.dices=dices
        # self.ious=ious
        # self.hds=hds

#%%Folder to storage
"""Create folder to storage"""
foldername=''
path='/scratch/202201016n'
ruta=path+"/corridas/"+str(foldername)
if not os.path.exists(ruta):
    os.makedirs(ruta)
    
#%% Data
# path_images='C://Users//josef//OneDrive - Universidad Veracruzana//Maestria CIIA//3_Semestre//Temas Selectos CE//Implementation//deap//neuroevolution_deap//images'
# path_images='/content/gdrive/MyDrive/images'
path_images = '/home/202201016n/serverBUAP/images'
in_channels=1

"""Get train, valid, test set and loaders"""
train_set, valid_set, test_set = dataSplit.get_data(0.7, 0.15, 0.15, path_images)
# train_set, valid_set, test_set = dataStatic.get_data(path_images)

IMAGE_HEIGHT = 256#288 
IMAGE_WIDTH = 256#480
NUM_WORKERS = 0 if torch.cuda.is_available() else 0 #Also used for dataloaders

dloaders = loaders(train_set, valid_set, test_set, batch_size=1, 
                 image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, in_channels=in_channels,
                 num_workers=NUM_WORKERS)

#%%Muestra ejemplos de datos aumentado
_, train_ds=loaders.get_train_loader()
_, val_ds = loaders.get_val_loader()
_, test_ds = loaders.get_test_loader()
# for idx in range(len(test_ds)):
visualize_augmentations(test_ds, idx=0, samples=3)
    
#%% Create Model
model = UNET(in_channels=3, out_channels=2)
# model = UNetPlusPlus(in_ch=3, out_ch=2)
# model = AttentionUNet(in_ch=3, out_ch=2)
# model = UNet_3Plus(in_channels=3, n_classes=2)

#%%Train, validate and test
"""Train, val and test loaders"""
train_loader, _= dloaders.get_train_loader()
val_loader, _  = dloaders.get_val_loader()
test_loader, _ = dloaders.get_test_loader()

"""Optimizer"""
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)

"""Epocas de re-entrenamiento"""
num_epochs=1

"""Loss function"""
alpha=0.5
beta=0.4
loss_fn = ComboLoss(alpha=alpha, beta=beta)

"""Device"""
device='cuda:2'

"""For save images or model"""
save_model_flag=True
save_images_flag=True
load_model=False

#%%Train, validate and test
"""Train and valid model"""
train_loss, valid_loss, train_dice, valid_dice = train_and_validate(
    model, train_loader, val_loader,
    num_epochs, optimizer, loss_fn,
    device, load_model, save_model_flag,
    ruta=ruta, verbose=True
    )

"""Test model"""
dices, ious, hds = test(test_loader, model, loss_fn,
                        save_imgs=save_images_flag, ruta=ruta, device=device, verbose=True)

#%%Attributes assigned
best_model=model_out(model)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)

best_model.dice = np.mean(dices)
best_model.params = params

best_model.train_loss = train_loss
best_model.valid_loss = valid_loss
best_model.train_dice = train_dice
best_model.valid_dice = valid_dice

best_model.dices = dices
best_model.ious = ious
best_model.hds = hds

#%%Dice and Loss Graphs of training
save_graphtv(train_loss, valid_loss, ruta=ruta, filename='ReTrainValidationLoss')
save_graphtvd(train_dice, valid_dice, ruta=ruta, filename='ReTrainValidationDice')

#%%No. parameters
"""No of parameters"""
model = model.to(device)
model_stats=summary(model, (1, in_channels, 256, 256), verbose=0)
summary_model = str(model_stats)

#%%Retrain Details
"""Save train details"""
saveTrainDetails(
    len(train_loader), len(val_loader),
    lr, num_epochs,
    IMAGE_HEIGHT, IMAGE_WIDTH,
    loss_fn, optimizer,
    dices, ious, hds,
    summary_model,
    ruta+'/Retrain_best_details.txt'
    )

#%%Save execution
save_execution(ruta, foldername+'.pkl', [], 0, best_model)

del model