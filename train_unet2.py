# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:09:39 2022

@author: josef
"""
import os
import torch
import torch.nn as nn
from loader import loaders
from visualization_utils import visualize_augmentations
from utils import save_graphtv, save_graphtvd
from model import UNET
import torch.optim as optim
from loss_f import combo_loss
from train_valid import train_and_validate
from predict import test
from torchinfo import summary
from utils import saveTrainDetails, save_model

#%%Directorio de imágenes y loaders
# path_images='C://Users//josef//OneDrive - Universidad Veracruzana//Maestria CIIA//3_Semestre//Temas Selectos CE//Implementation//deap//neuroevolution_deap//images'
# path_images='/content/gdrive/MyDrive/images'
path_images = '/home/202201016n/serverBUAP/images' 

NUM_WORKERS = 0 if torch.cuda.is_available() else 0 #Also used for dataloaders
IMAGE_HEIGHT = 288
IMAGE_WIDTH = 480
"""Get train and valid loader"""
loaders=loaders(0.7, 0.15, 0.15, path_images=path_images, batch_size=1, num_workers=NUM_WORKERS)

#%%Muestra ejemplos de datos aumentado
# _, train_ds=loaders.get_train_loader(288,480)
# _, val_ds = loaders.get_val_loader(288,480)
# _, test_ds = loaders.get_test_loader(288,480)
# # for idx in range(len(test_ds)):
# visualize_augmentations(test_ds, idx=0, samples=3)
    

#%%Folder to storage
"""Create folder to storage"""
corrida='unet'
# path=os.getcwd()
path='/scratch/202201016n/'
ruta=path+"/"+str(corrida)
if not os.path.exists(ruta):
    os.makedirs(ruta)
    
#%% Create U-Net Model
model = UNET(in_channels=3, out_channels=2)

#%%Hyperparameters for train and validate model
"""Hyperparameters for train"""
EPOCHS = 10 if torch.cuda.is_available() else 0
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False

"""Optimizer"""
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
"""Loss function"""
LOSS_FN = combo_loss#nn.CrossEntropyLoss()#combo_loss
"""For save images or model"""
save_model_flag=False
save_images_flag=True

"""Train, val and test loaders"""
train_loader, _= loaders.get_train_loader(IMAGE_HEIGHT, IMAGE_WIDTH)
val_loader, _  =loaders.get_val_loader(IMAGE_HEIGHT, IMAGE_WIDTH)
test_loader, _ =loaders.get_test_loader(IMAGE_HEIGHT, IMAGE_WIDTH)

#%%Train and validate model
"""Train and valid model"""
train_loss, valid_loss, train_dice, valid_dice = train_and_validate(
    model, train_loader, val_loader,
    EPOCHS, optimizer, LOSS_FN,
    DEVICE, LOAD_MODEL, save_model_flag,
    ruta=ruta
    )

#%%Test model
dices, ious, hds = test(test_loader, model, LOSS_FN,
                        save_imgs=save_images_flag, ruta=ruta, device=DEVICE)

#%%Save model
save_model(model, optimizer, ruta)

#%%Train and validation loss
save_graphtv(train_loss, valid_loss, ruta, 'TrainValidationLoss')
save_graphtvd(train_dice, valid_dice, ruta, 'TrainValidationDice')


#%%Número de parámetros
model = model.to(DEVICE)
model_stats=summary(model, (1, 3, 288, 480), verbose=0)
summary_model = str(model_stats)

"""Save train details"""
saveTrainDetails(
    len(train_loader), len(val_loader),
    LEARNING_RATE, EPOCHS,
    IMAGE_HEIGHT, IMAGE_WIDTH,
    LOSS_FN, optimizer,
    dices, ious, hds,
    summary_model,
    ruta+'/Retrain_best_details.txt'
    )