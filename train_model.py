# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 21:02:45 2021

@author: josef
"""
import torch
from tqdm import tqdm
from utils import (
    check_accuracy2,
    save_predictions_as_imgs3,
    save_model,
    load_model,
    )
import numpy as np

def train_fn(loader, model, optimizer, loss_fn, scaler, DEVICE):
    loop = tqdm(loader)
    running_loss=0.0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        #targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.detach().item())
        
        running_loss+=loss*loader.batch_size
        
        del data, targets, predictions
        torch.cuda.empty_cache()
        
    loss_out=running_loss/len(loader)
    
    return loss_out.detach().item() #loss.detach().item()#loss.item()

def trainAndvalid(
          model,
          train_loader,
          val_loader,
          NUM_EPOCHS,
          
          loss_fn,
          LEARNING_RATE,
          optimizer,
          
          DEVICE,
          NUM_WORKERS,
          LOAD_MODEL,
          ruta='results_exp/',
          ):
    
    """Set model into device"""    
    model = model.to(DEVICE)
    
    """Load model"""    
    if LOAD_MODEL:
        #load_checkpoint(torch.load(ruta+"/my_checkpoint.pth.tar"), model)
        load_model(model, ruta, 'my_checkpoint.pth.tar')
    
    """For monitoring and graph train and valid loss"""
    train_loss, valid_loss = [], []
    dice, _ = check_accuracy2(val_loader, model, loss_fn, device=DEVICE)
    # #dices, _ = check_dice(val_loader, model, loss_fn,
    #                      save=False, folder=ruta, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        print('Epoch:\t', epoch)
        losst = train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
        train_loss.append(losst) #For plot trainning loss
        
        #Save model
        save_model(model, optimizer, ruta)
        #Check Accuracy
        dices, lossv = check_accuracy2(val_loader, model, loss_fn, device=DEVICE)
        # dices, lossv = check_dice(val_loader, model, loss_fn,
        #                          save=False, folder=ruta, device=DEVICE)
        valid_loss.append(lossv) #For plot valid loss
        #Print examples to a folder
        save_predictions_as_imgs3(val_loader, model, folder=ruta, device=DEVICE)
    
    return dices, train_loss, valid_loss