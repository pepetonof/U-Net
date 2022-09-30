import torch
import torchvision
from dataset import PlacentaDataSetMC
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import draw_segmentation_masks
#from torchgeometry.losses import dice_loss

from PIL import Image, ImageFont, ImageDraw
import numpy as np

import matplotlib.pyplot as plt

from loss_f import hdistance_loss, IoU_loss, dice_loss

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    #print("=> Saving checkpoint")
    torch.save(state, filename)
    return

def load_checkpoint(checkpoint, model):
    #print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return

def save_model(model, optimizer, ruta):
    print('=> Saving model...\t')
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        }
    save_checkpoint(checkpoint, filename=ruta+"/my_checkpoint.pth.tar")
    del checkpoint
    torch.cuda.empty_cache()
    return

def load_model(model, ruta, filename):
    print('=> Loading model\t')
    load_checkpoint(ruta+'/'+filename, model)
    return

def save_graphtv(train_loss, valid_loss, ruta, filename):
    epochs=[i for i in range(len(train_loss))]
    
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(epochs, train_loss, "b-", label="Train loss")
    line2 = ax1.plot(epochs, valid_loss, "r-", label="Valid loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    lns=line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(line1+line2, labs, loc="center right")
    
    plt.close(fig)
    plt.show()
    
    #TrainValidationLoss.png
    fig.savefig(ruta+"/"+filename)
    return

def save_graphtvd(train_dice, valid_dice, ruta, filename):
    epochs=[i for i in range(len(train_dice))]
    
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(epochs, train_dice, "b-", label="Train Dice")
    line2 = ax1.plot(epochs, valid_dice, "r-", label="Valid Dice")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dice")
    
    lns=line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(line1+line2, labs, loc="center right")
    
    plt.close(fig)
    plt.show()
    
    #TrainValidationLoss.png
    fig.savefig(ruta+"/"+filename)
    return

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True,
):
    train_ds = PlacentaDataSetMC(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = PlacentaDataSetMC(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            print(preds.shape, preds.dtype)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.3f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return

def check_accuracy2(loader, model, loss_fn, device="cuda"):
    model.eval()
    num_correct=0
    num_pixels=0
    
    """For monitoring validation loss"""
    running_loss = 0.0
    #Other metrics in order to compare
    dices=[]
    ious=[]
    hds=[]
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().to(device)
            preds = model(x)
            
            #Takes the placenta class
            pbool = F.softmax(preds, dim=1)
            sem_classes=['__background__', 'placenta']
            sem_class_to_idx={cls:idx for (idx, cls) in enumerate(sem_classes)}
            class_dim=1
            pbool = pbool.argmax(class_dim) == sem_class_to_idx['placenta']
            
            """Metrics for evaluation"""
            #Accuracy
            num_correct += (pbool==y.unsqueeze(1)).sum()
            num_pixels += torch.numel(pbool)
            
            #Dice coefficient
            dice = dice_loss(preds, y)
            dice = 1-dice
            dices.append(dice.detach().item()) #Append to dices

            #IoU
            iou = IoU_loss(preds, y)
            iou = 1-iou
            ious.append(iou.detach().item())#Append to ious
            
            #Hausdorff Distance
            hd = hdistance_loss(preds, y)
            hds.append(hd)
            
            #Loss function
            loss=loss_fn(preds, y)
            running_loss += loss*loader.batch_size
            
            #For saving memory
            del x, y, preds
            torch.cuda.empty_cache()
            
    print(f"Got Dice score mean:\t {np.mean(dices):.8f}")
    print(f"Got Dice score max:\t {np.max(dices):.8f}")
    print(f"Got Dice score min:\t {np.min(dices):.8f}")
    print(f"Got Dice score std:\t {np.std(dices):.8f}")
    print(f"Got IoU score mean:\t {np.mean(ious):.8f}")
    print(f"Got IoU score max:\t {np.max(ious):.8f}")
    print(f"Got IoU score min:\t {np.min(ious):.8f}")
    print(f"Got IoU score std:\t {np.std(ious):.8f}")
    print(f"Got H. distance mean:\t {np.mean(hds):.8f}")
    print(f"Got H. distance max:\t {np.max(hds):.8f}")
    print(f"Got H. distance min:\t {np.min(hds):.8f}")
    print(f"Got H. distance std:\t {np.std(hds):.8f}")
    print(f"Got Accuracy:\t\t {num_correct/num_pixels*100:.3f} %")
    
    """Determina dice promedio y valor de función coste"""
    loss_out=running_loss/len(loader)
    model.train()    
    return dices, loss_out.detach().item() #2n argument: dices, dice_out.detach().item()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    print("=> Saving images")
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
        torchvision.utils.save_image(x, f"{folder}/in_{idx}.png")
        torchvision.utils.save_image(preds, f"{folder}/prd_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/msk_{idx}.png")

    model.train()
    return

def set_title(tensor,string):
    tensor=tensor.squeeze()
    numpy=tensor.cpu().detach().numpy()
    numpy=np.transpose(numpy, (1,2,0))
    pil_image=Image.fromarray(numpy)    
    
    #font=ImageFont.truetype('arial.ttf',25)
    font=ImageFont.truetype(r'/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', 20)
    
    w, h = font.getsize(string)
    draw=ImageDraw.Draw(pil_image)
    draw.text(((176-w)/2, 14*(192-h)/15), string, font=font, fill='white')
    numpy=np.array(pil_image)
    numpy=np.transpose(numpy, (2,0,1))
    tensor=torch.from_numpy(numpy)
    tensor=tensor.unsqueeze(dim=0) 
    
    return tensor

#When CrossEntropyLoss or DiceLoss is used
def save_predictions_as_imgs2(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    dices=[]
    dice_score=0
    
    print("=> Saving images")
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = F.softmax(model(x), dim=1)
            
        #Convert y to calculate dice and put into image
        yaux=y.long().to(device)
        
        #Takes the placenta class
        sem_classes=['__background__', 'placenta']
        sem_class_to_idx={cls:idx for (idx, cls) in enumerate(sem_classes)}
        class_dim=1
        pbool = preds.argmax(class_dim) == sem_class_to_idx['placenta']
        
        #Dice coefficient
        dice=dice_loss(preds, yaux)
        dice= 1-dice
        dice= dice.detach().item()
        dices.append(dice)
        dice_score+=dice
        
        #Save input image, target and pred images
        torchvision.utils.save_image(x, f"{folder}/in_{idx}.png")
        #torchvision.utils.save_image(preds.unsqueeze(1).float(), f"{folder}/prd_{idx}.png")
        #torchvision.utils.save_image(y.unsqueeze(1).float(), f"{folder}/msk_{idx}.png")
        
        #Obtain original images with masks and predictions on top
        over_masks, over_preds = overlay_imgs2(x, y, pbool)
                
        #Set dice index to the over_preds
        over_preds=set_title(over_preds, 'Dice='+str(dice))
        
        #For save with torch.utils.save_image()
        over_masks = (over_masks.float())/255.00
        over_preds = (over_preds.float())/255.00        
        
        torchvision.utils.save_image(over_masks, f"{folder}/overmsk_{idx}.png")
        torchvision.utils.save_image(over_preds, f"{folder}/overprd_{idx}.png")
                                
        del x, y, preds, pbool
        torch.cuda.empty_cache()
    
    print(f"Got Dice score:    \t {dice_score/len(loader)}")    
    print(f"Got Dice score std:\t {np.std(dices):.6f}")
    print(f"Got Dice score min:\t {np.min(dices):.6f}")
    print(f"Got Dice score max:\t {np.max(dices):.6f}")
    print(f"Got Dice score mean:\t {np.mean(dices):.6f}")    
    model.train()
    
    return

#When CrossEntropyLoss or DiceLoss is used
def save_predictions_as_imgs3(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    dice_score=0
    dices=[]
    print("=> Saving images")
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x=x.to(device)
            yaux=y.long().to(device)
            preds=model(x)
            
            #Takes the placenta class
            pbool = F.softmax(preds, dim=1)
            sem_classes=['__background__', 'placenta']
            sem_class_to_idx={cls:idx for (idx, cls) in enumerate(sem_classes)}
            class_dim=1
            pbool = pbool.argmax(class_dim) == sem_class_to_idx['placenta']
            
            #Dice coefficient
            dice = dice_loss(preds, yaux)
            dice = 1-dice
            dice = dice.detach().item()
            dices.append(dice)
            dice_score += dice
            
            #Save original image
            torchvision.utils.save_image(x, f"{folder}/in_{idx}.png")
            #Obtain original images with masks and predictions on top
            over_masks, over_preds = overlay_imgs2(x, y, pbool)
            #Set dice index to the over_preds
            over_preds=set_title(over_preds, 'Dice='+str(dice))
            
            #For save with torch.utils.save_image()
            over_masks = (over_masks.float())/255.00
            over_preds = (over_preds.float())/255.00 
            
            torchvision.utils.save_image(over_masks, f"{folder}/overmsk_{idx}.png")
            torchvision.utils.save_image(over_preds, f"{folder}/overprd_{idx}.png")
            
            #For savinf memory
            del x,y,preds
            torch.cuda.empty_cache()
            
    #print(f"Got Dice score: \t {dice_score/len(loader)}")
    # print(f"Got Dice score mean:\t {np.mean(dices):.6f}")
    # print(f"Got Dice score max:\t {np.max(dices):.6f}")
    # print(f"Got Dice score min:\t {np.min(dices):.6f}")
    # print(f"Got Dice score std:\t {np.std(dices):.6f}")
    model.train()
    
    return


def overlay_imgs(inputs, masks, preds, alpha=0.2):
    #Lists to concatenate and recover a batch
    lst_masks=[]
    lst_preds=[]
    inputs=inputs.to("cpu")
    preds=preds.to("cpu")
   
    for i in range(inputs.shape[0]):
        img=inputs[i]*255
        img=img.type(torch.uint8)
        mask=masks[i].type(torch.bool)
        pred=preds[i]
        img_and_mask=draw_segmentation_masks(image=img, masks=mask, 
                                             alpha=alpha, colors=(0,250,0))
        img_and_pred=draw_segmentation_masks(image=img, masks=pred, 
                                             alpha=alpha, colors=(250,0,0))
        
        lst_masks.append(img_and_mask.unsqueeze(dim=0))
        lst_preds.append(img_and_pred.unsqueeze(dim=0))
    
    #Recover a tensor BxCxHxW
    tensor_masks=torch.cat(lst_masks, dim=0)
    tensor_preds=torch.cat(lst_preds, dim=0)
    
    return tensor_masks, tensor_preds

def overlay_imgs2(inputs, masks, preds, alpha=0.2):
    #Lists to concatenate and recover a batch
    lst_masks=[]
    lst_preds=[]
    inputs=inputs.to("cpu")
    preds=preds.to("cpu")

    for i in range(inputs.shape[0]):
        img=inputs[i]*255
        img=img.type(torch.uint8)
        
        mask=masks[i].type(torch.bool)
        pred=preds[i]
        img_and_mask=draw_segmentation_masks(image=img, masks=mask, 
                                             alpha=alpha, colors=(0,250,0))
        img_and_pred=draw_segmentation_masks(image=img, masks=pred, 
                                             alpha=alpha, colors=(250,0,0))
        lst_masks.append(img_and_mask.unsqueeze(dim=0))
        lst_preds.append(img_and_pred.unsqueeze(dim=0))
    
    #Recover a tensor BxCxHxW
    tensor_masks=torch.cat(lst_masks, dim=0)
    tensor_preds=torch.cat(lst_preds, dim=0)
    
    return tensor_masks, tensor_preds

def saveResults(filename, *args, **kargs):
    f=open(filename, 'w')
    for i in args:
        f.writelines(str(i)+'\n')
    f.close()
    return

def saveTrainDetails(train_size, valid_size,
                      learning_rate, nepochs,
                      im_h, im_w,
                      loss_fn, optimizer,                      
                      dices, ious, hds,
                      # dices, dice_std,
                      # dice_min, dice_max,
                      summary_model,
                      filename,
                      ):
    saveResults(filename, 
                'Train_size:', train_size, 'Valid_size:', valid_size,
                'Learning_rate:', learning_rate, 'Num_epochs:', nepochs,
                'Image_height:', im_h, 'Image_width:', im_w,
                'Loss_fn:', loss_fn, 'Optimizer:', optimizer,
                
                'DiceMean:', np.mean(dices), 'DiceMax:', np.max(dices),
                'DiceMin:', np.min(dices), 'DiceStd:', np.std(dices),
                
                'IoUMean:', np.mean(ious), 'IoUMax:', np.max(ious),
                'IoUMin:', np.min(ious), 'IoUStd:', np.std(ious),
                
                'HdMean:', np.mean(hds), 'HdMax:', np.max(hds),
                'HdMin:', np.min(hds), 'HdStd:', np.std(hds),
                
                'Summary_Model:', summary_model,
                )
    return

#Receive 2 tensors BxCxWxH, predictions and labels,     
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice