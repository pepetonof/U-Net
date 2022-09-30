from torch.utils.data import Dataset
import numpy as np

from skimage import io
import torch

from scipy import ndimage, signal
from skimage.color.adapt_rgb import adapt_rgb, each_channel

##Dataset
class PlacentaDataSet(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir= image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, index:int):
        img_id  =self.image_dir[index]
        mask_id =self.mask_dir[index]
    
        image=io.imread(img_id).astype(np.float32)
        mask =io.imread(mask_id).astype(np.float32)
        mask[mask==255.0]=1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]
        
        image=np.transpose(image, (2,0,1))
        #Convert into a Tensor
        image=torch.from_numpy(image)
        mask=torch.from_numpy(mask)
        #image = image.unsqueeze(dim=0)
        #mask = mask.unsqueeze(dim=0)
        
        return image, mask

def gaussian(image, sigma):
    return ndimage.gaussian_filter(image, sigma=sigma)

def median(image, size):
    return ndimage.median_filter(image, size=size)

@adapt_rgb(each_channel)
def wiener(image, size):
    image=image.astype('float64')
    result = signal.wiener(image/255, mysize=size)
    result = result * 255
    result = result.astype('uint8')
    return result

#For multiclass segmentation and CrossEntropyLoss
class PlacentaDataSetMC(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir= image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_dir)
       
    def __getitem__(self, index:int):
        img_id  =self.image_dir[index]
        mask_id =self.mask_dir[index]
        
        image=io.imread(img_id)
        mask =io.imread(mask_id)
        
        # image = median(image, size=7)
        # image = gaussian(image, sigma=3)
        # image = wiener(image, size=7)
        
        # print('DS', image.shape, type(image))
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]#.astype(np.int_)
            mask[mask==255]=1
        
        return image, mask