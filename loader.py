# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:09:07 2021

@author: josef

Get validation and train loader. Validation and train loader are not always
same size
"""
from dataset import PlacentaDataSetMC
from data import get_data3
from torch.utils.data import DataLoader
from pathlib import Path
import albumentations as A
from albumentations.augmentations.geometric.transforms import ElasticTransform
#from albumentations.augmentations.transforms import ElasticTransform
from albumentations.pytorch.transforms import ToTensorV2

class loaders():
    def __init__(self, size_train, size_val, size_test, path_images, batch_size, num_workers=2, pin_memory=True):
        path=Path(path_images)
        #self.TRAIN_IMG_DIR, self.VAL_IMG_DIR, self.TRAIN_MASK_DIR, self.VAL_MASK_DIR = get_data2(size_train, path)
        self.TRAIN_IMG_DIR, self.VAL_IMG_DIR, self.TEST_IMG_DIR,self.TRAIN_MASK_DIR, self.VAL_MASK_DIR, self.TEST_MASK_DIR =get_data3(size_train, size_val, size_test, path)
        
        self.BATCH_SIZE=batch_size
        self.NUM_WORKERS=num_workers
        self.PIN_MEMORY=pin_memory
        
    def get_train_loader(self, H, W):
        train_transform = A.Compose(
            [   
                A.ToGray(p=1.0),
                # A.Equalize(p=1.0),
                A.Resize(height=H, width=W),
                A.HorizontalFlip(p=0.5),
                ElasticTransform(alpha=1, sigma=10, alpha_affine=20, interpolation=1, 
                                 border_mode= 0, approximate=True, p=0.8),
                A.GridDistortion(num_steps=10, border_mode=0, p=0.5),
                #A.RandomGamma(gamma_limit=[10,20],p=1.0),
                #A.GaussianBlur(p=1.0),
                # A.Normalize(
                #     mean=[0.0, 0.0, 0.0],
                #     std=[1.0, 1.0, 1.0],
                #     max_pixel_value=255.0,),
                A.Normalize(
                    mean=[0.0],
                    std=[1.0],
                    max_pixel_value=255.0,),
                ToTensorV2(),
            ],)
        train_ds = PlacentaDataSetMC(
            image_dir=self.TRAIN_IMG_DIR,
            mask_dir=self.TRAIN_MASK_DIR,
            transform=train_transform,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            pin_memory=self.PIN_MEMORY,
            shuffle=True,
        )
        return train_loader, train_ds,
    
    def get_val_loader(self, H, W, num_workers=2):
        val_transform = A.Compose(
                        [   
                            A.ToGray(p=1.0),
                            # A.Equalize(p=1.0),
                            A.Resize(height=H, width=W),
                            # A.Normalize(
                            #     mean=[0.0, 0.0, 0.0],
                            #     std=[1.0, 1.0, 1.0],
                            #     max_pixel_value=255.0,),
                            A.Normalize(
                                mean=[0.0],
                                std=[1.0],
                                max_pixel_value=255.0,),
                            ToTensorV2(),
                        ],)
        val_ds = PlacentaDataSetMC(
            image_dir=self.VAL_IMG_DIR,
            mask_dir=self.VAL_MASK_DIR,
            transform=val_transform,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            pin_memory=self.PIN_MEMORY,
            shuffle=False,
        )
        return val_loader, val_ds
    
    def get_test_loader(self, H, W, num_workers=2):
        test_transform = A.Compose(
                        [   
                            A.ToGray(p=1.0),
                            # A.Equalize(p=1.0),
                            A.Resize(height=H, width=W),
                            # A.Normalize(
                            #     mean=[0.0, 0.0, 0.0],
                            #     std=[1.0, 1.0, 1.0],
                            #     max_pixel_value=255.0,),
                            A.Normalize(
                                mean=[0.0],
                                std=[1.0],
                                max_pixel_value=255.0,),
                            ToTensorV2(),
                        ],)
        test_ds = PlacentaDataSetMC(
            image_dir=self.TEST_IMG_DIR,
            mask_dir=self.TEST_MASK_DIR,
            transform=test_transform,
        )
        
        test_loader = DataLoader(
            test_ds,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            pin_memory=self.PIN_MEMORY,
            shuffle=False,
        )
        return test_loader, test_ds