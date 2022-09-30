# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 07:51:38 2021

@author: josef
"""

##Para lectura de archivos desde github
import requests
from bs4 import BeautifulSoup
import re

##Para lecutra de archivos localmente
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

#Obtiene los nombres de los archivos de la liga a repositorio
def get_filenames(github_url:str):
    result = requests.get(github_url)
    soup = BeautifulSoup(result.text, 'html.parser')
    pngfiles = soup.find_all(title=re.compile("\.png$"))
    filename = []
    for i in pngfiles:
        filename.append(i.extract().get_text())
    return filename

#Determina el indice del titulo de las imágenes
def get_ind(string:str):
    num=""
    for i in string:
        if i.isdigit():
            num=num+i  
    return int(num)

#Concatena con dirección de repositorio de imagen y máscara
def concat_git(url,lst:list):
    for i in range(len(lst)):
        lst[i]=url+lst[i]

#Get validation an train data
def get_data(train_size):
    ##url de imágenes y máscaras para lectura
    inputs='https://raw.githubusercontent.com/pepetonof/unet_hu/main/input/'
    masks= 'https://raw.githubusercontent.com/pepetonof/unet_hu/main/target/'
    ##List files on github repository
    github_url_inp = 'https://github.com/pepetonof/unet_hu/tree/main/input'
    github_url_msk = 'https://github.com/pepetonof/unet_hu/tree/main/target'
    
    #Obtiene lista y la ordena con base en el indice de la imagen    
    filenames_inp=get_filenames(github_url_inp)
    filenames_msk=get_filenames(github_url_msk)
        
    filenames_inp.sort(key=get_ind)
    filenames_msk.sort(key=get_ind)
    
    #Determina los url de las imágenes con url completo
    concat_git(inputs,filenames_inp)
    concat_git(masks, filenames_msk)
    
    ##Split into train and valid data to 70% and 30%
    #train_size=0.7
    random_seed=0
    TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR = train_test_split(
        filenames_inp,
        filenames_msk,
        train_size=train_size,
        random_state=random_seed,
        shuffle=True
    )
    
    return TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR

def get_ind2(p:Path()):
    string=os.path.basename(p)
    num=""
    for i in string:
        if i.isdigit():
            num=num+i  
    return int(num)

def get_data2(train_size, p):
    if not os.path.exists(p):
        raise ValueError('Debe existir el directorio')
    dirs=[x for x in p.iterdir() if x.is_dir()] #Directorios
    
    files_inp=list(dirs[0].glob('**/*.png')) #Images input
    files_msk=list(dirs[1].glob('**/*.png')) #Images output
        
    #Ordering according to index in tittle image
    files_inp.sort(key=get_ind2)
    files_msk.sort(key=get_ind2)

    random_seed=0
    TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR = train_test_split(
        files_inp,
        files_msk,
        train_size=train_size,
        random_state=random_seed,
        shuffle=True
    )
    return TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR

#Se toma en cuenta conjunto de entrenamiento, validación y prueba
def get_data3(train_size, val_size, test_size, p):
    if not os.path.exists(p):
        raise ValueError('Debe existir el directorio')
    if train_size + val_size + test_size != 1.0:
        raise ValueError('La suma de los tamaños debe ser igual a 1')
    
    dirs=[x for x in p.iterdir() if x.is_dir()] #Directorios
    files_inp=list(dirs[0].glob('**/*.png')) #Images input
    files_msk=list(dirs[1].glob('**/*.png')) #Images output
    
    #Ordering according to index in tittle image
    files_inp.sort(key=get_ind2)
    files_msk.sort(key=get_ind2)
    
    random_seed=0
    
    #Test data
    x_remain, TEST_IMG_DIR, y_remain, TEST_MASK_DIR = train_test_split(
        files_inp,
        files_msk,
        test_size=test_size,
        random_state=random_seed,
        shuffle=False
        )
    
    #Adjust val_size, train_size
    remain_size = 1.0 - test_size
    val_size_adj =val_size / remain_size
    
    TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR = train_test_split(
        x_remain,
        y_remain, 
        train_size = 1-val_size_adj,
        random_state = random_seed,
        shuffle=False
        )
    
    return TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR, TEST_MASK_DIR
    