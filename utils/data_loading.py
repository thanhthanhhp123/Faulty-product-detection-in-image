import logging 
import numpy as np
import os
from PIL import Image
from functools import partial, lru_cache
from  itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext,isfile, join
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(Image.open(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    
class BottleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        images = sorted(os.listdir(self.image_dir))
        masks = sorted(os.listdir(self.mask_dir))
        img_path = os.path.join(self.image_dir, images[index])
        mask_path = os.path.join(self.mask_dir, masks[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        
        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask

