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
    
class BDataset(Dataset):
    def __init__(self, 
                 image_dir: str,
                 mask_dir: str,
                 scale: float = 1.0,
                 mask_suffix: str = '_mask',):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [
            splitext(file)[0] for file in listdir(image_dir) if not file.startswith('.')
        ]

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning images and masks folders...')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))
            self.mask_values =list(sorted(
                np.unique(
                    np.concatenate(
                        unique, axis=0
                    ).tolist()
                )
            ))
        logging.info(f'Found {len(self.mask_values)} unique classes')
    
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        pil_img = pil_img.resize((newW, newH), resample = Image.NEAREST if is_mask else Image.BICUBIC)
        
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        img = Image.open(img_file[0])
        mask = Image.open(mask_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

