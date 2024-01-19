import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

random.seed(42)


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
        # mask[mask == 255.0] = 1.0
        
        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask
    
if __name__ == '__main__':
    image_dir = "bottle/train/images/"
    mask_dir = "bottle/train/masks/"
    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))
    for i, j in zip(images, masks):
        if i != j:
            print(i, j)
