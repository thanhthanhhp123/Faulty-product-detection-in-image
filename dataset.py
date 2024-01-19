import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, 
                 images_dir,
                 masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        self.images =sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))


        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask
    

if __name__ == '__main__':
    images_dir = 'bottle/train/images/'
    masks_dir = 'bottle/train/masks/'

    dataset = MyDataset(images_dir, masks_dir)

    image, mask = dataset[0]
    print(image.shape)
    print(mask.shape)