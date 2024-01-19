import torch
import albumentations as A
from tqdm import tqdm
from model import UNet
import torch.nn as nn
from dataset import *
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from utils import *
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_images = "/content/Faulty-product-detection-in-image/bottle/train/images/"
train_masks = "/content/Faulty-product-detection-in-image/bottle/train/masks/"

val_images = "/content/Faulty-product-detection-in-image/bottle/val/images/"
val_masks = "/content/Faulty-product-detection-in-image/bottle/val/masks/"
num_epochs = 20
learning_rate = 1e-4
batch_size = 4
pin_memory = True
num_workers = 2
load_model = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)
        
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
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    
    val_transforms = A.Compose(
        [A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
        ]
    )
    model = UNet(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loaders(
        train_images,
        train_masks,
        val_images,
        val_masks,
        batch_size,
        train_transform,
        val_transforms,
        pin_memory,
    )
    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        print(f"---------Epoch {epoch+1}/{num_epochs}-------")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"Epoch_{epoch}_checkpoint.pth.tar")

        # check accuracy
        check_accuracy(val_loader, model, device=device)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=device
        )

if __name__ == "__main__":
    main()
