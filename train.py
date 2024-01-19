from utils import *
from dataset import BottleDataset
from model import UNet

from torch.utils.data import DataLoader
import torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from glob import glob
import os
import torch
import torch.nn as nn

train_size = 224

train_transforms = A.Compose(
    [
        A.Resize(width=train_size, height=train_size),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        A.Blur(),
        A.Sharpen(),
        A.RGBShift(),
        A.Cutout(num_holes=5, max_h_size=25, max_w_size=25, fill_value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(width=train_size, height=train_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
n_workers = os.cpu_count()
print(f"Using {n_workers} workers")

train_loader = DataLoader(
    BottleDataset(
        image_dir="bottle/train/images/",
        mask_dir="bottle/train/masks/",
        transforms=train_transforms,
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_workers,
)

val_loader = DataLoader(
    BottleDataset(
        image_dir="bottle/val/images/",
        mask_dir="bottle/val/masks/",
        transforms=val_transforms,
    ),
    batch_size=batch_size,
    shuffle=False,
    num_workers=n_workers,
)

model = UNet(1).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
n_eps = 30

dice_fn = torchmetrics.Dice(eps=1e-7, average="macro").to(device)

iou_fn = torchmetrics.IoU(eps=1e-7, average="macro").to(device)

acc_meter = AverageMeter()
train_loss_meter = AverageMeter()
val_loss_meter = AverageMeter()
dice_meter = AverageMeter()
iou_meter = AverageMeter()

for epoch in range(1, 1+n_eps):
    acc_meter.reset()
    train_loss_meter.reset()
    val_loss_meter.reset()
    iou_meter.reset()
    dice_meter.reset()
    model.train()

    for batch_id, (x, y,) in enumerate(
        tqdm(train_loader, desc=f"Training epoch {epoch}", total=len(train_loader))
    ):
        optimizer.zero_grad()
        n = x.shape[0]
        x = x.to(device).float()
        y = y.to(device).long()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_mask = y_pred.argmax(dim=1).squeeze()
            dice_score = dice_fn(y_pred_mask, y.long())
            iou_score = iou_fn(y_pred_mask, y.long())
            acc = accuracy_function(y_pred_mask, y.long())

        train_loss_meter.update(loss.item(), n)
        dice_meter.update(dice_score.item(), n)
        acc_meter.update(acc.item(), n)
        iou_meter.update(iou_score.item(), n)
    print(
        f"Training epoch {epoch}: train_loss: {train_loss_meter.avg:.4f}, acc: {acc_meter.avg:.4f}, dice: {dice_meter.avg:.4f}, iou: {iou_meter.avg:.4f}"
    )
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"bottle_model_epoch_{epoch}.pth")
    