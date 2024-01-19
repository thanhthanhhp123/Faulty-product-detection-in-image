import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from dataset import MyDataset
from utils import *


lr = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
epochs = 100

train_dataset = MyDataset('bottle/train/images', 'bottle/train/masks')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = MyDataset('bottle/val/images', 'bottle/val/masks')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

generator = torch.Generator().manual_seed(42)

model = UNet(in_channels=3, num_classes=1).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

#Create metrics for the model

dice_loss = DiceLoss()
iou = IoU()

#Training with evaluation after each epoch and saving the best model

best_loss = float('inf')

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    print('-' * 10)

    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += accuracy(outputs, masks)
        train_loss += loss.item()

        #Add IoU and Dice Loss
        train_iou = iou(outputs, masks)
        train_dice = dice_loss(outputs, masks)


    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_acc += accuracy(outputs, masks)
            val_loss += loss.item()

            #Add IoU and Dice Loss
            val_iou = iou(outputs, masks)
            val_dice = dice_loss(outputs, masks)


    train_acc = train_acc / len(train_loader)
    train_loss = train_loss / len(train_loader)
    val_acc = val_acc / len(val_loader)
    val_loss = val_loss / len(val_loader)



    print(f'Train loss: {train_loss:.4f} accuracy: {train_acc:.4f}')
    print(f'Val loss: {val_loss:.4f} accuracy: {val_acc:.4f}')
    print(f'Train IoU: {train_iou:.4f} Dice Loss: {train_dice:.4f}')
    print(f'Val IoU: {val_iou:.4f} Dice Loss: {val_dice:.4f}')


    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print('Saved best model')