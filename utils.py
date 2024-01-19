#Create metrics for the model

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        smooth = 1e-6

        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        score = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        return 1 - score

class IoU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        smooth = 1e-6

        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum() - intersection
        score = (intersection + smooth) / (union + smooth)
        return score

#Create a function to calculate the accuracy of the model

def accuracy(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    y_pred = torch.round(y_pred)
    correct = (y_pred == y_true).sum().float()
    acc = correct / y_true.shape[0]
    return acc

#Create a function to calculate the mean loss of the model

