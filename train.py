import torch
import albumentations as A
from tqdm import tqdm
from model import UNet
import torch.nn as nn
from dataset import RivetDataset
from albumentations.pytorch import ToTensorV2
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")