from model import unet
from utils import *

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, concatenate_images

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

import warnings
warnings.filterwarnings("ignore")

images_path = 'bottle/image'
mask_path = 'bottle/ground_truth'

train_aug = dict(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=5,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

val_aug = dict(
    validation_split=0.1
)

train_ds, val_ds = create_ds(
    images_path,
    mask_path,
    train_aug,
    val_aug
)

model = unet(input_size=(256, 256, 3))
model.compile(optimizer=Adam(learning_rate=0.001), loss=dice_loss, metrics=['accuracy', iou_coef, dice_coef])

model.summary()

epochs = 100
batch_size = 4

callbacks = [
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True),
]

history = model.fit(
    train_ds,
    steps_per_epoch=len(train_ds) / batch_size,
    validation_data=val_ds,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    validation_steps=len(val_ds) / batch_size
)