import numpy as np 
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def dice_coef(y_true, y_pred, smooth = 100):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth = 100):
    return 1 - dice_coef(y_true, y_pred, smooth)

def iou_coef(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    return (intersection + smooth) / ( union + smooth)

def show_images(images, mask):
    plt.figure(figsize=(15,15))
    for i in range(25):
        plt.subplot(5,5,i+1)
        img_path = images[i]
        mask_path = mask[i]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)

        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training(hist):
    '''
    This function take training model and plot history of accuracy and losses with the best epoch in both of them.
    '''

    # Define needed variables
    tr_acc = hist.history['accuracy']
    tr_iou = hist.history['iou_coef']
    tr_dice = hist.history['dice_coef']
    tr_loss = hist.history['loss']

    val_acc = hist.history['val_accuracy']
    val_iou = hist.history['val_iou_coef']
    val_dice = hist.history['val_dice_coef']
    val_loss = hist.history['val_loss']

    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    index_iou = np.argmax(iou_coef)
    iou_highest = val_iou[index_iou]
    index_dice = np.argmax(dice_coef)
    dice_highest = val_dice[index_dice]
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]

    Epochs = [i+1 for i in range(len(tr_acc))]

    acc_label = f'best epoch= {str(index_acc + 1)}'
    iou_label = f'best epoch= {str(index_iou + 1)}'
    dice_label = f'best epoch= {str(index_dice + 1)}'
    loss_label = f'best epoch= {str(index_loss + 1)}'

    # Plot training history
    plt.figure(figsize= (20, 20))
    plt.style.use('fivethirtyeight')

    # Training Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Training IoU
    plt.subplot(2, 2, 2)
    plt.plot(Epochs, tr_iou, 'r', label= 'Training IoU')
    plt.plot(Epochs, val_iou, 'g', label= 'Validation IoU')
    plt.scatter(index_iou + 1 , iou_highest, s= 150, c= 'blue', label= iou_label)
    plt.title('Training and Validation IoU Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    # Training Dice
    plt.subplot(2, 2, 3)
    plt.plot(Epochs, tr_dice, 'r', label= 'Training Dice')
    plt.plot(Epochs, val_dice, 'g', label= 'Validation Dice')
    plt.scatter(index_dice + 1 , dice_highest, s= 150, c= 'blue', label= dice_label)
    plt.title('Training and Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()

    # Training Loss
    plt.subplot(2, 2, 4)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout
    plt.savefig('training_history.png')
    plt.show()

def create_ds(images_path, masks_path, train_aug, val_aug):
    img_size = (256, 256)

    batch_size = 4

    train_img_gen = ImageDataGenerator(**train_aug)
    train_mask_gen = ImageDataGenerator(**train_aug)

    train_img_ds = train_img_gen.flow_from_directory(images_path, target_size=img_size, batch_size=batch_size, class_mode=None, seed=42, subset='training')
    train_mask_ds = train_mask_gen.flow_from_directory(masks_path, target_size=img_size, batch_size=batch_size, class_mode=None, seed=42, subset='training')

    train_ds = zip(train_img_ds, train_mask_ds)

    val_img_gen = ImageDataGenerator(**val_aug)
    val_mask_gen = ImageDataGenerator(**val_aug)

    val_img_ds = val_img_gen.flow_from_directory(images_path, target_size=img_size, batch_size=batch_size, class_mode=None, seed=42, subset='validation')
    val_mask_ds = val_mask_gen.flow_from_directory(masks_path, target_size=img_size, batch_size=batch_size, class_mode=None, seed=42, subset='validation')

    val_ds = zip(val_img_ds, val_mask_ds)

    return train_ds, val_ds
    