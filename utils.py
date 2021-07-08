import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as image
from PIL import Image
import os, cv2

from parameter import *

def save_history_history(fname, history_history, fold=''):
    np.save(os.path.join(fold, fname), history_history)


def load_history_history(fname, fold=''):
    history_history = np.load(os.path.join(fold, fname)).item(0)
    return history_history

def plot_dice_coeff(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['dice_coeff'])
    plt.plot(history['val_dice_coeff'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Dice_coeff')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)


def plot_loss(history, title=None):
    # summarize history for loss
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)


def plot_history(history):
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 2, 1)
    plot_dice_coeff(history, "Dice_coeff")
    plt.subplot(1, 2, 2)
    plot_loss(history, "Loss")
    plt.savefig(os.path.join(RESULT_PATH, "plot_history.jpg"))

def visualize_image(mode, original_images, masks_images, image_cnt):
    image_selector = random.sample(range(0, len(original_images)), image_cnt)

    plt.figure(figsize=(10, 15))
    for i, idx in enumerate(image_selector):
        plt.subplot(image_cnt, 2, i*2+1)
        target_image = cv2.normalize(original_images[i], None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        target_image = Image.fromarray(target_image)
        plt.imshow(target_image)
        plt.title(mode+"__"+str(i)+"__Original")

        plt.subplot(image_cnt, 2, i*2+2)
        target_image = cv2.normalize(masks_images[i], None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        target_image = Image.fromarray(target_image)
        plt.imshow(target_image)
        plt.title(mode+"__"+str(i)+"__Masks")

    plt.show()
    plt.savefig(os.path.join(RESULT_PATH, "{0}_{1}_plots.jpg".format(mode, str(image_cnt))))

