import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

import cv2
from sklearn.model_selection import train_test_split

from parameter import *
from utils import visualize_image

def resize_image(img, size):
    re_size = (size, size)
    re_image = cv2.resize(img, dsize=re_size, interpolation=cv2.INTER_AREA)

    return re_image

def processed_cnt(cnt):
    if cnt % 100 == 0:
        print("{0}`s order".format(str(cnt)))

def generate_train_test_data_npz(mode, data_list, size):
    print("generate {0} data".format(mode))

    def generate_npz(size, name_list, *data_lists):
        x_data, y_data = data_lists
        
        print(type(x_data), type(y_data))

        if y_data is not None:
            for i, n in enumerate(name_list):
                processed_cnt(i)
                X = cv2.imread(os.path.join(TRAIN_PATH, n+IMAGE_EXT), cv2.IMREAD_COLOR)
                X = cv2.normalize(X, None, 0, 1, cv2.NORM_MINMAX)
                X = resize_image(X, size)

                Y = cv2.VideoCapture(os.path.join(TRAIN_MASKS_PATH, n+MASKS_EXT)).read()[1]
                Y = cv2.normalize(Y, None, 0, 1, cv2.NORM_MINMAX)
                Y = resize_image(Y, size)

                x_data[i] = X
                y_data[i] = Y

            np.savez(os.path.join(DATA_PATH, '{0}_{1}_processed_data.npz'.format(str(size), TRAIN)), x_data=x_data, y_data=y_data)

        elif y_data is None:
            for i, n in enumerate(name_list):
                processed_cnt(i)
                X = cv2.imread(os.path.join(TEST_PATH, n+IMAGE_EXT), cv2.IMREAD_COLOR)
                X = cv2.normalize(X, None, 0, 1, cv2.NORM_MINMAX)
                X = resize_image(X, size)

                x_data[i] = X

            np.savez(os.path.join(DATA_PATH, '{0}_{1}_processed_data.npz'.format(str(size), TEST)), x_data=x_data)

        return x_data, y_data



    if mode == TRAIN:
        # TRAIN_LIST, TEST_LIST are defiend in utils.parameter.py
        train_name_list = [os.path.splitext(d)[0] for d in data_list]

        x_data = np.ones([len(data_list), size, size, 3], dtype="uint8")
        y_data = np.ones([len(data_list), size, size, 3], dtype="uint8")

        x_data, y_data = generate_npz(size, train_name_list, x_data, y_data)

    elif mode == TEST:
        # TRAIN_LIST, TEST_LIST are defiend in utils.parameter.py
        test_name_list = [os.path.splitext(d)[0] for d in data_list]

        x_data = np.ones([len(data_list), size, size, 3], dtype="uint8")

        x_data, _ = generate_npz(size, test_name_list, x_data, None)

    else:
        print("mode error")
        exit()



if __name__ == "__main__":
    generate_train_test_data_npz(TRAIN, TRAIN_LIST, 128)
    generate_train_test_data_npz(TEST, TEST_LIST, 128)



