import os, argparse
import numpy as np
from parameter import *
from unet import unet_input_128, callbacks
from utils import plot_history, visualize_image
import cv2
from PIL import Image

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def config_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)

    return sess

def load_data_npz(size, mode):

    data = os.path.join(DATA_PATH, NPZ_NAME.format(str(size), mode))
    if not os.path.exists(data):
        print("First, you must generate npz using ""data.py"" file")
        exit()

    processed_data = np.load(data)

    if mode == TRAIN:
        x_train = processed_data["x_data"]
        y_train = processed_data["y_data"]

        return x_train, y_train

    elif mode == TEST:
        x_test = processed_data["x_data"]

        return x_test



if __name__ == "__main__":
    set_session(config_session())

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128, help="select 128 or 256(default:128)--but now, just can select 128")
    parser.add_argument("--val", type=float, default=0.2, help="enter float value(default:0.2)")
    parser.add_argument("--epoch", type=int, default=30, help="enter int value(default:30)")
    parser.add_argument("--batch_size", type=int, default=24, help="enter int value(default:24)")
    parser.add_argument("--mode", type=str, default="train", help="select train or test(default:train)")
    parser.add_argument("--test_cnt", type=int, default=0, help="if you select mode test, enter int value(default:0)\n\
                                                                 if you dont input(default), predict test_data[:]\n\
                                                                 else predict test_data[:test_cnt]")
    args = parser.parse_args()

    if args.mode == TRAIN:
        x_train, y_train = load_data_npz(args.size, TRAIN)
        visualize_image(TRAIN, x_train, y_train, 5)

        print("x_train     :  ", x_train.shape)
        print("y_train     :  ", y_train.shape)
        print("x_train[0]  :  ", x_train[0].shape)
        print("y_train[0]  :  ", y_train[0].shape)

        model = unet_input_128((None, args.size, args.size, 3), 3)    
        model.summary()
        callbacks = callbacks(args.size)
        history = model.fit(x_train, y_train, epochs=args.epoch, validation_split=args.val, batch_size=args.batch_size, callbacks=callbacks)
        plot_history(history)

    if args.mode == TEST:
        x_test = load_data_npz(args.size, TEST)

        print("x_test   :  ", x_test.shape)

        model = unet_input_128((None, args.size, args.size, 3), 3)
        checkpoint_path = os.path.join(CHECKPOINT_PATH, "train_result_{0}.ckpt".format(str(args.size)))
        model.load_weights(checkpoint_path)
        y_prediction = model.predict(x_test[:args.test_cnt])
        visualize_image(TEST, x_test[:args.test_cnt], y_prediction, 5)



