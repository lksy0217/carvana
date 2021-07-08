import os

# mode
TRAIN = "train"
TEST = "test"

# variables
IMAGE_EXT = ".jpg"
MASKS_EXT = "_mask.gif"

# data_path_list
DATA_PATH = "/home/lksy0217/carvana/Dataset"
CHECKPOINT_PATH = "/home/lksy0217/carvana/Checkpoint"
RESULT_PATH = "/home/lksy0217/carvana/Result"
TEST_PATH = os.path.join(DATA_PATH, "test")
TEST_HQ_PATH = os.path.join(DATA_PATH, "test_hq")
TRAIN_PATH = os.path.join(DATA_PATH, "train")
TRAIN_HQ_PATH = os.path.join(DATA_PATH, "train_hq")
TRAIN_MASKS_PATH = os.path.join(DATA_PATH, "train_masks")

# data_load
# train, train_hq, train_masks`s name is same
# test, test_hq`s name is same
TEST_LIST = sorted(os.listdir(TEST_PATH))
TEST_NAME_LIST = [os.path.splitext(t)[0] for t in TEST_LIST]
TRAIN_LIST = sorted(os.listdir(TRAIN_PATH))
TRAIN_NAME_LIST = [os.path.splitext(t)[0] for t in TRAIN_LIST]
#TEST_HQ_LIST = sorted(os.listdir(TEST_HQ_PATH))
#TRAIN_HQ_LIST = sorted(os.listdir(TRAIN_HQ_PATH))
#TRAIN_MASKS_LIST = sorted(os.listdir(TRAIN_MASKS_PATH))

# npz_name
NPZ_NAME = "{0}_{1}_processed_data.npz"
