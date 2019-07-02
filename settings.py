import os
import sys

# TRAIN_DATASET_NAMES = ['AffectNet', 'ExpW', 'FER2013', 'KDEF', 'SFEW']
# TRAIN_DATASET_NAMES = ['AffectNet', 'ExpW', 'FER2013']
TRAIN_DATASET_NAMES = ['FERcolor']
TRAIN_DATASET_NAMES = sorted(TRAIN_DATASET_NAMES)
INPUT_SHAPE = (64, 64, 1) # if colored, set to nxnx3
# DATASET_ROOT = 'data'
DATASET_ROOT = "data"
MODEL_ROOT = 'models'
BIN_ROOT = 'bin'
SPLIT = 0.1
UNDERSAMPLE_FLAG = False
WEIGHT_TYPE = "rw"  # bw or rw (balanced weight or randomized)
TRAIN_ON_ALL = False
NUM_CLASSES = 7

WHITE_LIST_FLAG = False
# WHITE_LIST = ['angry', 'happy', 'neutral', 'sad']
WHITE_LIST = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
WHITE_LIST = sorted([entry.lower() for entry in WHITE_LIST])

COLOR_FLAG = False

TEST_DATASET_NAMES = ['AffectNet', 'ExpW', 'FER2013'] #  'SFEW'
# TEST_DATASET_NAMES = ['FERcolor'] #  'SFEW'
# TEST_DATASET_NAMES = ['FER2013'] #  'SFEW'
# MODEL_PATHS = {'AffectNet': 'AffectNet_64_10_rw_mini_XCEPTION.44-0.83.hdf5',
#                'ExpW': 'ExpW_64_10_rw_mini_XCEPTION.08-0.82.hdf5',
#                'FER2013': 'FER2013_64_10_rw_mini_XCEPTION.47-0.72.hdf5',
#                'All': 'All_64_10_rw_mini_XCEPTION.07-0.82.hdf5'}
MODEL_PATHS = {'FER2013':''}
TEST_ON_ALL = False


PREDS_PATHS = 'FERcolor_preds_10_rw_2019-5-16-1558.csv'
PROBS_PATHS = 'FERcolor_probs_10_rw_2019-5-16-1558.csv'
# NUM_CLASSES depends on the number of classes in the original csv files