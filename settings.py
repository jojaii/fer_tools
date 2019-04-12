import os
import sys

# TRAIN_DATASET_NAMES = ['AffectNet', 'ExpW', 'FER2013', 'KDEF', 'SFEW']
TRAIN_DATASET_NAMES = ['AffectNet', 'ExpW', 'FER2013']
INPUT_SHAPE = (64, 64, 1)
DATASET_ROOT = 'data'
MODEL_ROOT = 'models'
BIN_ROOT = 'bin'
SPLIT = 0.1
UNDERSAMPLE_FLAG = False
WEIGHT_TYPE = "rw"  # bw or rw (balanced weight or randomized)
TRAIN_ON_ALL = True
NUM_CLASSES = 7

WHITE_LIST = ['angry', 'happy', 'neutral', 'sad']
WHITE_LIST = sorted([entry.lower() for entry in WHITE_LIST])


TEST_DATASET_NAMES = ['SFEW']
MODEL_PATHS = {'FER2013': 'FER2013_64_10_rw_mini_XCEPTION.48-0.74.hdf5'}
# PREDS_PATHS = {'SFEW':'SFEW_preds_10_rw_2019-4-13-43.csv'}
# PROBS_PATHS = {'SFEW':'SFEW_probs_10_rw_2019-4-13-43.csv'}
PREDS_PATHS = 'SFEW_preds_10_rw_2019-4-13-43.csv'
PROBS_PATHS = 'SFEW_probs_10_rw_2019-4-13-43.csv'
# NUM_CLASSES depends on the number of classes in the original csv files














# ds_names = ['AffectNet', 'ExpW', 'FER2013', 'KDEF', 'SFEW']

# m_paths = ["FER2013_64_20_bw_mini_XCEPTION.57-0.61.hdf5"]
# m_paths = ['fer2013_mini_XCEPTION.110-0.65.hdf5']
# m_paths = ['FER2013_64_10_rw_mini_XCEPTION.51-0.75.hdf5']
# m_paths = ["AffectNet_64_10_bw_mini_XCEPTION.30-0.68.hdf5",
#            "ExpW_64_10_bw_mini_XCEPTION.13-0.56.hdf5",
#            "FER2013_64_10_bw_mini_XCEPTION.73-0.64.hdf5",
#            "KDEF_64_10_bw_mini_XCEPTION.47-0.22.hdf5"]
# m_paths = ["AffectNet_64_20_bw_mini_XCEPTION.26-0.71.hdf5",
#            "FER2013_64_20_bw_mini_XCEPTION.57-0.61.hdf5",
#            "KDEF_64_20_bw_mini_XCEPTION.45-0.20.hdf5",
#            "ExpW_64_20_bw_mini_XCEPTION.39-0.55.hdf5"]
# m_paths = ["AffectNet_64_10_us_mini_XCEPTION.82-0.55.hdf5",
#            "FER2013_64_10_us_mini_XCEPTION.57-0.54.hdf5",
#            "KDEF_64_10_us_mini_XCEPTION.67-0.23.hdf5",
#            "ExpW_64_10_us_mini_XCEPTION.56-0.42.hdf5"]
# m_paths = ["AffectNet_64_20_us_mini_XCEPTION.85-0.58.hdf5",
#            "FER2013_64_20_us_mini_XCEPTION.45-0.48.hdf5",
#            "KDEF_64_20_us_mini_XCEPTION.56-0.19.hdf5",
#            "ExpW_64_20_us_mini_XCEPTION.42-0.42.hdf5"]