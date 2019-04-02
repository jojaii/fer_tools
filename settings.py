import os
import sys

DATASET_NAMES = ['AffectNet', 'ExpW', 'FER2013', 'KDEF', 'SFEW']
INPUT_SHAPE = (64, 64, 1)
NUM_CLASSES = 7
DATASET_ROOT = '/home/ubuntu/Datasets'
SPLIT = 0.2
UNDERSAMPLE_FLAG = False
WEIGHT_TYPE = "rw"  # bw (balanced weight or randomized)
TRAIN_ON_ALL = False