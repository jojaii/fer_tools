"""
File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""

import os
from utils.model_manager import ModelManager
from utils.data_manager import DSManager
from utils.common import encode_onehot, get_labels, shuffle_in_unison
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from utils.preprocessor import preprocess_input

# datasets = ['AffectNet', 'FER2013', 'KDEF', 'ExpW', 'SFEW']
datasets = ['AffectNet', 'ExpW', 'FER2013']
input_shape = (64, 64, 1)
num_classes = 7
ds_root = 'data'
ds_dict = {}
split = 0.2
all_train_faces = []
all_train_labels = []
all_val_faces = []
all_val_labels = []

# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(datasets):
    # load/create csv

    csv_name = str(ds_name)+"_"+str(input_shape[0])+"_"+str(int(split*100))+"_bw.csv"
    header = "["+str(ds_name)+"]"
    # TODO: default input for csv_file
    data_loader = DSManager(ds_name,
                            ds_root=ds_root,
                            image_size=input_shape[:2],
                            csv_file=csv_name)
    try:
        csv_file = open(os.path.join(ds_root, csv_name))
    except:
        print(header + " File "+csv_name+" does not exist. Exiting.")
        exit()
    else:
        print(header + " File "+csv_name+" exists. Loading file...")
        data_loader.load_split_CSV()

    data_loader.emotions = pd.get_dummies(data_loader.emotions).as_matrix()
    data_loader.faces = preprocess_input(data_loader.faces)
    for i, split_count in enumerate(data_loader.split_counts):
        maximum = sum(split_count)
        if i == 0:
            train_faces = data_loader.faces[:split_count[0]]
            train_labels = data_loader.emotions[:split_count[0]]
            val_faces = data_loader.faces[split_count[0]:split_count[0]+split_count[1]]
            val_labels = data_loader.emotions[split_count[0]:split_count[0]+split_count[1]]
            test_faces = data_loader.faces[split_count[0]+split_count[1]:maximum]
            test_labels = data_loader.emotions[split_count[0]+split_count[1]:maximum]
            lindex = maximum
        else:
            train_faces = np.vstack((train_faces, data_loader.faces[lindex:lindex+split_count[0]]))
            train_labels = np.vstack((train_labels, data_loader.emotions[lindex:lindex+split_count[0]]))
            val_faces = np.vstack((val_faces, data_loader.faces[lindex+split_count[0]:lindex+split_count[0]+split_count[1]]))
            val_labels = np.vstack((val_labels, data_loader.emotions[lindex+split_count[0]:lindex+split_count[0]+split_count[1]]))
            test_faces = np.vstack((test_faces, data_loader.faces[lindex+split_count[0]+split_count[1]:lindex+maximum]))
            test_labels = np.vstack((test_labels, data_loader.emotions[lindex+split_count[0]+split_count[1]:lindex+maximum]))
            lindex = lindex + maximum

    train_faces, train_labels = shuffle_in_unison(train_faces, train_labels)
    val_faces, val_labels = shuffle_in_unison(val_faces, val_labels)
    test_faces, test_labels = shuffle_in_unison(test_faces, test_labels)

    # DONT FORGET TO PREPROCESS

    # compile all data into a single group for training and validation
    if all_train_faces == []:
        all_train_faces = train_faces
        all_train_labels = train_labels
        all_val_faces = val_faces
        all_val_labels = val_labels
    else:
        all_train_faces = np.vstack((all_train_faces, train_faces))
        all_train_labels = np.vstack((all_train_labels, train_labels))
        all_val_faces = np.vstack((all_val_faces, val_faces))
        all_val_labels = np.vstack((all_val_labels, val_labels))

    del(data_loader)
    del(train_faces)
    del(train_labels)
    del(val_faces)
    del(val_labels)
    del(test_faces)
    del(test_labels)

model_loader = ModelManager("All")
model_loader.train_data = (all_train_faces, all_train_labels)
model_loader.val_data = (all_val_faces, all_val_labels)
print(header + " Training...")
model_loader.train()
