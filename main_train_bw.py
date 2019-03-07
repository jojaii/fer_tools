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

datasets = ['AffectNet', 'FER2013', 'KDEF', 'ExpW']
# datasets = ['AffectNet', 'KDEF', 'ExpW']
input_shape = (64, 64, 1)
# model_manager = ModelManager(datasets,
#                              input_shape)
num_classes = 7
ds_root = '/home/ubuntu/Desktop'
ds_dict = {}
split = 0.1
# LOAD DATASETS INTO DICTIONARY


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

    # load train, test, val

    # print(header + " Grouping data...")
    # print(data_loader.split_counts)
    data_loader.emotions = pd.get_dummies(data_loader.emotions).as_matrix()
    data_loader.faces = preprocess_input(data_loader.faces)
    for i, split_count in enumerate(data_loader.split_counts):
        maximum = sum(split_count)
        # print(split_count[0])
        # print(split_count[1])
        # print(maximum)
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
        # print("----"+str(lindex))

    train_ints = [emotion.argmax() for emotion in train_labels]
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_ints),
                                                      train_ints)
    labels = get_labels(ds_name)
    # print class per line
    print("["+str(ds_name)+"] (LABEL) class weight:")
    for i in range(num_classes):
        temp = "["+str(ds_name)+"]  ("+str(labels[i])+") "+str(class_weights[i])
        print(temp)

    # change class_weights from an array to a dictionary
    class_weights_dict = dict(enumerate(class_weights))
    
    train_faces, train_labels = shuffle_in_unison(train_faces, train_labels)
    val_faces, val_labels = shuffle_in_unison(val_faces, val_labels)
    test_faces, test_labels = shuffle_in_unison(test_faces, test_labels)

    ## DONT FORGET TO PREPROCESS
    model_loader = ModelManager(ds_name)
    model_loader.train_data = (train_faces, train_labels)
    model_loader.val_data = (val_faces, val_labels)
    model_loader.test_data = (test_faces, test_labels)
    print(header + " Training...")
    model_loader.train_with_weights(class_weights_dict)
    
    del(model_loader)

    # Mmanager.test()
    del(data_loader)

    del(train_faces)
    del(train_labels)
    del(val_faces)
    del(val_labels)
    del(test_faces)
    del(test_labels)
