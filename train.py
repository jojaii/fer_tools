
import os
import settings
from utils.model_manager import ModelManager
from utils.data_manager import DSManager
from utils.common import (get_labels, shuffle_in_unison,
                          compute_weights, filter_labels,
                          convert_labels)
import numpy as np
import pandas as pd
from utils.preprocessor import preprocess_input

datasets = settings.TRAIN_DATASET_NAMES
input_shape = settings.INPUT_SHAPE
num_classes = len(settings.WHITE_LIST)
# num_classes = settings.NUM_CLASSES
ds_root = settings.DATASET_ROOT
m_root = settings.MODEL_ROOT
split = settings.SPLIT
split_str = str(int(split*100))
undersample = settings.UNDERSAMPLE_FLAG
weight_type = settings.WEIGHT_TYPE
train_on_all = settings.TRAIN_ON_ALL
white_list_flag = settings.WHITE_LIST_FLAG
color_flag = settings.COLOR_FLAG
ds_dict = {}
class_weights_dict = []
all_train_faces, all_train_labels = [], []
all_val_faces, all_val_labels = [], []

# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(datasets):
    # load/create csv
    csv_name = str(ds_name)+"_"+str(input_shape[0])+"_"+split_str+".csv"
    if undersample:
        csv_name = str(ds_name)+"_"+str(input_shape[0])+"_"+split_str+"_us.csv"
    header = "["+str(ds_name)+"]"
    # TODO: default input for csv_file
    data_loader = DSManager(ds_name,
                            ds_root=ds_root,
                            image_size=input_shape, # changed for color (wxh) to (wxhxcs)
                            csv_file=csv_name,
                            color_flag=color_flag)
    try:
        csv_file = open(os.path.join(ds_root, csv_name))
    except:
        print(header + " File "+csv_name+" does not exist.")
        # print(header + " Loading original files.")
        # TODO: verify all dataset loading functions
        # data_loader.get_data()
    else:
        print(header + " File "+csv_name+" exists. Loading file...")
        data_loader.load_split_CSV()

    faces, emotions = (data_loader.faces, data_loader.emotions)
    # TODO: split_counts only valid for data from csv files
    split_counts = data_loader.split_counts
    del(data_loader)

    # TODO: again, works with the assumption that data is from csv
    if white_list_flag is True:
        print(header + " Removing expressions not in white list...")
        faces, emotions, split_counts = filter_labels(ds_name,
                                                      faces,
                                                      emotions,
                                                      split_counts)
        emotions = convert_labels(ds_name, emotions)

    print(header + " Preprocessing data...")
    emotions = pd.get_dummies(emotions).as_matrix()
    faces = preprocess_input(faces)

    # there is an issue with split counts in data_manager (depends on num_classes)
    print(header + " Grouping data...")
    train_faces, train_labels = [], []
    val_faces, val_labels = [], []
    test_faces, test_labels = [], []
    lindex = 0
    for i, split_count in enumerate(split_counts):
        maximum = sum(split_count)
        train_faces.extend(faces[lindex:lindex+split_count[0]])
        train_labels.extend(emotions[lindex:lindex+split_count[0]])
        val_faces.extend(faces[lindex+split_count[0]:
                               lindex+split_count[0]+split_count[1]])
        val_labels.extend(emotions[lindex+split_count[0]:
                                   lindex+split_count[0]+split_count[1]])
        test_faces.extend(faces[lindex+split_count[0]+split_count[1]:
                                lindex+maximum])
        test_labels.extend(emotions[lindex+split_count[0]+split_count[1]:
                                    lindex+maximum])
        lindex = lindex + maximum

    train_faces, train_labels = (np.asarray(train_faces),
                                 np.asarray(train_labels))
    val_faces, val_labels = (np.asarray(val_faces),
                             np.asarray(val_labels))
    test_faces, test_labels = (np.asarray(test_faces),
                               np.asarray(test_labels))

    train_faces, train_labels = shuffle_in_unison(train_faces, train_labels)
    val_faces, val_labels = shuffle_in_unison(val_faces, val_labels)
    test_faces, test_labels = shuffle_in_unison(test_faces, test_labels)

    if train_on_all is True:
        all_train_faces.extend(train_faces)
        all_train_labels.extend(train_labels)
        all_val_faces.extend(val_faces)
        all_val_labels.extend(val_labels)


    # DONT FORGET TO PREPROCESS
    model_loader = ModelManager(ds_name=ds_name, num_classes=num_classes,
                                img_shape=input_shape, split=split,
                                ds_root=ds_root, model_root=m_root)
    model_loader.train_data = (train_faces, train_labels)
    model_loader.val_data = (val_faces, val_labels)
    model_loader.test_data = (test_faces, test_labels)

    if weight_type == "bw":
        class_weights_dict = compute_weights(ds_name, train_labels)

    print(header + " Training...")
    model_loader.train(weight_type=weight_type,
                       class_weights=class_weights_dict)

    # del(model_loader)
    del(train_faces)
    del(train_labels)
    del(val_faces)
    del(val_labels)
    del(test_faces)
    del(test_labels)


if train_on_all is True:

    all_train_faces, all_train_labels = (np.asarray(all_train_faces),
                                         np.asarray(all_train_labels))
    all_val_faces, all_val_labels = (np.asarray(all_val_faces),
                                     np.asarray(all_val_labels))
    model_loader = ModelManager(ds_name="All", num_classes=num_classes,
                                img_shape=input_shape, split=split,
                                ds_root=ds_root, model_root=m_root)

    model_loader.train_data = (all_train_faces, all_train_labels)
    model_loader.val_data = (all_val_faces, all_val_labels)
    if weight_type == "bw":
        class_weights_dict = compute_weights(ds_name, all_train_labels)
    print("[All] Training...")
    model_loader.train(weight_type=weight_type,
                       class_weights=class_weights_dict)
