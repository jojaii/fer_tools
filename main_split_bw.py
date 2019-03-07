"""
LOAD
SHUFFLE
GROUP
UNDESAMPLE
SPLIT
SAVE TO CSV

File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import os
# from model_manager import ModelManager
from utils.data_manager import DSManager
from utils.common import shuffle_in_unison, strat_sample, group, truncate, combine_classes, generate_csv, count_labels

# parameters

# data generator
data_generator = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=.1,
                                    horizontal_flip=True)

datasets = ['AffectNet', 'FER2013', 'KDEF', 'ExpW']
# datasets = ['AffectNet']
# datasets = ['KDEF']#, 'ExpW']
input_shape = (64, 64, 1)
# model_manager = ModelManager(datasets,
#                              input_shape)
num_classes = 7
ds_root = '/home/ubuntu/Datasets'
ds_dict = {}
split = 0.1  # TODO: should be changeable by arg
# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(datasets):
    # load/create csv
    csv_name = str(ds_name) + "_" + str(input_shape[0]) + ".csv"
    header = "["+str(ds_name)+"]"
    # TODO: default input for csv_file
    data_loader = DSManager(ds_name,
                            ds_root=ds_root,
                            image_size=input_shape[:2],
                            csv_file=csv_name)
    try:
        csv_file = open(os.path.join(ds_root, ds_name, csv_name))
    except:
        print(header + " File "+csv_file+" does not exist. \
              Loading and converting to csv.")
        print(header + " Loading and converting to csv \
              with image size " + str(input_shape[:2]))
        data_loader.get_data()
        data_loader.generate_csv()
    else:
        print(header + " File "+csv_name+" exists. Loading file...")
        data_loader.load_CSV()

# SHUFFLE
    print(header + " Shuffling images and labels simultaneously...")
    (shuf_faces, shuf_emotions) = shuffle_in_unison(data_loader.faces,
                                                    data_loader.emotions)

# GROUP (according to class)
    grouped_faces = group(shuf_faces, shuf_emotions, num_classes)
    # print(np.shape(grouped_faces))
    # exit()
# UNDERSAMPLE
    # count labels first
    labels_count = count_labels(shuf_emotions)
    minimum = min(labels_count)
    print(str(labels_count)+"!!!!!!!!!!!!!")
    #grouped_faces = truncate(grouped_faces, minimum)

# SPLIT get int values list for split
    classes_split_labels, classes_split_counts = strat_sample(grouped_faces,
                                                              split)

    # combine classes, remove groupings
    combined_faces, labels = combine_classes(grouped_faces)

    csv_name = "/home/ubuntu/Desktop/"+ds_name+"_"+str(input_shape[0])+"_"+str(int(split*100))+"_bw.csv"
    generate_csv(ds_name, csv_name, num_classes, combined_faces, labels, classes_split_labels, classes_split_counts)

    del(data_loader)
    del(shuf_faces)
    del(shuf_emotions)
    del(grouped_faces)
    del(labels_count)
    del(minimum)
    del(classes_split_labels)
    del(classes_split_counts)
    del(combined_faces)
    del(labels)
    

# SAVE TO CSV
    # column 1: emotion
    # column 2: face
    # column 3: group (train, test, val as 0, 1 and 2)
    # column 4: class count! (array? how best to save this?)


