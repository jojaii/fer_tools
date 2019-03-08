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

ds_root = '/home/ubuntu/Desktop'
ds_names = ['AffectNet', 'FER2013', 'KDEF', 'ExpW']
m_root = ""
m_names = []
fer_path = "/home/ubuntu/Repos/face_classification/trained_models/undersampled"
m_names = ["KDEF_mini_XCEPTION.158-1.00.hdf5",
           "ExpW_mini_XCEPTION.36-1.00.hdf5"]

# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(ds_names):
    # load/create csv

    csv_name = str(ds_name)+"_"+str(input_shape[0])+"_"+str(int(split*100))+"_us.csv"
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

    train_faces, train_labels = shuffle_in_unison(train_faces, train_labels)
    val_faces, val_labels = shuffle_in_unison(val_faces, val_labels)
    test_faces, test_labels = shuffle_in_unison(test_faces, test_labels)

    ## DONT FORGET TO PREPROCESS
    model_loader = ModelManager(ds_name)
    model_loader.train_data = (train_faces, train_labels)
    model_loader.val_data = (val_faces, val_labels)
    model_loader.test_data = (test_faces, test_labels)

    model_loader.test()

    del(model_loader)

    # Mmanager.test()
    del(data_loader)

    del(train_faces)
    del(train_labels)
    del(val_faces)
    del(val_labels)
    del(test_faces)
    del(test_labels)


from keras.models import load_model

# TESTING SECTION
# for i, dataset in enumerate(datasets):
# load model
# open file
# for all data inputted to model
# get classification
# print actual then predicted on file
# close filie

for i, m_name in enumerate(m_names):
    m_path = os.path.join(m_root, m_name)
    label_refs = get_labels(ds_names[i])

    # move to model manager
    emotion_classifier = load_model(m_path, compile=False)
    emotion_target_size = emotion_classifier.input_shape[1:3]

    for j, test_dataset in enumerate(datasets):
        if j == i:
            continue

        f = open(datasets[i]+"_"+datasets[j]+".txt", 'w')

        for k, image in enumerate(dataset_dict[datasets[j]][0]):    

            try:
                gray_face = cv2.resize(image, (emotion_target_size))
            except:
                continue


            gray_face = preprocess_input(gray_face, True)

            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)  # returns probability value
            label_ref_arg = np.argmax(emotion_prediction)  # returns index of highest probability value

            actual_label = int(np.argwhere(dataset_dict[datasets[j]][1][k]==1))
            pred_label = label_ref_arg
#             f.write(str(pred_label)+","+str(actual_label)+"\n")
            print(str(pred_label)+","+str(actual_label))
        f.close
    del(dataset_dict[dataset[i]][2])
    gc.collect()