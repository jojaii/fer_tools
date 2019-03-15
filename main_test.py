"""
File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""

import os
# from utils.model_manager import ModelManager
from utils.data_manager import DSManager
from utils.common import shuffle_in_unison
# from sklearn.utils import class_weight
import numpy as np
# import pandas as pd
from utils.preprocessor import preprocess_input
from keras.models import load_model
import cv2
import time
# import tensorflow as tf

def keras_load(m_path):
    model = load_model(m_path)
    return model


def keras_test(model, data):
    correct = 0
    images = data[0]
    actual = data[1]
    sum_time = 0
    for i, image in enumerate(images):
        image = cv2.resize(image, (64, 64))
        # image = preprocess_input(image, True)
        start = time.time()
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, -1)
        prediction = model.predict(image)
        prediction = np.argmax(prediction)
        sum_time += time.time() - start
        if prediction == actual[i]:
            correct += 1

    accuracy = correct/len(images)
    print("keras latency", (sum_time/(len(images))))

    return accuracy


ds_root = '/home/ubuntu/Desktop'
ds_names = ['AffectNet', 'ExpW', 'FER2013', 'KDEF']
# ds_names = ['FER2013']
m_root = "models"
m_names = []
fer_path = "/home/ubuntu/Repos/face_classification/trained_models/undersampled"
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

input_shape = (64, 64)
split = .1
type = "bw"
accuracies = []
ds_dict = {}

# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(ds_names):
    # load/create csv

    csv_name = str(ds_name)+"_"+str(input_shape[0])+"_"+str(int(split*100))+"_"+type+".csv"
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
    data_loader.faces = preprocess_input(data_loader.faces)
    ds_dict[ds_name] = [data_loader.faces,
                        data_loader.emotions,
                        mini_XCEPTION(input_shape, num_classes)]
    # for j, split_count in enumerate(data_loader.split_counts):
    #     maximum = sum(split_count)
    #     if j == 0:
    #         # train_faces = data_loader.faces[:split_count[0]]
    #         # train_labels = data_loader.emotions[:split_count[0]]
    #         # val_faces = data_loader.faces[split_count[0]:split_count[0]+split_count[1]]
    #         # val_labels = data_loader.emotions[split_count[0]:split_count[0]+split_count[1]]
    #         test_faces = data_loader.faces[split_count[0]+split_count[1]:maximum]
    #         test_labels = data_loader.emotions[split_count[0]+split_count[1]:maximum]
    #         lindex = maximum
    #     else:
    #         # train_faces = np.vstack((train_faces, data_loader.faces[lindex:lindex+split_count[0]]))
    #         # train_labels = np.vstack((train_labels, data_loader.emotions[lindex:lindex+split_count[0]]))
    #         # val_faces = np.vstack((val_faces, data_loader.faces[lindex+split_count[0]:lindex+split_count[0]+split_count[1]]))
    #         # val_labels = np.vstack((val_labels, data_loader.emotions[lindex+split_count[0]:lindex+split_count[0]+split_count[1]]))
    #         test_faces = np.vstack((test_faces, data_loader.faces[lindex+split_count[0]+split_count[1]:lindex+maximum]))
    #         test_labels = np.hstack((test_labels, data_loader.emotions[lindex+split_count[0]+split_count[1]:lindex+maximum]))
    #         lindex = lindex + maximum

    # # train_faces, train_labels = shuffle_in_unison(train_faces, train_labels)
    # # val_faces, val_labels = shuffle_in_unison(val_faces, val_labels)
    # test_faces, test_labels = shuffle_in_unison(test_faces, test_labels)
    # test_data = (test_faces, test_labels)
    
    # KERAS TEST   
    # /home/ubuntu/Repos/fer_tools/models/ 
    m_path = os.path.join(m_root, m_paths[i])
    model = keras_load(m_path)
    accuracies.append(keras_test(model, test_data))

    # ## DONT FORGET TO PREPROCESS
    # model_loader = ModelManager(ds_name)
    # # model_loader.train_data = (train_faces, train_labels)
    # # model_loader.val_data = (val_faces, val_labels)
    # model_loader.test_data = (test_faces, test_labels)

    # model_loader.test()

    # del(model_loader)

    # # Mmanager.test()
    del(data_loader)

    # # del(train_faces)
    # # del(train_labels)
    # # del(val_faces)
    # # del(val_labels)
    del(test_faces)
    del(test_labels)

print(accuracies)

# from keras.models import load_model

# # TESTING SECTION
# # for i, dataset in enumerate(datasets):
# # load model
# # open file
# # for all data inputted to model
# # get classification
# # print actual then predicted on file
# # close filie

# for i, m_name in enumerate(m_names):
#     m_path = os.path.join(m_root, m_name)
#     label_refs = get_labels(ds_names[i])

#     # move to model manager
#     emotion_classifier = load_model(m_path, compile=False)
#     emotion_target_size = emotion_classifier.input_shape[1:3]

#     for j, test_dataset in enumerate(datasets):
#         if j == i:
#             continue

#         f = open(datasets[i]+"_"+datasets[j]+".txt", 'w')

#         for k, image in enumerate(dataset_dict[datasets[j]][0]):    

#             try:
#                 gray_face = cv2.resize(image, (emotion_target_size))
#             except:
#                 continue


#             gray_face = preprocess_input(gray_face, True)

#             emotion_prediction = emotion_classifier.predict(gray_face)
#             emotion_probability = np.max(emotion_prediction)  # returns probability value
#             label_ref_arg = np.argmax(emotion_prediction)  # returns index of highest probability value

#             actual_label = int(np.argwhere(dataset_dict[datasets[j]][1][k]==1))
#             pred_label = label_ref_arg
# #             f.write(str(pred_label)+","+str(actual_label)+"\n")
#             print(str(pred_label)+","+str(actual_label))
#         f.close
#     del(dataset_dict[dataset[i]][2])
#     gc.collect()