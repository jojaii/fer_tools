"""
File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""

import os
from utils.data_manager import DSManager
import numpy as np
from utils.preprocessor import preprocess_input
from keras.models import load_model
from keras import Model
import cv2
import time


def keras_layer_load(base_model, layer_name):
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
    # conv2d_7
    # add_4
    return model


def keras_load(m_path):
    model = load_model(m_path, compile=False)
    return model


def keras_test(model, data):
    images = data[0]
    probs = []
    preds = []
    for i, image in enumerate(images):
        image = cv2.resize(image, (64, 64))
        # image = preprocess_input(image, True)
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, -1)
        pred = model.predict(image)
        probs.append(np.max(pred))
        preds.append(np.argmax(pred))

    return preds, probs


def keras_embed(model, images):
    preds = []
    for i, image in enumerate(images):
        image = cv2.resize(image, (64, 64))
        # image = preprocess_input(image, True)
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, -1)
        pred = model.predict(image)
        if preds == []:
            preds = pred
        else:
            preds = np.vstack((preds, pred))

    return preds


np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.inf)
ds_root = '/home/ubuntu/Desktop'
ds_names = ['AffectNet', 'ExpW', 'FER2013', 'KDEF']
m_root = "models"
m_names = []
m_paths = ["AffectNet_64_10_bw_mini_XCEPTION.30-0.68.hdf5",
           "ExpW_64_10_bw_mini_XCEPTION.13-0.56.hdf5",
           "FER2013_64_10_bw_mini_XCEPTION.73-0.64.hdf5",
           "KDEF_64_10_bw_mini_XCEPTION.47-0.22.hdf5"]
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
    m_path = os.path.join(m_root, m_paths[i])
    data_loader.faces = preprocess_input(data_loader.faces)
    base = keras_load(m_path)
    add_4 = keras_layer_load(base, "add_4")
    conv2d_7 = keras_layer_load(base, "conv2d_7")

    print(header+" Generating embeddings for add_4 layer...")
    embeddings_1 = keras_embed(add_4, data_loader.faces)
    print(header+" Generating embeddings for conv2d_7 layer...")
    embeddings_2 = keras_embed(conv2d_7, data_loader.faces)

    # open embeddings csv
    f_name = "bin/"+ds_name+"_embds_"+str(int(split*100))+"_"+type+".csv"
    print(header+" Saving embeddings to " +
          f_name + ".\n Please wait...")
    f = open(f_name, "w")
    columns = "add_4,conv2d_7"
    f.write(columns+"/n")
    for j, embedding_1 in enumerate(embeddings_1):
        embedding_1 = ' '.join(map(str,
                               (embedding_1.astype(int)).flatten()))
        embedding_2 = ' '.join(map(str,
                               (embeddings_2[j].astype(int)).flatten()))
        line = embedding_1+","+embedding_2
        f.write(line+"\n")
    f.close()
    print(header+" Embeddings csv saved.")
    # close multi-labelled csv

    del(data_loader)
    del(embeddings_1)
    del(embeddings_2)
    del(base)
    del(add_4)
    del(conv2d_7)
