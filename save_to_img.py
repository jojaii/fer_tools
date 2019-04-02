"""

File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""

# # uncomment to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import os
from utils.data_manager import DSManager
import numpy as np
from utils.preprocessor import preprocess_input
from keras.models import load_model
import cv2
from utils.common import get_labels
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imsave

def keras_load(m_path):
    model = load_model(m_path)
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


# np.set_printoptions(threshold=np.nan)
# np.set_printoptions(linewidth=np.inf)
ds_root = 'data'
ds_names = ['AffectNet', 'ExpW', 'FER2013']#, 'KDEF']#, 'SFEW']
# ds_names = ['FER2013']#, 'KDEF']#, 'SFEW']

input_shape = (64, 64)
split = .2
type = "rw"
accuracies = []
ds_dict = {}
f_root = "bin"
num_classes = 7
file_list = open("all_images.txt", "w")
labels = get_labels("FER2013")

# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(ds_names):
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

    ds_dict[ds_name] = [data_loader.faces,
                        {}]
    ds_dict[ds_name][1][ds_name] = data_loader.emotions

    print("---- Acquiring multiple datasets")
    f_preds = ds_name+"_preds_"+str(int(split*100))+"_"+type+".csv"
    preds = pd.read_csv(os.path.join(f_root, f_preds))
    m_names = ds_names.copy()
    m_names.remove(ds_name)
    for j, m_pred in enumerate(m_names):
        print(header+" Loading predictions of "+str(m_pred)+" to dict...")
        ds_dict[ds_name][1][m_pred] = [int(x) for x in preds[m_pred].tolist()]

    # Labels from model trained on consolidated datasets ("All")
    f_preds = ds_name+"_preds_"+str(int(split*100))+"_All.csv"
    preds = pd.read_csv(os.path.join(f_root, f_preds))
    m_pred = "All"
    print(header+" Loading predictions of "+str(m_pred)+" to dict...")
    ds_dict[ds_name][1][m_pred] = [int(x) for x in preds[m_pred].tolist()]

    folder = os.path.join("data", ds_name)
    for label in range(num_classes):
        if not os.path.exists(os.path.join(folder, str(label))):
            os.makedirs(os.path.join(folder, str(label)))
    data = ds_dict[ds_name]
    faces = data_loader.faces
    emotions = data[1]
    for j, face in enumerate(faces):
        folder_2 = os.path.join(folder, str(emotions[ds_name][j]))
        face = np.squeeze(face, axis=-1)
        filename = os.path.join(folder_2, str(j)+".jpg")
        plt.imsave(filename, face, cmap="gray")
        file_list.write("%s %d %d %d %d\n" % (filename,
                                              emotions["AffectNet"][j],
                                              emotions["FER2013"][j],
                                              emotions["ExpW"][j],
                                              emotions["All"][j]))
                                           #  data.emotions["KDEF"]))
                                           # TODO: are the images from KDEF reliable enough
                                           # to be added here? check for future
    del(data_loader)

file_list.close()