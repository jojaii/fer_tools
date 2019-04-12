"""
File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""
import os
import cv2
import settings
import datetime
# # uncomment to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

from utils.data_manager import DSManager
import numpy as np
from utils.preprocessor import preprocess_input
from keras.models import load_model
from utils.common import filter_labels, convert_labels


m_paths = settings.MODEL_PATHS
model_ds_names = list(m_paths.keys())
test_ds_names = settings.TEST_DATASET_NAMES
ds_names = model_ds_names + test_ds_names
input_shape = settings.INPUT_SHAPE
num_classes = settings.NUM_CLASSES
ds_root = settings.DATASET_ROOT
bin_root = settings.BIN_ROOT
split = settings.SPLIT
split_str = str(int(split*100))
undersample = settings.UNDERSAMPLE_FLAG
weight_type = settings.WEIGHT_TYPE
train_on_all = settings.TRAIN_ON_ALL
m_paths = settings.MODEL_PATHS
# np.set_printoptions(threshold=np.nan)
m_root = settings.MODEL_ROOT
np.set_printoptions(linewidth=np.inf)

accuracies = []
ds_dict = {}

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


# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(ds_names):

    if ds_name in test_ds_names:

        # load/create csv
        if undersample:
            csv_name = str(ds_name)+"_"+str(input_shape[0])+"_" + \
                    str(int(split*100))+"_us.csv"
        else:
            csv_name = str(ds_name)+"_"+str(input_shape[0])+"_" + \
                    str(int(split*100))+".csv"

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
        faces, emotions = (data_loader.faces, data_loader.emotions)
        faces, emotions, _ = filter_labels(ds_name, faces, emotions)
        emotions = convert_labels(ds_name, emotions)
        ds_dict[ds_name] = [preprocess_input(faces),
                            emotions,
                            [],
                            {}]
        print(header + " Removing expressions not in white list...")
        del(data_loader)
    if ds_name in model_ds_names:
        header = "["+str(ds_name)+"]"
        m_path = os.path.join(m_root, m_paths[ds_name])
        ds_dict[ds_name] = [[],
                            [],
                            keras_load(m_path)]



print("---- Multi-labelling datasets")
for i, ds_model in enumerate(ds_names):
    ds_data = "SFEW"
    if ds_model != ds_data:
        header = "["+str(ds_model)+"]"
        print(header+" Creating predictions for "+str(ds_data)+" data...")
        test_data = (ds_dict[ds_data][0], ds_dict[ds_data][1])
        preds, probs = keras_test(ds_dict[ds_model][2],
                                  test_data)
        # save labels to ds_name2 since it is the source of the data
        ds_dict[ds_data][3][ds_model] = (preds, probs)


print("---- Saving all labels to csvs")
ds_name = "SFEW"
header = "["+str(ds_name)+"]"
print(header+" Saving actual labels and "+ds_name+" predictions...")
all_labels = ds_dict[ds_name][1]
all_probs = []
columns = ["actual"]
for j, ds_name2 in enumerate(ds_names):
    if ds_name2 != ds_name:
        print(header+" Saving "+ds_name2+" predictions...")
        columns.append(ds_name2)
        preds = ds_dict[ds_name][3][ds_name2][0]
        probs = ds_dict[ds_name][3][ds_name2][1]
        all_labels = [str(k)+","+str(preds[iter]) for iter, k in enumerate(all_labels)]
        if all_probs == []:
            all_probs.extend(probs)
        else:
            all_probs = [str(k)+","+str(probs[iter]) for iter, k in enumerate(all_probs)]

# open multi-labelled csv
now = datetime.datetime.now()
f_name = "%s/%s_preds_%d_%s_%d-%d-%d-%d.csv" % (bin_root, ds_name,
                                                int(split*100), weight_type,
                                                now.year, now.month,
                                                now.hour, now.minute)

print(header+" Saving all predictions to " +
      f_name + ".\n Please wait...")
f = open(f_name, "w")
temp = ','.join(map(str, columns))
f.write(temp+"\n")
for i, line in enumerate(all_labels):
    f.write(line+"\n")
f.close()
print(header+" Preds csv saved.")
del(all_labels)
# close multi-labelled csv

# open probs csv
f_name = "%s/%s_probs_%d_%s_%d-%d-%d-%d.csv" % (bin_root, ds_name,
                                                int(split*100), weight_type,
                                                now.year, now.month,
                                                now.hour, now.minute)
# f_name = "bin/"+ds_name+"_probs_"+str(int(split*100))+"_"+weight_type+".csv"

print(header+" Saving all probabilities to " +
      f_name + ".\n Please wait...")
f = open(f_name, "w")
temp = ','.join(map(str, columns[1:]))
f.write(temp+"\n")
for i, line in enumerate(all_probs):
    f.write(str(line)+"\n")
f.close()
print(header+" Probs csv saved.")
del(all_probs)
del(columns)
# close probs csv

del(ds_dict)
