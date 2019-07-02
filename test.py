import os
import gc
import cv2
import settings
import datetime
import numpy as np
# # uncomment to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

from statistics import mean
from texttable import Texttable
from keras.models import load_model
from utils.data_manager import DSManager
from sklearn.metrics import accuracy_score
from utils.preprocessor import preprocess_input
from utils.common import (filter_labels, convert_labels,
                          shuffle_in_unison, get_labels)
from collections import Counter


m_paths = settings.MODEL_PATHS
model_ds_names = sorted(list(m_paths.keys()))
test_ds_names = settings.TEST_DATASET_NAMES
ds_names = set(model_ds_names + test_ds_names)
input_shape = settings.INPUT_SHAPE
num_classes = settings.NUM_CLASSES
ds_root = settings.DATASET_ROOT
bin_root = settings.BIN_ROOT
split = settings.SPLIT
split_str = str(int(split*100))
undersample = settings.UNDERSAMPLE_FLAG
weight_type = settings.WEIGHT_TYPE
train_on_all = settings.TRAIN_ON_ALL
test_on_all = settings.TEST_ON_ALL
m_paths = settings.MODEL_PATHS
white_list = settings.WHITE_LIST
white_list_flag = settings.WHITE_LIST_FLAG
color_flag = settings.COLOR_FLAG

# np.set_printoptions(threshold=np.nan)
m_root = settings.MODEL_ROOT
np.set_printoptions(linewidth=np.inf)

accuracies = []
ds_dict = {}


def keras_load(m_path):
    model = load_model(m_path)
    return model


def keras_test(model, images):
    probs = []
    preds = []
    for i, image in enumerate(images):
        image = np.asarray(image).reshape(input_shape)
        # TODO: color option
        # image = cv2.resize(image, input_shape)
        # image = cv2.resize(image, (64, 64))
        # image = image.astype('uint8')
        # image = preprocess_input(image, True)
        # TODO: why is this necessary if the image has already been expanded in load_split_csv?
        image = np.expand_dims(image, 0)
        # image = np.expand_dims(image, -1)
        pred = model.predict(image)
        probs.append(np.max(pred))
        preds.append(np.argmax(pred))

    return preds, probs


# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(ds_names):

    faces, emotions, model = ([], [], [])

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
                                image_size=input_shape,
                                csv_file=csv_name,
                                color_flag=color_flag)
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
        split_counts = data_loader.split_counts
        # TODO: consider creating public function for data loading
        #       so you don't have to delete the class every time
        del(data_loader)

        # if only test data will be used and not the whole dataset
        if not test_on_all:

            print(header+" Loading only test data portion...")
            test_faces, test_labels = ([], [])
            lindex = 0
            for j, split_count in enumerate(split_counts):
                maximum = sum(split_count)
                test_faces.extend(faces[lindex+split_count[0]+split_count[1]:
                                        lindex+maximum])
                test_labels.extend(emotions[lindex+split_count[0]+split_count[1]:
                                            lindex+maximum])
                lindex = lindex + maximum

            faces, emotions = (np.asarray(test_faces), np.asarray(test_labels))
            del(test_faces)
            del(test_labels)

        if white_list_flag is True:
            print(header + " Removing expressions not in white list...")
            faces, emotions, split_counts = filter_labels(ds_name,
                                                          faces,
                                                          emotions,
                                                          split_counts)
            emotions = convert_labels(ds_name, emotions)

        faces = preprocess_input(faces)

    if ds_name in model_ds_names:
        header = "["+str(ds_name)+"]"
        m_path = os.path.join(m_root, m_paths[ds_name])
        model = keras_load(m_path)

    # values can be empty if model is not to be loaded
    ds_dict[ds_name] = [faces,
                        emotions,
                        model,
                        {}]
    del(faces)
    del(emotions)
    del(model)


print("---- Multi-labelling datasets")
for model_ds_name in model_ds_names:

    header = "["+str(model_ds_name)+"]"
    model = ds_dict[model_ds_name][2]

    for test_ds_name in test_ds_names:

        faces, emotions = (ds_dict[test_ds_name][0], ds_dict[test_ds_name][1])
        print(header+" Creating predictions for "+str(test_ds_name)+" data...")
        preds, probs = keras_test(model, faces)
        accuracy = accuracy_score(emotions, preds)
        ds_dict[test_ds_name][3][model_ds_name] = [preds, probs, accuracy]

    del(model)
    # del(ds_dict[model_ds_name][2])
    gc.collect()


print("---- Saving all labels to csvs")
for i, test_ds_name in enumerate(test_ds_names):
    header = "["+str(test_ds_name)+"]"
    print(header+" Saving actual labels and "+test_ds_name+" predictions...")

    # initialize labels and probs list
    all_labels = ds_dict[test_ds_name][1]
    all_probs = []
    columns = ["actual"]

    for j, model_ds_name in enumerate(model_ds_names):

        print(header+" Saving "+model_ds_name+" predictions...")
        columns.append(model_ds_name)
        preds = ds_dict[test_ds_name][3][model_ds_name][0]
        probs = ds_dict[test_ds_name][3][model_ds_name][1]
        all_labels = [str(k)+","+str(preds[iter]) for iter, k in enumerate(all_labels)]
        if all_probs == []:
            all_probs.extend(probs)
        else:
            all_probs = [str(k)+","+str(probs[iter]) for iter, k in enumerate(all_probs)]

    now = datetime.datetime.now()
    preds_f = "%s/%s_preds_%s_%s_%d-%d-%d-%d%d.csv" % (bin_root, test_ds_name,
                                                       split_str, weight_type,
                                                       now.year, now.month,
                                                       now.day, now.hour,
                                                       now.minute)
    probs_f = "%s/%s_probs_%s_%s_%d-%d-%d-%d%d.csv" % (bin_root, test_ds_name,
                                                       split_str, weight_type,
                                                       now.year, now.month,
                                                       now.day, now.hour,
                                                       now.minute)
    # open multi-labelled preds csv
    print(header+" Saving predictions to " +
          preds_f + ".\n Please wait...")
    f = open(preds_f, "w")
    temp = ','.join(map(str, columns))
    f.write(temp+"\n")
    for i, line in enumerate(all_labels):
        f.write(line+"\n")
    f.close()
    print(header+" Preds csv saved.")
    del(all_labels)
    # close multi-labelled preds csv

    # open probs csv
    print(header+" Saving probabilities to " +
          probs_f + ".\n Please wait...")
    f = open(probs_f, "w")
    temp = ','.join(map(str, columns[1:]))
    f.write(temp+"\n")
    for i, line in enumerate(all_probs):
        f.write(str(line)+"\n")
    f.close()
    print(header+" Probs csv saved.")
    del(all_probs)
    del(columns)
    # close probs csv


table = Texttable()
alignment = ["l", "c"] # additional two for the row headers and average at end
alignment.extend(["c" for name in test_ds_names])
table.set_cols_align(alignment)
alignment = ["m", "m"]  # additional two for the row headers and average at end
alignment.extend(["m" for name in test_ds_names])
table.set_cols_valign(alignment)

column_headers = [r"MODEL\DATA"]
column_headers.extend([name for name in test_ds_names]+["AVERAGE"])
table.add_row(column_headers)
for model_ds_name in model_ds_names:
    row = [model_ds_name]
    accuracies = []
    for test_ds_name in test_ds_names:
        accuracies.append(ds_dict[test_ds_name][3][model_ds_name][2])
        row.append(accuracies[-1]*100)
    row.append(mean(accuracies)*100)
    table.add_row(row)

print("---- Results:")
print("\n"+table.draw()+"\n\n")
# saving results to separate file
f_name = "%s/results_summary_%d-%d-%d-%d%d.txt" % (bin_root, now.year,
                                                   now.month, now.day,
                                                   now.hour, now.minute)
f = open(f_name, "w")
f.write(table.draw() + "\n\n")

f.write("SETTINGS:\n")
f.write("> input shape - %s\n" % (input_shape,))
f.write(">       split - %s\n" % (split_str))
f.write("> undersample - %s\n" % (undersample))
f.write("> weight type - %s\n" % (weight_type))

if white_list_flag is True:
    f.write("> expressions - %s\n" % (white_list))
else:
    f.write("> expressions - %s\n" % (get_labels("FER2013").values()))

if test_on_all:
    f.write(">   test data - whole datasets used\n\n")
else:
    f.write(">   test data - only test portions used\n\n")
f.write("MODEL FILES:\n")
for m_path in m_paths.values():
    f.write(m_path+"\n")

f.close()
print("---- Saved results to %s" % (f_name))

del(ds_dict)
