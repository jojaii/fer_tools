import os
import gc
import cv2
import settings
import datetime
import numpy as np
# # uncomment to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import onnx
import caffe2.python.onnx.backend

# from statistics import mean
# from texttable import Texttable
# from keras.models import load_model
from utils.data_manager import DSManager
from sklearn.metrics import accuracy_score
from utils.preprocessor import preprocess_input
from utils.common import (filter_labels, convert_labels,
                          shuffle_in_unison, get_labels)


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

# np.set_printoptions(threshold=np.nan)
m_root = settings.MODEL_ROOT
np.set_printoptions(linewidth=np.inf)

accuracies = []
ds_dict = {}
fer_model_path = '/home/ubuntu/Repos/onnx_inference/minixception_66_wgi.onnx'
onnx_model = onnx.load(fer_model_path)


def onnx_test(onnx_model, images):
    probs = []
    preds = []
    for i, image in enumerate(images):
        image = cv2.resize(image, (64, 64))
        # image = preprocess_input(image, True)
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, -1)
        pred = caffe2.python.onnx.backend.run_model(onnx_model, [image])
        probs.append(np.max(pred))
        preds.append(np.argmax(pred))

    return preds, probs

def keras_load(m_path):
    model = load_model(m_path)
    return model


def keras_test(model, images):
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
        split_counts = data_loader.split_counts
        # TODO: consider creating public function for data loading
        #       so you don't have to delete the class every time
        del(data_loader)

        if white_list_flag is True:
            print(header + " Removing expressions not in white list...")
            faces, emotions = filter_labels(ds_name,
                                            faces,
                                            emotions)
            emotions, split_counts = convert_labels(ds_name,
                                                    emotions,
                                                    split_counts)

        # if only test data will be used and not the whole dataset
        if not test_on_all:
            print(header+" Loading only test data portion...")
            for j, split_count in enumerate(split_counts):
                maximum = sum(split_count)
                if j == 0:
                    test_faces = faces[split_count[0]+split_count[1]:maximum]
                    test_labels = emotions[split_count[0]+split_count[1]:maximum]
                    lindex = maximum
                else:
                    test_faces = np.vstack((test_faces, faces[lindex+split_count[0]+split_count[1]:lindex+maximum]))
                    test_labels = np.hstack((test_labels, emotions[lindex+split_count[0]+split_count[1]:lindex+maximum]))
                    lindex = lindex + maximum
            # test_faces, test_labels = shuffle_in_unison(test_faces, test_labels)
            faces, emotions = (test_faces, test_labels)
            del(test_faces)
            del(test_labels)

        faces = preprocess_input(faces)

    # if ds_name in model_ds_names:
    #     header = "["+str(ds_name)+"]"
    #     m_path = os.path.join(m_root, m_paths[ds_name])
    #     model = keras_load(m_path)

    # values can be empty if model is not to be loaded
    ds_dict[ds_name] = [faces,
                        emotions,
                        model,
                        {}]
    del(faces)
    del(emotions)
    del(model)


# print("---- Multi-labelling datasets")
# for model_ds_name in model_ds_names:

#     header = "["+str(model_ds_name)+"]"
#     model = ds_dict[model_ds_name][2]

accuracies = []
for test_ds_name in test_ds_names:

    faces, emotions = (ds_dict[test_ds_name][0], ds_dict[test_ds_name][1])
    print(header+" Creating predictions for "+str(test_ds_name)+" data...")
    preds, probs = onnx_test(onnx_model, faces)
    accuracy = accuracy_score(emotions, preds)
    accuracies.append(accuracy)
    print("%s: %.3f" % (test_ds_name, accuracy*100))
    # ds_dict[test_ds_name][3][model_ds_name] = [preds, probs, accuracy]

    # del(model)
    # # del(ds_dict[model_ds_name][2])
    # gc.collect()

print("\nSUMMARY")
# print(accuracies)
for i, test_ds_name in enumerate(test_ds_names):
    print("%s: %.2f" % (test_ds_name, 100*(accuracies[i])))
