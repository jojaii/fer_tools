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

batchsize = 64
src_size = 64
input_data_str = 
def get_emoti_and_loss_from_batch(net, imgpaths, gtlbls,
                                  output_str='output', nclass=7):

    for i in range(batchsize):
        if i >= len(imgpaths):
            # can this load from numpy image?
            # or should entry always be a directory
            image = caffe.io.load_image(imgpaths[-1])
            # simply returns image with type np.float32 with size HxWx1
            self.net.blobs["label"].data[i] = gtlbls[-1]
        else:
            image = caffe.io.load_image(imgpaths[i])
            self.net.blobs["label"].data[i] = gtlbls[i]
        image = caffe.io.resize_image(image, self.src_size)
        image = self.crop_center(image)
        transformed_img = self.transformer.preprocess(self.input_data_str, image)
        self.net.blobs[self.input_data_str].data[i] = transformed_img
    self.net.forward()
    prob = self.net.blobs[output_str].data
    prob = self.reshape_feature(prob)
    is_predict_correct = []
    for i in range(len(imgpaths)):
        is_predict_correct.append(0)
        bidx = i*nclass
        eidx = (i+1)*nclass
        is_predict_correct[i] = 1 if prob[i*nclass+gtlbls[i]]==max(prob[bidx:eidx]) else 0
    loss = self.net.blobs["loss"].data
    loss = self.reshape_feature(loss)
    return is_predict_correct, loss[0]


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
    for j, split_count in enumerate(data_loader.split_counts):
        maximum = sum(split_count)
        if j == 0:
            # val_faces = data_loader.faces[split_count[0]:split_count[0]+split_count[1]]
            # val_labels = data_loader.emotions[split_count[0]:split_count[0]+split_count[1]]
            test_faces = data_loader.faces[split_count[0]+split_count[1]:maximum]
            test_labels = data_loader.emotions[split_count[0]+split_count[1]:maximum]
            lindex = maximum
        else:
            # val_faces = np.vstack((val_faces, data_loader.faces[lindex+split_count[0]:lindex+split_count[0]+split_count[1]]))
            # val_labels = np.vstack((val_labels, data_loader.emotions[lindex+split_count[0]:lindex+split_count[0]+split_count[1]]))
            test_faces = np.vstack((test_faces, data_loader.faces[lindex+split_count[0]+split_count[1]:lindex+maximum]))
            test_labels = np.hstack((test_labels, data_loader.emotions[lindex+split_count[0]+split_count[1]:lindex+maximum]))
            lindex = lindex + maximum

    # val_faces, val_labels = shuffle_in_unison(val_faces, val_labels)
    test_faces, test_labels = shuffle_in_unison(test_faces, test_labels)
    test_data = (test_faces, test_labels)
    
    # KERAS TEST   
    # /home/ubuntu/Repos/fer_tools/models/ 
    m_path = os.path.join(m_root, m_paths[i])
    model = keras_load(m_path)
    accuracies.append(keras_test(model, test_data))

    del(data_loader)

    # # del(val_faces)
    # # del(val_labels)
    del(test_faces)
    del(test_labels)

print(accuracies)
