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
import pandas as pd
from utils.preprocessor import preprocess_input
from utils.common import shuffle_in_unison
from keras.models import load_model
import cv2
import tensorflow as tf
import time

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


def tf_run(m_path, data):
    pb_file = m_path
    detectionGraph = tf.Graph()
    with detectionGraph.as_default():
        odGraphDef = tf.GraphDef()
        with tf.gfile.GFile(pb_file, 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

    with detectionGraph.as_default():
        config = tf.ConfigProto()  # device_count = {'GPU': 0}
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=detectionGraph, config=config)

    # imageTensor = detectionGraph.get_tensor_by_name('input_2:0')
    # scores = detectionGraph.get_tensor_by_name('predictions/Softmax:0')

    correct = 0
    images = data[0]
    actual = data[1]
    sum_time = 0
    for i, image in enumerate(images):
        image = cv2.resize(image, (64, 64))
        # image = preprocess_input(image, True)

        start = time.time()
        imageTensor = detectionGraph.get_tensor_by_name('input_2:0')
        scores = detectionGraph.get_tensor_by_name('predictions/Softmax:0')

        
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, -1)
        (scores) = sess.run([scores],
                            feed_dict={imageTensor:image})
        prediction = np.argmax(scores)
        sum_time += time.time() - start
    
        if prediction == actual[i]:
            correct += 1
    print("tf latency (seconds/image)", (sum_time/len(images)))

    # print("correct:"+str(correct))
    # print("len(data):"+str(len(data)))
    # exit()
    accuracy = correct/len(images)

    return accuracy

# def tf_load(m_path):
#     pb_file = m_path
#     detectionGraph = tf.Graph()
#     with detectionGraph.as_default():
#         odGraphDef = tf.GraphDef()
#         with tf.gfile.GFile(pb_file, 'rb') as fid:
#             serializedGraph = fid.read()
#             odGraphDef.ParseFromString(serializedGraph)
#             tf.import_graph_def(odGraphDef, name='')

#     with detectionGraph.as_default():
#         config = tf.ConfigProto()  # device_count = {'GPU': 0}
#         config.gpu_options.allow_growth = True
#         sess = tf.Session(graph=detectionGraph, config=config)

#     imageTensor = detectionGraph.get_tensor_by_name('input_2:0')
#     scores = detectionGraph.get_tensor_by_name('predictions/Softmax:0')

#     return sess, imageTensor, scores


# def tf_test(sess, data, imageTensor, scores):
#     correct = 0
#     images = data[0]
#     actual = data[1]
#     for i, image in enumerate(images):
#         image = cv2.resize(image, (64, 64))
#         # image = preprocess_input(image, True)

#         imageTensor = detectionGraph.get_tensor_by_name('input_2:0')
#         scores = detectionGraph.get_tensor_by_name('predictions/Softmax:0')

#         image = np.expand_dims(image, 0)
#         image = np.expand_dims(image, -1)
#         (scores) = sess.run([scores],
#                             feed_dict={imageTensor:image})
#         prediction = np.argmax(scores)
#         if prediction == actual[i]:
#             correct += 1

#     accuracy = correct/len(data)

#     return accuracy

ds_root = '/home/ubuntu/Desktop'
ds_names = ['FER2013']
input_shape = (64, 64)
split = .20

# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(ds_names):

    csv_name = str(ds_name)+"_"+str(input_shape[0])+"_"+str(int(split*100))+"_us.csv"
    header = "["+str(ds_name)+"]"
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

    # data_loader.emotions = pd.get_dummies(data_loader.emotions).as_matrix()
    data_loader.faces = preprocess_input(data_loader.faces)
    for i, split_count in enumerate(data_loader.split_counts):
        maximum = sum(split_count)
        if i == 0:
            test_faces = data_loader.faces[split_count[0]+split_count[1]:maximum]
            test_labels = data_loader.emotions[split_count[0]+split_count[1]:maximum]
            lindex = maximum
        else:
            test_faces = np.vstack((test_faces, data_loader.faces[lindex+split_count[0]+split_count[1]:lindex+maximum]))
            test_labels = np.hstack((test_labels, data_loader.emotions[lindex+split_count[0]+split_count[1]:lindex+maximum]))
            lindex = lindex + maximum

    test_faces, test_labels = shuffle_in_unison(test_faces, test_labels)
    test_data = (test_faces, test_labels)

    # KERAS TEST    
    m_path = "models/FER2013_64_10_bw_mini_XCEPTION.73-0.64.hdf5"
    model = keras_load(m_path)
    kr_accuracy = keras_test(model, test_data)

    # TF TEST
    m_path = "models/FER2013_64_10_bw_mini_XCEPTION.73-0.64.pb"
    # sess, imageTensor, scores = tf_load(m_path)
    # tf_accuracy = tf_test(sess, test_data, imageTensor, scores)
    tf_accuracy = tf_run(m_path, test_data)

    print(header+" Keras accuracy: "+str(kr_accuracy))
    print(header+" Tflow accuracy: "+str(tf_accuracy))

    del(test_faces)
    del(test_labels)
