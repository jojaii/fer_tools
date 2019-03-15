import numpy as np
import pandas as pd
from utils.preprocessor import preprocess_input
from keras.models import load_model
import cv2
import tensorflow as tf
import time
from collections import Counter

image_size = (64, 64)
def load_fertest(ds_path):
    data = pd.read_csv(ds_path)
    pixels = data['pixels'].tolist()
    usage = data['Usage'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    # emotions = pd.get_dummies(data['emotion']).as_matrix()
    emotions = [int(x) for x in data["emotion"].tolist()]
    usage_count = Counter(usage)
    # start_i = usage_count["Training"]
    # end_i = start_i + usage_count["PublicTest"]
    start_i = usage_count["Training"] + usage_count["PublicTest"]
    end_i = start_i + usage_count["PrivateTest"]
    faces = faces[start_i:end_i]
    emotions = emotions[start_i:end_i]

    return faces, emotions


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
        image = preprocess_input(image, True)
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

    correct = 0
    images = data[0]
    actual = data[1]
    sum_time = 0
    for i, image in enumerate(images):
        image = cv2.resize(image, (64, 64))
        image = preprocess_input(image, True)

        start = time.time()
        imageTensor = detectionGraph.get_tensor_by_name('input_1:0')
        scores = detectionGraph.get_tensor_by_name('predictions/Softmax:0')

        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, -1)
        (scores) = sess.run([scores],
                            feed_dict={imageTensor: image})
        prediction = np.argmax(scores)
        sum_time += time.time() - start
    
        if prediction == actual[i]:
            correct += 1
    print("tf latency (seconds/image)", (sum_time/len(images)))

    accuracy = correct/len(images)

    return accuracy


test_faces, test_labels = load_fertest('/home/ubuntu/Datasets/FER2013/fer2013.csv')
test_data = (test_faces, test_labels)
# KERAS TEST   
m_path = "models/fer2013_mini_XCEPTION.66-0.66.hdf5"
model = keras_load(m_path)
kr_accuracy = keras_test(model, test_data)

# TF TEST
m_path = "models/fer2013_mini_XCEPTION.66-0.66.pb"
# sess, imageTensor, scores = tf_load(m_path)
# tf_accuracy = tf_test(sess, test_data, imageTensor, scores)
tf_accuracy = tf_run(m_path, test_data)

print(" Keras accuracy: "+str(kr_accuracy))
print(" Tflow accuracy: "+str(tf_accuracy))

del(test_faces)
del(test_labels)
