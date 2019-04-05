import os
from utils.data_manager import DSManager
from utils.common import shuffle_in_unison
import numpy as np
from utils.preprocessor import preprocess_input
from keras.models import load_model
import cv2
import time
import argparse
import math
import matplotlib.pyplot as plt


caffe_root = '/home/ubuntu/caffe/' ### Change your caffe_root here
import caffe
from caffe import layers as L, params as P
from ltnet import EmotEngine

parser = argparse.ArgumentParser(description="")
parser.add_argument('--weight', '-w', type=str, help='evaluate model')
args = parser.parse_args()

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


ds_root = 'data'
# ds_names = ['AffectNet', 'ExpW', 'FER2013', 'KDEF']
ds_names = ['FER2013']
# ds_names = ['FER2013']
m_root = "models"
m_names = []
fer_path = "/home/ubuntu/Repos/face_classification/trained_models/undersampled"
input_shape = (64, 64)
split = .2
type = "bw"
accuracies = []
ds_dict = {}
num_classes = 7

batchsize = 64
src_size = 64
input_data_str = "input"


def eval_validation(model_weights, model_def="res80_fer_deploy.prototxt",
                    device_no=1, batchsize=64, faces=[], labels=[],
                    num_classes=7):

    predictor = EmotEngine.EmotEngine(model_def, model_weights,
                                      device_no=device_no, norm_flag=0,
                                      src_size=(64, 64))
    batchsize = predictor.batchsize
    print("Batchsize: %d" % batchsize)
    sumloss = 0.0
    sumacc = 0
    num_images = len(faces)
    num_batches = float(num_images)/float(batchsize)
    print("Batches: %.2f" % num_batches)
    for i in range(math.floor(num_batches)):
        face_list = faces[i*batchsize:i*batchsize+batchsize]
        label_list = labels[i*batchsize:i*batchsize+batchsize]
        print("Passing batch %d" % i)
        iacc, iloss = predictor.get_emoti_and_loss_from_batch(face_list,
                                                              label_list,
                                                              nclass=num_classes)
        sumloss += iloss*num_images
        sumacc += sum(iacc)
    if num_batches % math.floor(num_batches) != 0:
        print("Passing extra batch")
        face_list = faces[i*batchsize+batchsize:len(faces)]
        label_list = labels[i*batchsize+batchsize:len(faces)]
        iacc, iloss = predictor.get_emoti_and_loss_from_batch(face_list,
                                                              label_list,
                                                              nclass=num_classes)
        sumloss += iloss*num_images
        sumacc += sum(iacc)
    sumacc = float(sumacc)/float(num_images)
    print("acc = %f" % sumacc)

    return sumacc


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
                        []]
                        # [mini_XCEPTION(input_shape, num_classes)]]

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

    model_weight = "/home/ubuntu/Repos/fer_tools/ltnet/snapshot/train_fer_iter_100000.caffemodel"
    device_no = -1
    print("LOG<<" + model_weight)
    # accuracy = eval_validation(model_weight,
    #                            model_def="ltnet/res80_fer_deploy.prototxt",
    #                            device_no=device_no, batchsize=64,
    #                            faces=test_faces, labels=test_labels,
    #                            num_classes=num_classes)

    model_def = "ltnet/res80_fer_deploy.prototxt"

    preds = []
    predictor = EmotEngine.EmotEngine(model_def, model_weight,
                                      device_no=device_no, norm_flag=0,
                                      src_size=(64, 64),
                                      is_colored=False)

    folder = os.path.join("data/ltnet", ds_name)
    for label in range(num_classes):
        if not os.path.exists(os.path.join(folder, str(label))):
            os.makedirs(os.path.join(folder, str(label)))

    f_name = "bin/ltnet_preds.csv"
    print(header+" Saving all probabilities to " +
          f_name + ".\n Please wait...")
    f = open(f_name, "w")
    correct = 0
    for j, face in enumerate(test_faces):
        preds.append(predictor.get_emoti_from_img(face))
        print(preds[-1])
        if preds[-1] == test_labels[j]:
            correct += 1
        f.write(str(preds[-1])+"\n")
        folder_2 = os.path.join(folder, str(preds[-1]))
        face = np.squeeze(face, axis=-1)
        filename = os.path.join(folder_2, str(j)+".jpg")
        plt.imsave(filename, face, cmap="gray")
    f.close()
    print(header+" Preds csv saved.")
    accuracy = float(correct)/float(len(test_labels))
    print("Correct: %.2f (%d/%d)" % (accuracy, correct, len(test_labels)))

    del(data_loader)
    # # del(val_faces)
    # # del(val_labels)
    del(test_faces)
    del(test_labels)
    del(preds)

# print(accuracies)
