import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import os.path as osp
import random
import math
import os
import pdb
import argparse
import sys
from shutil import copyfile
import pandas as pd

caffe_root = '/home/ubuntu/caffe/' ### Change your caffe_root here
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L, params as P
import EmotEngine

parser = argparse.ArgumentParser(description="")
parser.add_argument('--weight', '-w', type=str, help='evaluate model')
args = parser.parse_args()


def parse_img_list_item(line):
    line = line.strip('\n')
    line = line.split(' ')
    if len(line) < 2:
        return -1, -1
    return line[0], int(line[1])

def get_data():
    # LOAD IMAGES AND LABELS IF CSV EXISTS
    data = pd.read_csv(os.path.join(self.ds_root,
                                    self.csv_file))
    split_values = data['split_count'][:num_classes].tolist()
    pixels = data['face'].tolist()
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(self.image_size[0],
                                        self.image_size[1])
        face = cv2.resize(face.astype('uint8'), self.image_size)
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = [int(x) for x in data['emotion'].tolist()]
    groups = [int(x) for x in data['group'].tolist()]

    split_counts = []
    for row in split_values[:num_classes]:
        values = np.asarray([int(split_int) for split_int in row.split(' ')])
        split_counts.append(values)

    self.faces = faces
    self.emotions = np.asarray(emotions)
    self.groups = np.asarray(groups)
    self.split_counts = split_counts

def eval_validation(model_weights, model_def="res80_fer_deploy.prototxt",
                    device_no=1,
                    validation_img_list="imglist.txt",
                    img_folder="/home/data/", batchsize=64):

    predictor = EmotEngine.EmotEngine(model_def, model_weights,
                                      device_no=device_no, norm_flag=2,
                                      src_size=(36, 36))
    batchsize = predictor.batchsize
    freader = open(validation_img_list, 'r')
    count = 0
    sumloss = 0.0
    sumacc = 0
    flag = 0
    imglist = []
    lbllist = []
    for line in freader:
        count += 1
        flag += 1
        img_file, ilbl = parse_img_list_item(line)
        if img_file == -1:
            flag += -1
            count += -1
            continue
        if flag % batchsize == 0:
            imglist.append(img_folder+img_file)
            lbllist.append(ilbl)
            iacc, iloss = predictor.get_emoti_and_loss_from_batch(imglist,
                                                                  lbllist,
                                                                  nclass=10)
            sumloss += iloss*len(imglist)
            sumacc += sum(iacc)
            imglist = []
            lbllist = []
            flag = 0
        else:
            imglist.append(img_folder+img_file)
            lbllist.append(ilbl)
        if count % 100 == 0:
            # break
            print("%d imgs predicted" % count)
    freader.close()
    iacc, iloss = predictor.get_emoti_and_loss_from_batch(imglist,
                                                          lbllist,
                                                          nclass=10)
    sumacc += sum(iacc)
    if count > 0:
        sumacc = float(sumacc)/float(count)
    print("acc = %f" % sumacc)
    return sumacc


def display_validation_test_result(model_weight, device_no=1):
    test_list = '/home/OtherData/CIFAR/cifar_test_list.txt'
    val_test_image_root = '/home/OtherData/CIFAR/'  # change the root of your images
    # MMI
    acc_cifar10 = eval_validation(model_weight,
                                  model_def="res20_cifar10_deploy_org.prototxt",
                                  validation_img_list=test_list,
                                  img_folder=val_test_image_root,
                                  device_no=device_no)
    print('LOG<< validation set: CIFAR1- ACC \t %.6f' % acc_cifar10)


model_weight = args.weight
device_no = 2
print("LOG<<" + model_weight)
display_validation_test_result(model_weight, device_no=device_no)
