"""
File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from models.cnn import mini_XCEPTION
from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input
from sklearn.utils import class_weight
import numpy as np

# parameters
batch_size = 32
num_epochs = 10000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = '../trained_models/emotion_models/'

# hardcoded! TODO: pass labels value
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
          4: 'sad', 5: 'surprise', 6: 'neutral'}

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

def count_instances(dataset, faces, emotions):
    label_indexes = []
    num_samples, num_classes = emotions.shape
    print("["+str(dataset)+"] Counting class instances...")
    
    # generate array for counting label instances
    instances_count = [0 for x in range(num_classes)]
    for i in range(len(emotions)):
        label_index = int((np.argwhere(emotions[i]==1)))
        label_indexes.append(label_index)
        instances_count[label_index] += 1
    minimum = min(instances_count)
    min_index = instances_count.index(minimum)
    
    # print class per line
    print("["+str(dataset)+"] (LABEL) count:")
    for i in range(num_classes):
        temp = "["+str(dataset)+"]  ("+str(labels[i])+") "+str(instances_count[i])
        if i == min_index:
            temp = temp+" <--- minimum"
        print(temp)
    
    return instances_count


datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callbacks
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
    # tensorboard logs
    tb_logs = TensorBoard(log_dir='./logs', batch_size=batch_size, write_graph=True, update_freq='epoch')
    trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)
    # list of callbacks to be compiled by used model; can just add here
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr, tb_logs]

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])

    faces, emotions = data_loader.get_data()

    # count class instances TODO: check if this is necessary
    classes_count = count_instances(dataset_name, faces, emotions)

    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    train_data, val_data = split_data(faces, emotions, validation_split)
    train_faces, train_emotions = train_data
    
    # compute class weights for imbalanced data (input is train data)
    train_ints = [emotion.argmax() for emotion in train_emotions]
    # print(train_ints[0:20])
    # print(train_emotions[0:20])
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_ints),
                                                 train_ints)

    # print class per line
    print("["+str(dataset_name)+"] (LABEL) class weight:")
    for i in range(num_classes):
        temp = "["+str(dataset_name)+"]  ("+str(labels[i])+") "+str(class_weights[i])
        print(temp)

    # change class_weights from an array to a dictionary
    class_weights_dict = dict(enumerate(class_weights))
    # print(class_weight_dict)

    model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data, class_weight = class_weights_dict)