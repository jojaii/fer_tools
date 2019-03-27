# # uncomment to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# splitting does not rely on weight type
# it only relies on whether the data is undersample or not
import os
import settings
from keras.preprocessing.image import ImageDataGenerator
from utils.data_manager import DSManager
from utils.common import (shuffle_in_unison, strat_sample,
                          group, truncate, combine_classes,
                          generate_csv, count_labels)
# data generator
data_generator = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=.1,
                                    horizontal_flip=True)

# TODO: should be imported from config file
datasets = settings.DATASET_NAMES
input_shape = settings.INPUT_SHAPE
num_classes = settings.NUM_CLASSES
ds_root = settings.DATASET_ROOT
split = settings.SPLIT
undersample = settings.UNDERSAMPLE_FLAG
# weight_type = settings.WEIGHT_TYPE
ds_dict = {}

# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(datasets):
    # load/create csv
    csv_name = str(ds_name) + "_" + str(input_shape[0]) + ".csv"
    header = "["+str(ds_name)+"]"
    # TODO: default input for csv_file
    data_loader = DSManager(ds_name,
                            ds_root=ds_root,
                            image_size=input_shape[:2],
                            csv_file=csv_name)
    try:
        csv_file = open(os.path.join(ds_root, ds_name, csv_name))
    except:
        print(header + " File "+csv_name+" does not exist. " +
              "Loading and converting to csv.")
        print(header + " Loading and converting to csv " +
              "with image size " + str(input_shape[:2]))
        # TODO: test conversion from original datasets
        data_loader.get_data()
        data_loader.generate_csv()
    else:
        print(header + " File "+csv_name+" exists. Loading file...")
        data_loader.load_CSV()

    # SHUFFLE
    print(header + " Shuffling images and labels simultaneously...")
    (shuf_faces, shuf_emotions) = shuffle_in_unison(data_loader.faces,
                                                    data_loader.emotions)

    # GROUP (according to class)
    grouped_faces = group(shuf_faces, shuf_emotions, num_classes)

    # UNDERSAMPLE if turned on
    if undersample is True:
        labels_count = count_labels(shuf_emotions)
        minimum = min(labels_count)
        grouped_faces = truncate(grouped_faces, minimum)

    # SPLIT get int values list for split
    classes_split_labels, classes_split_counts = strat_sample(grouped_faces,
                                                              split)

    # combine classes, remove groupings
    combined_faces, labels = combine_classes(grouped_faces)
    if undersample is True:
        csv_name = ds_root+ds_name+"_"+str(input_shape[0])+"_"+str(int(split*100))+"_us.csv"
    else:
        csv_name = ds_root+ds_name+"_"+str(input_shape[0])+"_"+str(int(split*100))+".csv"
    generate_csv(ds_name, csv_name, num_classes, combined_faces, labels, classes_split_labels, classes_split_counts)

    del(data_loader)
    del(shuf_faces)
    del(shuf_emotions)
    del(labels_count)
    del(grouped_faces)
    del(minimum)
    del(classes_split_labels)
    del(classes_split_counts)
    del(combined_faces)
    del(labels)
