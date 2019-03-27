# # uncomment to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import os
from utils.data_manager import DSManager
import numpy as np
from utils.preprocessor import preprocess_input
from keras.models import load_model
import cv2


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


np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.inf)
ds_root = 'data'
ds_names = ['AffectNet', 'ExpW', 'FER2013', "All"]#, 'KDEF']#, 'SFEW']
m_root = "models"
m_names = []
fer_path = "/home/ubuntu/Repos/face_classification/trained_models/undersampled"
# m_paths = ["AffectNet_64_10_bw_mini_XCEPTION.30-0.68.hdf5",
#            "ExpW_64_10_bw_mini_XCEPTION.13-0.56.hdf5",
#            "FER2013_64_10_bw_mini_XCEPTION.73-0.64.hdf5",
#            "KDEF_64_10_bw_mini_XCEPTION.47-0.22.hdf5"]
m_paths = ["AffectNet_64_20_bw_mini_XCEPTION.26-0.71.hdf5",
           "ExpW_64_20_bw_mini_XCEPTION.39-0.55.hdf5",
           "FER2013_64_20_bw_mini_XCEPTION.57-0.61.hdf5"]
        #    "KDEF_64_20_bw_mini_XCEPTION.45-0.20.hdf5",
           # All hdf5
# m_paths = ["AffectNet_64_10_us_mini_XCEPTION.82-0.55.hdf5",
#            "FER2013_64_10_us_mini_XCEPTION.57-0.54.hdf5",
#            "KDEF_64_10_us_mini_XCEPTION.67-0.23.hdf5",
#            "ExpW_64_10_us_mini_XCEPTION.56-0.42.hdf5"]
# m_paths = ["AffectNet_64_20_us_mini_XCEPTION.85-0.58.hdf5",
#            "FER2013_64_20_us_mini_XCEPTION.45-0.48.hdf5",
#            "KDEF_64_20_us_mini_XCEPTION.56-0.19.hdf5",
#            "ExpW_64_20_us_mini_XCEPTION.42-0.42.hdf5"]
# m_paths = ['AffectNet_64_20_rw_mini_XCEPTION.142-0.76.hdf5',
#            'ExpW_64_20_rw_mini_XCEPTION.117-0.69.hdf5',
#            'FER2013_64_20_rw_mini_XCEPTION.70-0.65.hdf5',
#            'KDEF_64_20_rw_mini_XCEPTION.51-0.19.hdf5',
#            'SFEW_64_20_rw_mini_XCEPTION.28-0.33.hdf5']

input_shape = (64, 64)
split = .2
type = "bw"
accuracies = []
ds_dict = {}

# LOAD DATASETS INTO DICTIONARY
print("Loading datasets...")
for i, ds_name in enumerate(ds_names):
    if ds_name != "All":
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
        m_path = os.path.join(m_root, m_paths[i])
        data_loader.faces = preprocess_input(data_loader.faces)
        ds_dict[ds_name] = [data_loader.faces,
                            data_loader.emotions,
                            keras_load(m_path),
                            {}]

        del(data_loader)
    else:
        # load train, test, val
        m_path = os.path.join(m_root, m_paths[i])
        ds_dict[ds_name] = [[]],
                            []],
                            keras_load(m_path),
                            {}]


print("---- Multi-labelling datasets")
for i, ds_model in enumerate(ds_names):
    header = "["+str(ds_model)+"]"
    for j, ds_data in enumerate(ds_names[:len(ds_names)-1]):
        print(header+" Creating predictions for "+str(ds_data)+" data...")
        test_data = (ds_dict[ds_data][0], ds_dict[ds_data][1])
        preds, probs = keras_test(ds_dict[ds_model][2],
                                  test_data)
        # save labels to ds_name2 since it is the source of the data
        ds_dict[ds_data][3][ds_model] = (preds, probs)


print("---- Saving all labels to csvs")
for i, ds_name in enumerate(ds_names[:len(ds_names-1)]):
    header = "["+str(ds_name)+"]"
    print(header+" Saving actual labels and "+ds_name+" predictions...")
    all_labels = ds_dict[ds_name][1]
    all_probs = []
    columns = ["actual", "prediction"]
    # add predictions for current folder first
    preds = ds_dict[ds_name][3][ds_name][0]
    probs = ds_dict[ds_name][3][ds_name][1]
    all_labels = [str(k)+","+str(preds[iter]) for iter, k in enumerate(all_labels)]
    all_probs.extend(probs)
    for j, ds_name2 in enumerate(ds_names):
        if i != j:
            print(header+" Saving "+ds_name2+" predictions...")
            columns.append(ds_name2)
            preds = ds_dict[ds_name][3][ds_name2][0]
            probs = ds_dict[ds_name][3][ds_name2][1]
            all_labels = [str(k)+","+str(preds[iter]) for iter, k in enumerate(all_labels)]
            all_probs = [str(k)+","+str(probs[iter]) for iter, k in enumerate(all_probs)]

    # open multi-labelled csv
    f_name = "bin/"+ds_name+"_preds_"+str(int(split*100))+"_"+type+".csv"

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
    f_name = "bin/"+ds_name+"_probs_"+str(int(split*100))+"_"+type+".csv"

    print(header+" Saving all probabilities to " +
          f_name + ".\n Please wait...")
    f = open(f_name, "w")
    temp = ','.join(map(str, columns[1:]))
    f.write(temp+"\n")
    for i, line in enumerate(all_probs):
        f.write(line+"\n")
    f.close()
    print(header+" Preds csv saved.")
    del(all_probs)
    del(columns)
    # close probs csv

del(ds_dict)
