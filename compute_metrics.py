
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,
                             f1_score,
                             precision_recall_fscore_support,
                             accuracy_score)


parser = argparse.ArgumentParser()
parser.add_argument("--thresh", nargs=1, type=float, metavar="<thresh>", 
                    help="Set threshold for acceptable confidence values.")
args = parser.parse_args()


labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.0f' if normalize else 'd'
#     fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual sentiment')
    plt.xlabel('Predicted sentiment')
    plt.tight_layout()


# TODO: these should be arguments
f_root = "bin"
ds_base = 'SFEW'
f_preds = "SFEW_preds_20_bw.csv"
f_probs = "SFEW_probs_20_bw.csv"
type = "bw"
# f_path = os.path.join(f_root, f_preds)
ds_names = ['AffectNet', 'ExpW', 'FER2013', 'KDEF']
# ds_names = ['FER2013']
thresh = args.thresh[0]
preds = pd.read_csv(os.path.join(f_root, f_preds))
probs = pd.read_csv(os.path.join(f_root, f_probs))

print("TEST DATA: "+ds_base)
for ds_name in ds_names:
    y_true = [int(x) for x in preds["actual"].tolist()]
    y_preds = [int(x) for x in preds[ds_name].tolist()]
    y_probs = [float(x) for x in probs[ds_name].tolist()]
    y_preds = np.asarray(y_preds)
    y_true = np.asarray(y_true)
    del_indexes = []
    num_classes = 7

    # filter out predictions less with classification less than threshold
    for i, prob in enumerate(y_probs):
        if prob < thresh:
            del_indexes.append(i)
    y_true = np.delete(y_true, del_indexes)
    y_preds = np.delete(y_preds, del_indexes)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_preds, labels=range(num_classes))
    np.set_printoptions(precision=2)
    title = "SFEW_"+ds_name+"_"+type+"_"+str(int(thresh*100))
    output_title = title + ".png"

    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True, title=title)

    plt.savefig(os.path.join(f_root, title))
    plt.close()
    # plt.show()
    p_mac, r_mac, f_mac, s_mac = precision_recall_fscore_support(y_true,
                                                                 y_preds,
                                                                 average="macro")
    remaining = len(y_true)
    total = len(y_probs)
    percent_rem = (remaining/total)*100
    print("===========================")
    print("-- MODEL: "+ds_name)
    print("-- %d/%d = %.2f" % (remaining, total, percent_rem))
    print("---------- MACRO ----------")
    print(" Precision: ", str(p_mac))
    print(" Recall:    ", str(r_mac))
    print(" F1 Score:  ", str(f_mac))
    p_mic, r_mic, f_mic, s_mic = precision_recall_fscore_support(y_true,
                                                                 y_preds,
                                                                 average="micro")
    print("---------- MICRO ----------")
    print(" Precision: ", str(p_mic))
    print(" Recall:    ", str(r_mic))
    print(" F1 Score:  ", str(f_mic))

    del(cnf_matrix)
    del(y_preds)
    del(y_true)

print("===========================")
