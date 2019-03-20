
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,
                             f1_score,
                             precision_recall_fscore_support)

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
f_name = "SFEW_preds_20_bw.csv"
f_path = os.path.join(f_root, f_name)
ds_names = ['AffectNet', 'ExpW', 'FER2013', 'KDEF']
# ds_names = ['FER2013']

for ds_name in ds_names:
    data = pd.read_csv(f_path)
    y_pred = [int(x) for x in data[ds_name].tolist()]
    y_actual = [int(x) for x in data["actual"].tolist()]
    y_pred = np.asarray(y_pred)
    y_actual = np.asarray(y_actual)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_actual, y_pred)
    np.set_printoptions(precision=2)
    title = os.path.join(f_root, "SFEW_"+ds_name+"_bw")
    output_title = title + ".png"

    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True, title=title)

    plt.savefig(output_title)
    plt.show()

    p_mac, r_mac, f_mac, s_mac = precision_recall_fscore_support(y_actual,
                                                                 y_pred,
                                                                 average="macro")
    print("-------------------------------------------------------")
    print(title)
    print("============== MACRO ==============")
    print("Precision: ", str(p_mac))
    print("Recall:    ", str(r_mac))
    print("F1 Score:  ", str(f_mac))
    p_mic, r_mic, f_mic, s_mic = precision_recall_fscore_support(y_actual,
                                                                 y_pred,
                                                                 average="micro")
    print("============== MICRO ==============")
    print("Precision: ", str(p_mic))
    print("Recall:    ", str(r_mic))
    print("F1 Score:  ", str(f_mic))
    print("-------------------------------------------------------")

    del(y_pred)
    del(y_actual)
    del(cnf_matrix)
