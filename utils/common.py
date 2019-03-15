import numpy as np
# TODO: define fxn inputs and outputs

# import settings

num_classes = 7


def count_labels(labels):
    # count labels from 1d array of labels
    # input: labels int array || num_classes int
    # output: array of instances count (size according to num_classes)
    labels_count = [0 for x in range(num_classes)]
    for label in labels:
        labels_count[label] += 1
    return labels_count


def get_subsets(minimum, faces, emotions, labels):
    # get first n instances per label from labels array, get faces array of same index
    # does not randomize because assumes input is already shuffled
    # outputs an array of numpy arrays (according to class)
    # input: n, faces, labels
    # output: faces_subset, labels_subset
    counter = [minimum for x in range(num_classes)]
    subsets_faces = [[] for x in range(num_classes)]
    subsets_emotions = [[] for x in range(num_classes)]

    i = 0
    while sum(counter) != 0:
        label = labels[i]
        if counter[label] != 0:
            if subsets_faces[label] == []:
                temp_face = np.expand_dims(faces[i], axis=0)
                temp_emotion = np.expand_dims(emotions[i], axis=0)
                subsets_faces[label] = temp_face
                subsets_emotions[label] = temp_emotion
            else:
                temp_face = np.expand_dims(faces[label], axis=0)
                temp_emotion = np.expand_dims(emotions[label], axis=0)
                subsets_faces[label] = np.vstack((subsets_faces[label],
                                                 temp_face))
                subsets_emotions[label] = np.vstack((subsets_emotions[label],
                                                    temp_emotion))
            counter[label] -= 1
        i += 1

    return subsets_faces, subsets_emotions


# def split_data(x, y, split):
#     # split data to train, test, and val according and percentage
#     # input: faces_subset, labels_subset, split_percentage
#     # output: train_data, val_data, test_data
#     num_samples = len(x)
#     num_val_samples = int(round(split*num_samples))
#     num_train_samples = int(round((1 - split*2)*num_samples))
#     train_x = x[:num_train_samples]
#     train_y = y[:num_train_samples]
#     val_x = x[num_train_samples:num_train_samples+num_val_samples]
#     val_y = y[num_train_samples:num_train_samples+num_val_samples]
#     test_x = x[num_train_samples+num_val_samples:]
#     test_y = y[num_train_samples+num_val_samples:]
#     train_data = (train_x, train_y)
#     val_data = (val_x, val_y)
#     test_data = (test_x, test_y)

#     return train_data, val_data, test_data


# def strat_sample(faces, emotions, split):
#     # input: faces array (array of numpy arrays),
#     #        emotions array (array of numpy arrays),
#     #        split int percentage of train/test data
#     # output: train_data - tuple of two numpy arrays
#     #         val_data - tuple of two numpy arrays
#     #         test_data - tuple of two numpy arrays
#     for i in range(num_classes):
#         # split according to class
#         temp_train, temp_val, temp_test = split_data(faces[i],
#                                                      emotions[i],
#                                                      split)
#         if i == 0:
#             train_faces, train_emotions = temp_train
#             val_faces, val_emotions = temp_val
#             test_faces, test_emotions = temp_test
#         else:
#             train_faces = np.vstack((train_faces, temp_train[0]))
#             train_emotions = np.vstack((train_emotions, temp_train[1]))
#             val_faces = np.vstack((val_faces, temp_val[0]))
#             val_emotions = np.vstack((val_emotions, temp_val[1]))
#             test_faces = np.vstack((test_faces, temp_test[0]))
#             test_emotions = np.vstack((test_emotions, temp_test[1]))

#     train_data = (train_faces, train_emotions)
#     val_data = (val_faces, val_emotions)
#     test_data = (test_faces, test_emotions)

    return train_data, val_data, test_data

def split_array(array, split):
    # split data to train, test, and val according and percentage
    # input: faces_subset, labels_subset, split_percentage
    # output: train_data, val_data, test_data
    num_samples = len(array)
    num_val_samples = int(round(split*num_samples))
    num_test_samples = num_val_samples
    num_train_samples = int(round((1 - split*2)*num_samples))
    # label as train data
    split_ints = [0 for x in range(num_train_samples)]
    # label as val data
    split_ints.extend([1 for x in range(num_val_samples)])
    # label as test data
    split_ints.extend([2 for x in range(num_test_samples)])

    # TODO: clean
    # ensures that labels do not exceed the original array size
    new_length = len(split_ints)
    old_length = len(array)
    if new_length > len(array):
        # print(np.shape(split_ints))
        # print(np.shape(array))
        split_ints = split_ints[:old_length]
        num_test_samples -= 1
    elif new_length < old_length:
        split_ints.append(2)
        num_test_samples += 1
    else:
        pass

    split_count = [num_train_samples, num_val_samples, num_test_samples]
    # print(np.shape(split_ints))
    # exit()
    return split_ints, split_count


def strat_sample(grouped_classes, split):

    classes_split_labels = []
    classes_split_counts = []
    for i, grouped_class in enumerate(grouped_classes):
        # split according to class
        split_ints, split_count = split_array(grouped_class, split)
        classes_split_labels.extend(split_ints)
        classes_split_counts.append(split_count)
    # print("---"+str(np.shape(classes_split_labels)))
    # print("---"+str(np.shape(grouped_classes)))
    # exit()
    return classes_split_labels, classes_split_counts


def combine_classes(grouped_classes):

    combined_faces = []
    labels = []
    for i, grouped_class in enumerate(grouped_classes):
        combined_faces.extend(grouped_class)
        labels.extend([i for x in range(len(grouped_class))])

    return combined_faces, labels


def generate_csv(ds_name, csv_name, num_classes, faces, labels, split_labels, split_counts):

    print("Saving " + ds_name + " dataset to " +
          csv_name + ".\n Please wait...")
    print(split_counts)
    f = open(csv_name, "w")

    f.write("emotion,face,group,split_count\n")
    for i, label in enumerate(labels):
        f.write(str(label)+",")
        face_pixels = ' '.join(map(str,
                               (faces[i].astype(int)).flatten()))
        f.write(str(face_pixels+","))
        f.write(str(split_labels[i]))
        if i in range(num_classes):
            temp = np.asarray([int(x) for x in split_counts[i]])
            temp = ' '.join(map(str,
                            temp.flatten()))
            f.write(","+str(temp))
        f.write("\n")
    f.close()
    print("CSV saved.")

def shuffle_in_unison(a, b):
    # SHUFFLE FACES AND EMOTION ARRAYS SIMULTANEOUSLY
    # source: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]

    return (shuffled_a, shuffled_b)


def adjust_label(label):
    # change AffectNet labels to FER2013 labels
    # TODO: raise exception if label not in range?
    new_label = {6: 0, 5: 1,
                    4: 2, 1: 3,
                    2: 4, 3: 5,
                    0: 6}
    return new_label[label]


def encode_onehot(labels):
    onehot_array = np.zeros((len(labels), max(labels)+1))
    onehot_array[np.arange(len(labels)), labels] = 1
    return onehot_array


def get_labels(ds_name):
    # TODO: set in settings.py
    if ds_name in ['FER2013', 'ExpW', 'ExpWCSV', 'SFEW']:
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif ds_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif ds_name in ['KDEF', 'KDEFCSV']:
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    elif ds_name in ['AffectNet', 'AffectNetCSV']:
        return {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
                4: 'fear', 5: 'disgust', 6: 'anger'}
    else:
        raise Exception('Invalid dataset name')


def get_class_to_arg(ds_name='FER2013'):
    # TODO: set in settings.py
    if ds_name in ['FER2013', 'ExpW', 'ExpWCSV', 'SFEW']:
        return {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4,
                'surprise': 5, 'neutral': 6}
    elif ds_name == 'imdb':
        return {'woman': 0, 'man': 1}
    elif ds_name in ['KDEF', 'KDEFCSV']:
        return {'AN': 0, 'DI': 1, 'AF': 2, 'HA': 3, 'SA': 4, 'SU': 5, 'NE': 6}
    elif ds_name in ['AffectNet', 'AffectNetCSV']:
        return {'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3,
                'fear': 4, 'disgust': 5, 'anger': 6}
    else:
        raise Exception('Invalid dataset name')


# GROUP (according to class)
def group(array, labels, num_classes):
    grouped_classes = [[] for x in range(num_classes)]
    for i, label in enumerate(labels):
        grouped_classes[label].append(array[i])

    return grouped_classes


# truncate per class according to minimum
# subset was already randomized when it was shuffled earlier
def truncate(array, minimum):
    for i, row in enumerate(array):
        array[i] = row[:minimum]
    return array
