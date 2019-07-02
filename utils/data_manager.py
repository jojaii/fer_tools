# original file from https://github.com/oarriaga/face_classification/blob/master/src/utils/datasets.py


from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
import sys
import cv2
from .common import get_class_to_arg, adjust_label, encode_onehot

num_classes = 7


class DSManager(object):

    def __init__(self, ds_name='FER2013',
                 ds_root="/home/Ubuntu/Datasets",
                 image_size=(64, 64),
                 csv_file="",
                 color_flag=False):
        self.ds_name = ds_name
        self.ds_root = ds_root
        self.image_size = image_size
        self.csv_file = csv_file
        self.color_flag = color_flag

        self.faces = []
        self.emotions = []
        self.groups = []
        self.split_counts = []

    def get_data(self):
        fxn_name = "_load_" + self.ds_name
        ground_truth_data = getattr(self, fxn_name)()
        return ground_truth_data

    def _load_imdb(self):
        face_score_treshold = 3
        ds = loadmat(self.ds_root)
        image_names_array = ds['imdb']['full_path'][0, 0][0]
        gender_classes = ds['imdb']['gender'][0, 0][0]
        face_score = ds['imdb']['face_score'][0, 0][0]
        second_face_score = ds['imdb']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return dict(zip(image_names, gender_classes))

    def _load_FER2013(self):
        # original fer dataset is saved as a single csv file
        data = pd.read_csv(os.path.join(self.ds_root, "fer2013.csv"))
        # data = pd.read_csv(os.path.join(self.ds_root, "fer2013_test.csv"))
        pixels = data['pixels'].tolist()
        # TODO: color option
        emotions = [int(x) for x in data['emotion'].tolist()]

        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))

        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        self.faces = faces
        self.emotions = np.asarray(emotions)

    def _load_AffectNet(self):
        ds_root = os.path.join(self.ds_root, "images")
        # all.csv is a combination of the training and testing csvs
        csv_path = os.path.join(self.ds_root, "all.csv")

        data = pd.read_csv(csv_path)
        file_paths = data['subDirectory_filePath'].tolist()
        faces_x = [int(x) for x in data['face_x'].tolist()]
        faces_y = [int(x) for x in data['face_y'].tolist()]
        faces_width = [int(x) for x in data['face_width'].tolist()]
        faces_height = [int(x) for x in data['face_height'].tolist()]
        expressions = [int(x) for x in data['expression'].tolist()]
        # width, height = 64, 64

        files = []
        faces = []
        emotions = []
        for i, expression in enumerate(expressions):
            if expression in range(7, 11):
                # ignore the following labels:
                # contempt, none, uncertain, and no-face
                pass
            else:
                print("Processing "+str(file_paths[i]))
                # TODO: color option
                image_array = cv2.imread(os.path.join(ds_root,
                                         file_paths[i]),
                                         cv2.IMREAD_GRAYSCALE)
                gray_face = image_array[int(faces_y[i]):
                                        int(faces_y[i]+faces_height[i]),
                                        int(faces_x[i]):
                                        int(faces_x[i]+faces_width[i])]
                gray_face = cv2.resize(gray_face.astype('uint8'),
                                       self.image_size)
                faces.append(gray_face.astype('float32'))

                emotion = adjust_label(expression)
                emotions.append(emotion)
                files.append(file_paths[i])

        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        self.faces = faces
        self.emotions = np.asarray(emotions)

    def _load_KDEF(self):
        ds_root = os.path.join(self.ds_root, "images")
        class_to_arg = get_class_to_arg(self.ds_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(ds_root):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)

        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = []
        files = []
        for file_arg, file_path in enumerate(file_paths):
            print("Processing "+str(file_path.split("/")[-1]))
            # TODO: color option
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[4:6]
            # there are two file names in the dataset
            # that don't match the given classes
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            emotions.append(emotion_arg)
            files.append(file_path)

        faces = np.expand_dims(faces, -1)

        self.faces = faces
        self.emotions = np.asarray(emotions)

    def _load_ExpW(self):
        ds_root = os.path.join(self.ds_root, "images")
        lst_path = os.path.join(self.ds_root, "label.lst")
        # lst_path = os.path.join(self.ds_root, "label_test.lst")

        # identified faces in files have face_confidences
        threshold = 0

        faces = []
        emotions = []
        files = []
        f = open(lst_path, "r")
        for line in f.read().splitlines():
            strings = line.split(" ")
            file_name = strings[0]
            # ratio of face is 1:1
            face_top = int(strings[2])
            face_left = int(strings[3])
            face_right = int(strings[4])
            face_bottom = int(strings[5])
            face_confidence = float(strings[6])
            emotion = int(strings[7])

            if face_confidence > threshold:
                print("Processing "+str(file_name))
                # TODO: color option
                image_array = cv2.imread(os.path.join(ds_root, file_name),
                                         cv2.IMREAD_GRAYSCALE)
                gray_face = image_array[face_top:face_bottom,
                                        face_left:face_right]
                gray_face = cv2.resize(gray_face.astype('uint8'),
                                       self.image_size)
                faces.append(gray_face.astype('float32'))
                emotions.append(emotion)
                files.append(file_name)

        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        self.faces = faces
        self.emotions = np.asarray(emotions)

    def generate_csv(self):
        np.set_printoptions(threshold=sys.maxsize)
        np.set_printoptions(linewidth=np.inf)

        print("Saving " + self.ds_name + " dataset to " +
              self.csv_file + ".\n Please wait...")
        f = open(self.csv_file, "w")

        f.write("emotion,face\n")
        for i, emotion in enumerate(self.emotions):
            f.write(str(emotion)+",")
            # TODO: color option
            face_pixels = ' '.join(map(str,
                                   (self.faces[i].astype(int)).flatten()))
            f.write(str(face_pixels+"\n"))

        f.close()
        print("CSV saved.")

    def load_CSV(self):
        # LOAD IMAGES AND LABELS IF CSV EXISTS

        # csv_path = "/home/ubuntu/Documents/AffectNet/Manually_Annotated_file_lists/AffectNet_test.csv"
        # csv_path = "/home/ubuntu/Documents/KDEF/KDEF.csv"
        # csv_path = "/home/ubuntu/Documents/ExpW/label/ExpW.csv"
        data = pd.read_csv(os.path.join(self.ds_root,
                                        self.ds_name,
                                        self.csv_file))
        pixels = data['face'].tolist()

        faces = []
        for pixel_sequence in pixels:
            # TODO: color option
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(self.image_size[0],
                                            self.image_size[1])
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))

        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = [int(x) for x in data['emotion'].tolist()]
        emotions = np.asarray(emotions)

        self.faces = faces
        self.emotions = np.asarray(emotions)

    def load_split_CSV(self):
        # LOAD IMAGES AND LABELS IF CSV EXISTS

        data = pd.read_csv(os.path.join(self.ds_root,
                                        self.csv_file))
        split_values = data['split_count'][:num_classes].tolist()
        pixels = data['face'].tolist()
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            # TODO: if grayscale, no third dimension
            # color has third, so wxh changed to image_size
            face = np.asarray(face).reshape(self.image_size)
            face = face.astype('uint8')
            # face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))

        faces = np.asarray(faces)
        # TODO: on if grayscale, off if not
        # faces = np.expand_dims(faces, -1)
        emotions = [int(x) for x in data['emotion'].tolist()]
        groups = [int(x) for x in data['group'].tolist()]

        split_counts = []
        for row in split_values[:num_classes]:
            values = np.asarray([int(split_int) for split_int in row.split(' ')])
            split_counts.append(values)

        self.faces = faces
        self.emotions = np.asarray(emotions)
        self.groups = np.asarray(groups)
        # TODO: propagate change to affected files
        self.split_counts = np.asarray(split_counts)


    def _load_FERplus(self):
        ds_root_1 = "/home/ubuntu/Repos/face_classification/datasets/FER2013/FER2013.csv"
        ds_root_2 = "/home/ubuntu/Repos/face_classification/datasets/FER2013/FER2013new.csv"

        data = pd.read_csv(self.ds_root)
        data_plus = pd.read_csv(ds_root_2)
        pixels = data['pixels'].tolist()
        width, height = 48, 48

        faces = []
        for pixel_sequence in pixels:
            # TODO: color option
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)

        faces = np.expand_dims(faces, -1)

        # get emotions from FER
        # emotions_FER = pd.get_dummies(data['emotion']).as_matrix()
        emotions_FER = data['emotion'].tolist()
        # print(emotions)
        # input()

        # TODO: get emotions from FERplus
        emotion_list = ["anger", "disgust", "fear", "happiness",
                        "sadness", "surprise", "neutral"]
        images_labels = data_plus[emotion_list].as_matrix()
        emotions = []
        for i, image_labels in enumerate(images_labels):
            # get indices of highest voted emotions
            max_value = np.max(image_labels)
            max_indices = np.argwhere(image_labels == max_value)
            # print(max_indices)
            # print(len(max_indices))
            # if highest vote has value 0, then that means
            # the highest voted is not part of the 7 traditional
            # expressions (so it's contempt, unknown, or NF)
            if np.max(image_labels) == 0:
                # get label from original FER labels, not from FER+
                emotions.append(int(emotions_FER[i]))
                # print("COND1 - appended " + str(emotions[-1]))
            # if there is more than one max index...
            elif len(max_indices) > 1:
                # append disgust (index 1) because it has the lowest number of data
                if 1 in max_indices:
                    emotions.append(1)
                # append fear (index 2) for similar reason
                elif 2 in max_indices:
                    emotions.append(2)
                # append angry (index 0) for similar reason
                elif 0 in max_indices:
                    emotions.append(0)
                # append emotion with least number of occurences in temporary list
                else:
                    index_count = []
                    index_count = [emotions.count(index) for index in max_indices]
                    min_index = np.argmin(index_count)
                    emotions.append(max_indices[min_index][0])
                # print("COND2 - appended " + str(emotions[-1]))
            else:
                emotions.append(np.argmax(image_labels))
                # print("COND3 - appended " + str(emotion[-1]))

        self.faces = faces
        self.emotions = np.asarray(emotions)

    def _load_SFEW(self):
        classes = get_class_to_arg(self.ds_name)
        dataset_path = "/home/ubuntu/Datasets/SFEW_ssd/TrainVal"
        # load images in folder specified
        labels = []
        faces = []
        for folder, _, filenames in os.walk(dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('.png')):
                    f_name = os.path.join(folder, filename)
                    label = folder.split('/')[-1].lower()
                    labels.append(classes[label])
                    # TODO: color option
                    face = cv2.imread(f_name,
                                      cv2.IMREAD_GRAYSCALE)
                    face = cv2.resize(face.astype('uint8'),
                                      self.image_size)
                    faces.append(face.astype('float32'))

        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        self.faces = faces
        self.emotions = np.asarray(labels)

    def _load_FERcolor(self):
        ds_root = os.path.join(self.ds_root, self.ds_name) 
        labels, faces = [], []
        for folder, subfolders, filenames in os.walk(ds_root):
            for filename in filenames:
                if filename.lower().endswith(('.png')):
                    f_name = os.path.join(folder, filename)
                    label = folder.split('/')[-1].lower()
                    labels.append(int(label))
                    # TODO: color option
                    face = cv2.imread(f_name)  # BGR
                    face = cv2.resize(face.astype('uint8'),
                                      self.image_size)
                    faces.append(face.astype('float32'))

        self.faces = np.asarray(faces)
        self.emotions = np.asarray(labels)


if __name__ == '__main__':
    input_shape = (64, 64, 3)
    ds_name = 'FERcolor'
    ds_root = "/home/ubuntu/Datasets"
    path = os.path.join(ds_root, ds_name)
    csv_name = str(ds_name)+"_"+str(input_shape[0])+".csv"
    csv_path = os.path.join(ds_root, ds_name, csv_name)
    test_train = DSManager(ds_name,
                           ds_root=path,
                           image_size=input_shape[:2],
                           csv_file=csv_path)
    test_train.get_data()
    test_train.generate_csv()
