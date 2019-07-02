import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from .preprocessor import preprocess_input
from .cnn import mini_XCEPTION, color_XCEPTION


class ModelManager(object):

    def __init__(self, ds_name="", img_shape=(64, 64, 1),
                 num_classes=7, split=0.2, ds_root="data",
                 model_root="models"):
        # TODO: move some vars to settings.py
        # TODO: add absolute path for trained models
        # TODO: root vars from main settings file (img shape, split)
        self.ds_root = ds_root
        self.model_root = model_root
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.ds_name = ds_name
        self.batch_size = 32
        self.num_epochs = 10000
        self.split = split
        self.verbose = 1
        self.patience = 50
        self.train_data = []
        self.test_data = []
        self.val_data = []
        self.model = mini_XCEPTION(self.img_shape, self.num_classes)
        # self.model = color_XCEPTION(self.img_shape, self.num_classes)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, weight_type, class_weights=[]):#, train_data, val_data, model, patience):
        # need variables declared in previous cell
        # consider overriding modelcheckpoint functions to return last successfully saved model
        # pass dataset name
        # callbacks

        data_generator = ImageDataGenerator(featurewise_center=False,
                                            featurewise_std_normalization=False,
                                            rotation_range=10,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            zoom_range=.1,
                                            horizontal_flip=True)

        log_file_path = self.ds_name + "_" + \
                        str(self.img_shape[0]) + "_" + \
                        str(int(self.split*100))+'_'+weight_type+'_training.log'
        log_file_path = os.path.join(self.model_root, log_file_path)
        csv_logger = CSVLogger(log_file_path, append=False)
        early_stop = EarlyStopping('val_loss', patience=self.patience)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                      patience=int(self.patience/4),
                                      verbose=1)
        trained_models_path = self.ds_name+"_"+str(self.img_shape[0])+"_" + \
                              str(int(self.split*100))+'_'+weight_type+'_mini_XCEPTION'
        trained_models_path = os.path.join(self.model_root, 
                                           trained_models_path)
        model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                           save_best_only=True)
        callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

        # loading dataset
        faces, emotions = self.train_data
        # faces = preprocess_input(faces)
        if weight_type == "bw":
            self.model.fit_generator(data_generator.flow(
                                faces, emotions, self.batch_size),
                                steps_per_epoch=len(faces)/self.batch_size,
                                epochs=self.num_epochs, verbose=1,
                                callbacks=callbacks, validation_data=self.val_data,
                                class_weight=class_weights)
        else:
            self.model.fit_generator(data_generator.flow(
                                faces, emotions, self.batch_size),
                                steps_per_epoch=len(faces)/self.batch_size,
                                epochs=self.num_epochs, verbose=1,
                                callbacks=callbacks, validation_data=self.val_data)

    def load_saved(self, model_path):
        self.model = model_path
