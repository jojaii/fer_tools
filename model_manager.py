from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from .preprocessor import preprocess_input
from .cnn import mini_XCEPTION

class ModelManager(object):

    def __init__(self, ds_name="", img_shape=""):
        # TODO: move some vars to settings.py
        self.ds_root = "/home/ubuntu/Desktop/"
        # TODO: add absolute path for trained models
        self.model_root = "/home/ubuntu/Datasets/Models/"
        self.batch_size = 32
        self.num_epochs = 10000
        self.img_shape = (64, 64, 1)
        self.split = 0.1
        self.verbose = 1
        self.num_classes = 7
        self.patience = 50
        self.ds_name = ds_name
        self.train_data = []
        self.test_data = []
        self.val_data = []
        self.model = mini_XCEPTION(self.img_shape, self.num_classes)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self):#, train_data, val_data, model, patience):
        # need variables declared in previous cell
        # consider overriding modelcheckpoint functions to return last successfully saved model
        # pass dataset name
        # callbacks

        # data generator
        data_generator = ImageDataGenerator(featurewise_center=False,
                                            featurewise_std_normalization=False,
                                            rotation_range=10,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            zoom_range=.1,
                                            horizontal_flip=True)

        log_file_path = self.model_root+self.ds_name + "_" + \
                        str(self.img_shape[0]) + "_" + \
                        str(int(self.split)*100)+'_us_training.log'
        csv_logger = CSVLogger(log_file_path, append=False)
        early_stop = EarlyStopping('val_loss', patience=self.patience)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                      patience=int(self.patience/4),
                                      verbose=1)
        trained_models_path = self.model_root+self.ds_name+"_"+str(self.img_shape[0])+"_"+str(int(self.split*100))+'_us_mini_XCEPTION'
        model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                           save_best_only=True)
        callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

        # loading dataset
        faces, emotions = self.train_data
        # faces = preprocess_input(faces)
        self.model.fit_generator(data_generator.flow(
                            faces, emotions, self.batch_size),
                            steps_per_epoch=len(faces)/self.batch_size,
                            epochs=self.num_epochs, verbose=1,
                            callbacks=callbacks, validation_data=self.val_data)

    def train_with_weights(self, class_weights_dict):#, train_data, val_data, model, patience):
        # need variables declared in previous cell
        # consider overriding modelcheckpoint functions to return last successfully saved model
        # pass dataset name
        # callbacks

        # data generator
        data_generator = ImageDataGenerator(featurewise_center=False,
                                            featurewise_std_normalization=False,
                                            rotation_range=10,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            zoom_range=.1,
                                            horizontal_flip=True)

        log_file_path = self.model_root+self.ds_name + "_" + \
                        str(self.img_shape[0]) + "_" + \
                        str(int(self.split)*100)+'bw_training.log'
        csv_logger = CSVLogger(log_file_path, append=False)
        early_stop = EarlyStopping('val_loss', patience=self.patience)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                      patience=int(self.patience/4),
                                      verbose=1)
        trained_models_path = self.model_root+self.ds_name+"_"+str(self.img_shape[0])+"_"+str(int(self.split*100))+'_bw_mini_XCEPTION'
        model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                           save_best_only=True)
        callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

        # loading dataset
        faces, emotions = self.train_data
        self.model.fit_generator(data_generator.flow(
                            faces, emotions, self.batch_size),
                            steps_per_epoch=len(faces)/self.batch_size,
                            epochs=self.num_epochs, verbose=1,
                            callbacks=callbacks, validation_data=self.val_data,
                            class_weight=class_weights_dict)
        
        def load_saved(self):
            
    # def test(self):
    #     self.model = 
    # def test_images():

    # def test_video()