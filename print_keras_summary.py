
# uncomment to run on CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from keras.models import load_model



def keras_load(m_path):
    model = load_model(m_path)
    model.summary()
    return model


model = "/home/ubuntu/Repos/fer_tools/models/FER2013_64_10_rw_mini_XCEPTION.48-0.74.hdf5"
keras_load(model)
