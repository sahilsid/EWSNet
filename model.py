import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Activation, GRU, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow  as tf
import pickle
from random import randint
import tensorflowjs as tfjs

def generate_dynamic_lstmfcn(NB_CLASS, NUM_CELLS=128):
    ip = Input(shape=(1, None))
    x = Permute((2, 1))(ip)
    x = LSTM(NUM_CELLS)(x)
    x = Dropout(0.2)(x)
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])
    x = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    out = Dense(NB_CLASS, activation='softmax',kernel_regularizer=regularizers.l2(0.001))(x)
    model = Model(ip, out)
    return model

epochs           = 25
LR               = 5e-5 
batch_size       = 512 
N_TRIALS         = 25
TRAINING        = True

for trial_no in range(1,N_TRIALS+1):
    seed = randint(0,1e3)
    tf.random.set_seed(seed)
    dataset_map      = [('T11-NOISE/TRIAL-{}'.format(trial_no),0),('T12-GAUSSIAN/TRIAL-{}'.format(trial_no),1)]
    base_log_name    = '%s_%d_cells_new_datasets.csv'
    base_weights_dir = '%s_%d_cells_weights/'
    normalize_dataset = False
    MODELS = [('dynamic_lstmfcn',generate_dynamic_lstmfcn),]
    CELLS  = [128]
    for model_id, (MODEL_NAME, model_fn) in enumerate(MODELS):
        for cell in CELLS:
            for dname, did in dataset_map:
                NB_CLASS            = 3
                K.clear_session()                    
                weights_dir = base_weights_dir % (MODEL_NAME, cell)
                os.makedirs('weights/' + weights_dir,exist_ok=True)
                dataset_name_ = weights_dir + dname
                model         = model_fn(NB_CLASS, cell)
                optm          = tf.keras.optimizers.Adam()
                model.compile(optimizer=optm, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.load_weights("./weights/%s_weights.h5" % dataset_name_)
                
                os.makedirs("weights/Pretrained/tfjs/Dataset-W/",exist_ok=True)
                os.makedirs("weights/Pretrained/tfjs/Dataset-C/",exist_ok=True)

                os.makedirs("weights/Pretrained/Dataset-W/",exist_ok=True)
                os.makedirs("weights/Pretrained/Dataset-C/",exist_ok=True)
                if("GAUSSIAN" in dname):
                    model.save("weights/Pretrained/Dataset-W/{}.h5".format(trial_no))
                    model =  tf.keras.models.load_model("weights/Pretrained/Dataset-W/{}.h5".format(trial_no))
                    tfjs.converters.save_keras_model(model, "weights/Pretrained/tfjs/Dataset-W/{}".format(trial_no))
                else:
                    model.save("weights/Pretrained/Dataset-C/{}.h5".format(trial_no))
                    model = tf.keras.models.load_model("weights/Pretrained/Dataset-C/{}.h5".format(trial_no))
                    tfjs.converters.save_keras_model(model, "weights/Pretrained/tfjs/Dataset-C/{}".format(trial_no))