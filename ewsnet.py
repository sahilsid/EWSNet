
import os
import tensorflow as tf 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Activation, GRU, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import numpy as np 

class EWSNet():
    def __init__(self,ensemble=1, weight_dir=None, prefix="",suffix=".h5"):
        self.ensemble = ensemble
        self.model    = [self.build_model() for _ in range(self.ensemble)]
        if weight_dir is not None:
            self.load_model(weight_dir,prefix,suffix)
        self.labels=["No Transition","Smooth Transition","Critical Transition"]

    def build_model(self):
        ip = Input(shape=(1, None))
        x = Permute((2, 1))(ip)
        x = LSTM(128)(x)
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
        out = Dense(3, activation='softmax',kernel_regularizer=regularizers.l2(0.001))(x)
        model = Model(ip, out)
        return model

    def load_model(self,weight_dir,prefix,suffix):
        print("=="*30)
        for i in range(self.ensemble):
            print("Loading Model : ","{}/{}{}{}".format(weight_dir,prefix,i+1,suffix))
            if(os.path.exists("{}/{}{}{}".format(weight_dir,prefix,i+1,suffix))):
                self.model[i] = tf.keras.models.load_model("{}/{}{}{}".format(weight_dir,prefix,i+1,suffix))
            else:
                raise NameError
        print("=="*30)
        
    def predict(self,x):
        x = np.array(x)
        x = np.reshape(x,(1,1,x.shape[0]))
        predictions = np.array([self.model[i](x)[0] for i in range(self.ensemble)])
        predictions = np.mean(predictions,axis=0)
        prediction_probability = {
            "No Transition"      :predictions[0],
            "Smooth Transition"  :predictions[1],
            "Critical Transition":predictions[2],
        }
        return self.labels[np.argmax(predictions)],prediction_probability

if __name__ == '__main__':
    
    weight_dir = "./weights/Pretrained"
    dataset    = "W"
    prefix     = ""
    suffix     = ".h5"
    ensemble   = 25

    ewsnet     = EWSNet(ensemble=ensemble, weight_dir=os.path.join(weight_dir,"Dataset-{}".format(dataset)), prefix=prefix,suffix=suffix)
    x = np.random.randint(1,100,(20,))
    print(ewsnet.predict(x))