

class EWSNet():
    def __init__(self,ensemble=1, weight_dir=None, prefix="TRIAL-",suffix="_weights.h5"):
        self.ensemble = ensemble
        self.model    = [self.build_model() for _ in range(self.ensemble)]
        if self.weight_dir is not None:
            self.load_model(weight_dir,prefix,suffix)

    def build_model(self,NB_CLASS=3, NUM_CELLS=128):
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

    def load_model(self,weight_dir,prefix,suffix):
        for i in range(self.ensemble):
            self.model[i].load_weights("{}/{}{}{}".format(weight_dir,prefix,suffix))
    
