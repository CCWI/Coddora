# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, Flatten, Dropout, \
    MaxPooling1D, Bidirectional, LSTM, Input, Activation, concatenate
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import numpy as np


class CDBLSTM(Model):

    # Training Parameters
    window_size = 30
    seed = 0
    batch_size = 128
    optimizer = optimizers.Adam()
    verbose = 1

    # Hyperparameters
    filters = [200, 50]
    kernel_size = [5, 3]
    pool_size = 2
    lstm_neurons = [50, 50, 50]
    fc_neurons = [100]
    dropout_rates = [0.5]


    def __init__(self, classes=2, features=1, metafeatures=1, *args, **kwargs):
        '''
        Parameters
        ----------
        classes : TYPE, optional
            Number of output classes. The default is 2.
            2 is a binary classification (occupancy detection).
            4, e.g., can be a classification into high, medium, low, none
              (occupancy estimation).
        features : TYPE, optional
            Number of features, e.g., sensor modalities. The default is 1.
        '''
        super(CDBLSTM, self).__init__()

        ## Parameter Setting
        self.classes = classes
        self.features = features
        self.metafeatures = metafeatures
        self.__dict__.update((k, v) for k, v in kwargs.items())
        print([k + "=" + str(v) for k, v in kwargs.items()])

        ## Model Layers ------------------------------------------

        # CNN
        self.conv1D_layer1 = Conv1D(filters=self.filters[0], kernel_size=self.kernel_size[0],
                                    input_shape=(self.window_size, self.features),
                                    activation='relu', name='conv1D_layer1')
        self.pooling_layer1 = MaxPooling1D(pool_size=self.pool_size,
                                           name='pooling_layer1')
        self.conv1D_layer2 = Conv1D(filters=self.filters[1], kernel_size=self.kernel_size[1],
                                    input_shape=(self.window_size, self.features),
                                    activation='relu', name='conv1D_layer2')
        self.pooling_layer2 = MaxPooling1D(pool_size=self.pool_size,
                                           name='pooling_layer2')

        # BLSTM
        self.blstm_layer1 = Bidirectional(LSTM(self.lstm_neurons[0], return_sequences=True),
                                          name='blstm_layer1')
        self.blstm_layer2 = Bidirectional(LSTM(self.lstm_neurons[1], return_sequences=True),
                                          name='blstm_layer2')
        self.blstm_layer3 = Bidirectional(LSTM(self.lstm_neurons[2], return_sequences=False),
                                          name='blstm_layer3')

        # Fully Connected Dense Network
        self.dropout_layer = Dropout(self.dropout_rates[0], seed=self.seed,
                                     name='dropout_layer')
        self.dense_layer = Dense(self.fc_neurons[0], activation='relu', kernel_initializer='uniform',
                                 name='dense_layer')
        
        self.output_layer = Dense(1, activation='sigmoid', name='output')


        ## Model Compilation ------------------------------------------
        prediction_loss = 'binary_crossentropy' if classes < 3 else \
            'sparse_categorical_crossentropy'

        self.compile(optimizer=self.optimizer,
                                     loss=prediction_loss,
                                     metrics=['accuracy'])

        ## ------------------------------------------------------------


    def call(self, inputs):
        
        if (type(inputs) in [tuple, list]) & (np.shape(inputs) == 2):
            meta_mode = True
        elif np.shape(inputs) == (2,):
            meta_mode = True
        else:
            meta_mode = False
            
        if meta_mode:
            input_timeseries, input_metadata = inputs
        else:
            print("Time Series Only Mode (no metadata considered)")
            input_timeseries = inputs
              
        # CNN
        x = self.conv1D_layer1(input_timeseries)
        x = self.pooling_layer1(x)
        x = self.conv1D_layer2(x)
        x = self.pooling_layer2(x)
        # BLSTM
        x = self.blstm_layer1(x)
        x = self.blstm_layer2(x)
        x = self.blstm_layer3(x)
        # Concatenate with Metadata
        if meta_mode:
            input_metadata = tf.cast(input_metadata, tf.float32)
            x = concatenate([x, input_metadata])
        # Fully Connected Dense Network
        x = self.dropout_layer(x)
        x = self.dense_layer(x)
        return self.output_layer(x)

    
    def predict_classes(self, inputs, threshold=0.5):
        y_proba = self.predict(inputs)
        return np.where(y_proba > threshold, 1, 0)
    
