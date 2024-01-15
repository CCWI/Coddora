import tensorflow as tf
import numpy as np
import dask.array as da
import logging

class Generator(tf.keras.utils.Sequence):
    """
    The generator is used during the training process to read the 
    upcoming samples and provide the next batch on the fly.
    This eliminates the requirement to fit all the data into memory.
    """

    def __init__(self, data, metadata, labels, batch_size, 
                 features=1, metafeatures=3, 
                 window_size=30, shuffle=False,
                 logfile=None):
        self.data = data
        self.metadata = metadata
        self.labels = labels
        self.index_list = np.arange(0, len(data))
        self.batch_size = batch_size
        self.features = features
        self.metafeatures = metafeatures
        self.window_size = window_size
        self.shuffle = shuffle
        self.batches = len(data) // batch_size
        self.calls = 0
        self.epoch = 0
        if logfile != None: # apply logging if logfile is passed
            logging.basicConfig(filename=logfile, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.on_epoch_end() # initial shuffle
        
    def on_epoch_end(self):
        logging.info('Epoch end.')
        if self.shuffle: # shuffling sequences (shuffling batches is done by model.fit function)
            self.epoch += 1
            np.random.seed(self.epoch) # use epoch number as random seed
            np.random.shuffle(self.index_list)
            logging.info('Shuffled samples for new epoch.')

    def __len__(self):
        return int(np.floor(len(self.index_list) / self.batch_size))

    def __generate(self, ids):
        x = da.take(self.data, ids).compute()
        x_meta = da.take(self.metadata, ids).compute()
        y = da.take(self.labels, ids).compute()
        return (x, x_meta), y
    
    def __getitem__(self, i):
        """Provides a new batch"""
        self.calls += 1
        if self.calls % int(self.batches/100) == 0:
            logging.info('Prepared {}/{} batches.'.format(self.calls, self.batches))
        ids = self.index_list[i * self.batch_size : (i+1) * self.batch_size]
        return self.__generate(ids)