import os
import abc
import numpy as np

class Data:
# Abtract class for Data.  Forces implementation of certain assumed features

    def __init__(self, batch_size=1):
        self.Batch_size = batch_size
        self.Num_of_samples = None
        self.Num_of_batches = None

    def setBatchSize(self, size):
        self.Batch_size = size

    @abc.abstractmethod
    def getBatch(self, idx):
        pass

    @abc.abstractmethod
    def splitData(self, fraction, seed=00000):
        pass

    @abc.abstractmethod
    def shuffle(self, seed=00000):
        pass

    def load_from(self, location, batch_size=1):
        print('No \'load_from\' function written')
        return
