from .Data import Data
import os
from imblearn.over_sampling import SMOTE
import numpy as np

class Classifier_data(Data):

    def __init__(self, batch_size=1, data=None, labels=None):
        super().__init__(batch_size)
        self.Data = data
        self.Labels = labels

        self.Num_of_samples = labels.shape[0]
        self.Num_of_batches = int(self.Num_of_samples / self.Batch_size) + 1


        #Get classes and one_hot encoding
        self.classes = np.unique(labels)
        self.Num_of_classes = len(self.classes)
        self.One_hot_labels = Classifier_data.one_hot(labels, self.Num_of_classes)

        self.Num_of_features = self.Data.shape[1]


        return

    ## Get one hot encoding for the data labels
    @staticmethod
    def one_hot(labels, Num_of_classes):
        One_hot_labels = np.zeros((labels.shape[0], Num_of_classes))
        One_hot_labels[np.arange(labels.shape[0]), labels.transpose()] = 1
        return One_hot_labels

    ## Shuffle the data and labels randomly
    def shuffle(self, seed=00000):
        np.random.seed(seed)
        np.random.shuffle(self.Data)

        np.random.seed(seed)
        np.random.shuffle(self.Labels)
        return

    ## Function to upsample data to manage data imbalance
    def balanceData(self, seed=00000):
        # SMOTE - Synthetic Minority Over-Sampling Technique
        sm = SMOTE(random_state = seed, ratio=1.0)

        # The balanced data is our new dataset
        new_data, new_labels = sm.fit_sample(self.Data, self.Labels)
        new_labels = new_labels.reshape(new_labels.shape[0], 1)

        # Reassign values to the instance variables
        self.Labels = new_labels
        self.Data = np.float32(new_data)
        self.One_hot_labels = Classifier_data.one_hot(self.Labels, self.Num_of_classes)

        self.Num_of_samples = self.Labels.shape[0]
        self.Num_of_batches = int(self.Num_of_samples / self.Batch_size) + 1

        # Print the number of samples in each class
        unique, counts = np.unique(new_labels, return_counts = True)
        print(dict(zip(unique, counts)))
        return

    ## Split the data given a splitting fraction
    def splitData(self, fraction, seed=00000, shuffle= True):
        if(shuffle):
            self.shuffle(seed = seed)

        # Compute cumulative sum of fraction and multiplies with number of samples
        ranges = np.cumsum(fraction)
        ranges = (self.Num_of_samples * ranges).astype(int)

        # Split data and label using numpy split
        dataSplits = np.split(self.Data, ranges)
        labelSplits = np.split(self.Labels, ranges)

        readers = []
        for i in range(0, len(fraction)+1):

            # Create new reader
            readers.append(Classifier_data(
                batch_size = self.Batch_size,
                data = dataSplits[i],
                labels = labelSplits[i]
                ))

            print('Split with %d members' % labelSplits[i].shape[0])
        return readers

    ## Read the data from a given file
    def load_from(self, data_file, class_file=0):
        g = pd.read_csv(data_file)
        if class_file:
            l = pd.read_csv(class_file)

    ## Get data batches for training/validation/testing
    def getBatch(self, idx):
        start = idx * self.Batch_size
        end = (idx + 1) * self.Batch_size

        Batch = {'X' : self.Data[start:end, :],
                'Y' : self.One_hot_labels[start:end,:]}
        return Batch

    ## Set the instance variable for batch size
    def setBatchSize(self, size):
        self.Batch_size = batch_size
        self.Num_of_samples = labels.shape[0]
        # Calculate number of batches
        self.Num_of_batches = int(self.Num_of_samples / self.Batch_size) + 1
