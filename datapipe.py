import csv
import numpy as np
from random import sample

class Data(object):
    """
    Abstraction for the training data for classification
    """
    def __init__(self, classes, input_dimension):
        """
        Initialize the Data object. 
    
        Arguments:
            `classes`        :  a array of all possible classes in the data
            `input_dimension`: a integer indicating the number of input dimensions
                               of training a classification NN
        """
        self.num_classes  = len(classes)
        self.class_dict = classes
        self.input_dimension = input_dimension

    def read_n_process(self, filename):
        """
        Read a .csv file and sets two parameters
        
        1. `self.samples` : a (n x k) array where n is the 
                          number of data points and k is the 
                          dimension of input data points

        2. `self.labels` : a (n x 1) array of one-hot encoding vector 
                          for the label of each data point
                           
        3. `self.d`     : a (n x m) array where n is the 
                          number of data points and m is the 
                          number of categories in a row of
                          the csv file.
       
        Each row is categorized in the following way:

        [sepal-length, sepal-width, petal-length, petal-width, class]

        where `class` is a one-hot encoding vector of the three categories:
        1. Iris Setosa
        2. Iris Versicolour
        3. Iris Virginica

        Ex) [1, 0, 0] is the encoding for Iris Setosa

        NOTE: all elements in self.d are type string
              but elements in input are converted to float when possible
        
        Arguments:
            `filename` : string
        """
        data = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                row = line
                data.append(row)
        
        self.d = np.array(data)
        self.num_samples = self.d.shape[0]
        self.samples_per_class = self.num_samples // self.num_classes
        # Pool of all available lookup indices for every class
        self.index_pool = set(range(self.samples_per_class))
        # Input is the first n column of the data
        self.samples = np.array(self.d[:, :self.input_dimension], dtype=np.float)
        # The last column is the label
        self.labels = np.array([self.one_hot_encode(s) \
                               for s in self.d[:, self.input_dimension:]])

    def one_hot_encode(self, s):
        """
        Given the string of a class, return the one-hot-encoding
        of that class
        """
        v = [0] * self.num_classes
        for index, class_name in enumerate(self.class_dict):
            if s == class_name:
                v[index] = 1
        return v

    def encoding_to_class(self, v):
        """
        Return the class associated with the one-hot encoding vector
        """
        index = np.argmax(v)
        return self.class_dict[index] 

    def split_samples(self, training_size, validation_size):
        """
        Randomly split the samples into training and testing sample sets.
        Not overlap between training, validating and testing set.

        Assume sample pools to have uniform distribution of sample across 
        all classes.
    
        Arguments:
           `training_size` : number of training samples per class
           `validation_size` : number of samples used for validation
                              (NOTE: the remaining samples are for testing)
        """
        assert training_size + validation_size <= self.samples_per_class, \
                "Training size + validiont size cannot exceed total number of samples"
        training_indices     = sample(self.index_pool, training_size)
        nontraining_indices  = self.index_pool.difference(set(training_indices))
        validation_indices   = list((sample(nontraining_indices, validation_size)))
        testing_indices      = list(nontraining_indices.difference(validation_indices))
        self.training_samples, self.training_labels = \
                self.select_samples_n_labels_from_classes(training_indices, self.samples, self.labels)
        self.validation_samples, self.validation_labels = \
                self.select_samples_n_labels_from_classes(validation_indices, self.samples, self.labels)
        self.testing_samples, self.testing_labels   = \
                self.select_samples_n_labels_from_classes(testing_indices, self.samples, self.labels)
        # A test set with str labels. For printing purpose
        self.labeled_testing_set = self.select_raw_data_with_labels(testing_indices)

    def resample(self, samples_per_class, sample_pool, label_pool):
        """
        Randomly select a batch of sample from the pool of data
        with equal distribution across all classes.

        Assuming data has equal distribution of samples across all classes
        and that every element from the same class is neighboring each other in the pool
        
        Example:
            [ClassA1, ClassA2, ..., ClassB1, ClassB2, ..., ClassC1, ClassC2, ...]
        
        Arguments:
            `sample_per_class` : (NOTE should be a multiple of the number of classes)
            `sample_pool` : a (n x k) array of samples to randomly select from
            `label_pool`  : a (n x k) array of sample labels
        """
        index_pool = range(sample_pool.shape[0] // self.num_classes)
        batch_indices = sample(index_pool, samples_per_class)
        return self.select_samples_n_labels_from_classes(batch_indices, sample_pool, label_pool)

    def select_samples_n_labels_from_classes(self, indices, sample_pool, label_pool=None):
        """
        Select the element from each classes in the sample_pool at each index

        Assume equal distribution of sample across all classes in the sample_pool
        and that every element from the same class is neighboring each other in the pool
        
        Example:
            [ClassA1, ClassA2, ..., ClassB1, ClassB2, ..., ClassC1, ClassC2, ...]

        Arguments:
            `indices` : an array of indices of the sample pool
            `sample_pool` : an array of samples
            `label_pool`  : an array of sample labels (must be same size as `sample_pool`)
        
        """
        samples_per_class = sample_pool.shape[0] // self.num_classes
        assert len(indices) <= samples_per_class, \
            "Number of selected samples per class cannot exceed total number of samples per class"
        
        sample_batch = np.array([])
        label_batch  = np.array([])
        for i in range(len(indices)):
            for j in range(self.num_classes):
                sample_index = indices[i] + j * samples_per_class
                sample_batch = np.vstack((sample_batch, sample_pool[sample_index])) \
                                if sample_batch.size else sample_pool[sample_index] 
                label_batch = np.vstack((label_batch, label_pool[sample_index])) \
                                if label_batch.size else label_pool[sample_index]
        return (sample_batch, label_batch)
    
    def select_raw_data_with_labels(self, indices):
        """
        Select the element from the raw input set (self.d)
        at each index
        """
        samples_per_class = self.d.shape[0] // self.num_classes
        data_batch  = np.array([])
        for i in range(len(indices)):
            for j in range(self.num_classes):
                data_index = indices[i] + j * samples_per_class
                data_batch = np.vstack((data_batch, self.d[data_index])) \
                                if data_batch.size else self.d[data_index] 

        return data_batch