import matplotlib.pyplot as plt
from matplotlib import interactive
from datapipe import Data
from nn import ANN

DATA_FILENAME = './data/iris-data.txt'
CLASSES       = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
NUM_FEATURES  = 4
NUM_CLASSES   = 3

# Size of samples selected for training the data from each class
TRAINING_SIZE_PER_CLASS  = 30
# Size of samples for validation
VALIDATING_SIZE_PER_CLASS = 10
# Size for batch training, select this many from each class
# NOTE: must be a factor of TRAINING_SIZE_PER_CLASS 
BATCH_SIZE_PER_CLASS      = 15

class TrainANN(object):
    def __init__(self, lr, momentum, init_w_mag, init_bias, epochs):
        self.training_size_per_class = TRAINING_SIZE_PER_CLASS
        self.validing_size_per_class = VALIDATING_SIZE_PER_CLASS
        self.batch_size_per_class    = BATCH_SIZE_PER_CLASS
        self.lr                      = lr
        self.momentum                = momentum
        self.init_w_mag              = init_w_mag
        self.init_bias               = init_bias
        self.num_epochs              = epochs
        self.training_loss_over_epoch = []
        self.testing_loss_over_epoch  = []
        self.accuracy_over_epoch      = []

    def run(self):
        self.setup_data()
        self.nn = ANN(NUM_FEATURES, NUM_CLASSES, self.momentum, self.init_w_mag, self.init_bias)
        steps_per_epoch = self.training_size_per_class // self.batch_size_per_class
        for i in range(self.num_epochs):
            # Train NN
            self.train(steps_per_epoch)
            self.training_loss_over_epoch.append(self.training_loss)
            # Validate NN on validating data
            self.validate()
            self.testing_loss_over_epoch.append(self.nn.error.value)
            # Record accuracy on test set
            self.test()
            self.accuracy_over_epoch.append(self.test_accuracy)

    def setup_data(self):
        self.data = Data(CLASSES, NUM_FEATURES)
        self.data.read_n_process(DATA_FILENAME)
        self.data.split_samples(self.training_size_per_class, self.validing_size_per_class)

    def train(self, iterations):
        training_loss = 0
        # Select n random sample batches to train with
        for n in range(iterations):
            # Randomly sample a batch of examples
            sample_batch, label_batch = \
                self.data.resample(self.batch_size_per_class, 
                                   self.data.training_samples, 
                                   self.data.training_labels)
            # Reset samples and labels for training
            self.nn.sample.value = sample_batch
            self.nn.sample_label.value = label_batch
            # Forward and backward passes to get the gradients
            self.nn.forward_n_backward()
            self.nn.update_weights_n_bias(self.lr)
            # Accumulate loss
            training_loss += self.nn.error.value
        # Average training loss over iterations
        self.training_loss = training_loss/iterations

    def validate(self):
        # Validate NN on validating data
        self.nn.sample.value       = self.data.validation_samples
        self.nn.sample_label.value = self.data.validation_labels
        self.nn.forward()

    def test(self):
        self.nn.sample.value = self.data.testing_samples
        self.nn.sample_label.value = self.data.testing_labels
        self.nn.forward()
        self.test_accuracy = self.nn.accuracy()

    def draw_loss_graph(self):
        x = range(len(self.training_loss_over_epoch))
        y1 = self.training_loss_over_epoch
        y2 = self.testing_loss_over_epoch
        
        plt.subplot(211)
        plt.plot(x, y1,'b-', label='Training Set')
        plt.plot(x, y2, 'r:', label='Validation Set')
        # Create empty plot with blank marker containing the extra label
        plt.plot([],[], ' ', label=f"LEARNING_RATE: {self.lr}")
        plt.plot([],[], ' ', label=f"MOMENTUM: {self.momentum}")
        plt.plot([],[], ' ', label=f"INIT_WEIGHT_MAG: {self.init_w_mag}")
        plt.plot([],[], ' ', label=f"INIT_BIAS: {self.init_bias}")
        plt.legend(loc='upper right', fontsize=8)
        
        plt.title('Loss Over Time', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        
    def draw_accuracy_graph(self):
        x = range(len(self.accuracy_over_epoch))
        y = self.accuracy_over_epoch

        plt.subplot(212)
        plt.plot(x, y,'-', label='Testing Set')
        plt.legend(loc='lower right', fontsize=8)
        
        plt.title('Test Set Accuracy Over Time', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)

    def plot_n_save_graphs(self, filename):
        self.draw_loss_graph()
        self.draw_accuracy_graph()
        plt.subplots_adjust(top=0.93, hspace=0.5)
        plt.savefig(filename)
        plt.show()

    def print_test_set_with_labels(self):
        print(self.data.labeled_testing_set)

    def predict_samples(self, samples):
        prediction = self.nn.run_samples(samples)
        return [self.data.encoding_to_class(v) for v in prediction]