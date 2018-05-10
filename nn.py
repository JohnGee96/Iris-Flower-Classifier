import numpy as np
import neuron

# First hidden layer
HIDDEN_NODES  = 5

class ANN(object):
    """
    An abstraction for the neural network trained 
    using batched inputs and stochastic gradient descent
    for classifying Iris
    """
    def __init__(self, num_features=4, num_classes=3, momentum=0.5, init_weight_mag=1e-3, init_bias=1e-4):
        # Dimension of the first hidden layer
        D1 = num_features
        # Number of nodes in the first hidden layer
        H1 = HIDDEN_NODES
        # Dimension of the second hidden layer
        D2 = HIDDEN_NODES
        # Number of nodes in the second hidden layer
        H2 = num_classes
        # Initialize weights and biase for each hidden layer
        W1 = init_weight_mag * np.random.randn(D1, H1)
        B1 = init_bias + np.zeros((1, H1))
        W2 = init_weight_mag* np.random.randn(D2, H2)
        B2 = init_bias + np.zeros((1, H2))

        # Setup network's input nodes
        x, y = neuron.Input(), neuron.Input(), 
        w1, w2, b1, b2, = neuron.Input(), neuron.Input(), neuron.Input(), neuron.Input()

        # Hidden layers
        h1 = neuron.Linear(x, w1, b1)
        f1 = neuron.ReLU(h1)
        h2 = neuron.Linear(f1, w2, b2)
        f2 = neuron.ReLU(h2)
        # Output layer
        s = neuron.Softmax(f2)
        e = neuron.CEL(y, s)

        # A dictionary that sets the value of all Input nodes in the NN
        feed_dict = {
            x: None, # Set during training
            y: None, # Set during training
            w1: W1,
            w2: W2,
            b1: B1,
            b2: B2
        }
        # Contains topologically sorted nodes in the neural network
        self.graph         = neuron.topological_sort(feed_dict)
        self.sample        = x
        self.sample_label  = y
        self.num_clases    = num_classes
        self.w1, self.b1   = w1, b1
        self.w2, self.b2   = w2, b2
        self.trainables    = [w1, b1, w2, b2]
        self.prediction    = s
        self.error         = e
        self.momentum      = momentum
        self.prior_update  = 0
        
    def forward(self):
        neuron.forward(self.graph)

    def forward_n_backward(self):
        neuron.forward_and_backward(self.graph)

    def update_weights_n_bias(self, lr):
        neuron.sgd_update_with_momentum(self.trainables, lr, self.momentum)

    def accuracy(self):
        label_accuracy = np.multiply(self.sample_label.value, self.prediction.value)
        # print(correct_label)
        # print(total_diff)
        correct_labels = sum(sum(label_accuracy))
        num_samples = self.sample.value.shape[0]
        return correct_labels / num_samples

    def run_samples(self, samples):
        """
        Run the neural on a set of samples. 
        Assume samples is not a numpy.array object
        """
        num_samples = len(samples)
        template_label = np.zeros((num_samples, self.num_clases))
        self.sample.value = np.array(samples)
        self.sample_label.value = template_label
        self.forward()
        return self.prediction.value

