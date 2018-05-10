# Iris Flower Classifier

[//]: # (Image References)
[ANNDiagram]: ./img/nn_graph.png "ANN Layout"
[Example]: ./img/example.png "Example error curve over training time"
[ExampleWithMomentum]: ./img/example_with_momentum.png "Example with momentum"
[InterestingCurve1]: ./img/interesting_curve.png "Interesting error curve over training time"
[InterestingCurve2]: ./img/interesting_curve2.png "Interesting error curve over training time"
[InterestingCurve3]: ./img/interesting_curve3.png "Interesting error curve over training time"
[InterestingCurve5]: ./img/interesting_curve5.png "Interesting error curve over training time"

## Problem Statement:

**To build a neural network based classifier for identifying the three types of Iris flowers**: Iris Setosa, Iris Versicolour, and Iris Virginica, given the following features:

Sepal length, sepal width, petal length, and petal width.


## Dependencies:

1. Python3
2. Numpy      
3. Matplotlib 


## Run:

    python irisclassifier.py [-t]
        -t {training mode}: Manipulate the hyperparameters that trains the classifier


## Example Output:

![ExampleWithMomentum]

## Input Data:

The data comes from Fisher’s Iris database (Fisher, 1936) which consisted of 50 instances for each of the three types of Iris in the following format

`[Sepal-Length, Sepal-Width, Petal-Length, Petal-Width, Iris-Type]`

The data is separated into three sets for cross validation: a training set, a validating set and a testing set. All sets contain an equal distribution of samples from all three classes.

The ANN will not be exposed to samples from both the validating and testing sets during training. The validating set is used to examine the model on overfitting issues and the testing set is used to calculate the accuracy of the model. 

You can manipulate the size of the three sets by changing the global constants `TRAINING_SIZE_PER_CLASS` and  `VALIDATING_SIZE_PER_CLASS` in the file `training.py`. The size of the testing set per class is the number of remaining samples: `50 - TRAINING_SIZE_PER_CLASS - VALIDATING_SIZE_PER_CLASS`.


## Details of Implementation:

### Neural Net Layout
The neural network is consisted of an input layer that accepts four inputs, two hidden layers (the first hidden layer has 5 nodes and the second has 2 nodes) and an output layer with three nodes. All nodes are fully connected in each layer.

![ANNDiagram]

Image source: https://www.neuraldesigner.com/learning/examples/iris_flowers_classification

The output layer produces a distribution of probability on the three labels and makes the prediction by choosing the label with the largest probability. 

Labels of the sample is represented using one-hot encoding. It is a vector of bits with length equal to the number of possible classes that contains one 1 and the rest are 0s. A 1 at the specific index of the vector represent a label for a specific class.

For example:

    [1, 0, 0] encodes Iris Setosa
    [0, 1, 0] encodes Iris Versicolour
    [0, 0, 1] encodes Iris Virginica
    

### Neural Net Setup

| Activation Function |  Logistic Function  |  Error Function   |
|  :---------------:  |  :---------------:  |  :-------------:  |
|        ReLU         |       Softmax       |  Cross Entropy    |

I use a stable version softmax function that normalizes the inputs by the maximum of the inputs. The error of the error is calculated using the cross entropy loss. 


## Training:

### Inital Setup
    
    LEARNING_RATE             = 0.0004
    MOMENTUM                  = 0.5
    # Initialize weight in the range [0, 0.1]
    INIT_WEIGHT_MAG           = 0.1
    INIT_BIAS                 = 0.1
    # Size of samples selected for training the data from each class
    TRAINING_SIZE_PER_CLASS   = 30
    # Size of samples for validation
    VALIDATING_SIZE_PER_CLASS = 10 
    # Size for batch training, select this many from each class
    # NOTE: must be a factor of TRAINING_SIZE_PER_CLASS 
    BATCH_SIZE_PER_CLASS      = 15
    # Maximum length of time to train the NN
    EPOCHS                    = 1000

### Training Strategy

The initial hyperparameter above is obtained through simple trial and error in finding a optimal range of hyperparameters. 

The neural net is trained using the **batch method**, where a fixed number of samples are randomly selected from the training set and forward propagate through the net simultaneously.

As a result, the output error of the NN is an average error across the batch of samples, and the gradient on the weights are accumulative across all the samples in the batch. We can compensate the larger weight gradients with a smaller learning rate. 

In addition, at each iteration of training, weights are updated using **momentum descent** with a fixed momentum.

The "correctness" of the model's prediction is obtained by summing over the product of the one-hot encoding vector of the correct label and the prediction vector.

The accuracy for the model over the test set is obtained by averaging the "correctness" for each test sample.

For example:

    y = [1, 0, 0]` is the label for the sample
    p = [0.91, 0.05, 0.04] is the prediction vector

    correctness = sum(y * p) = 0.91

    accuracy = sum of correctness over test set / number of sample in test set


## Reflection:

It is interesting for me to see how a small modification on the hyperparameter can greatly impact the performance of the model. I evolve the evaluation of my model from simply judging on the loss function over time to include the accuracy function over time. I start by using stochastic gradient descent. I find that the model could do pretty well but it sometimes does not converge to a low loss because it is stuck on local minimum. An increase in learning rate may help, but a higher learning rate tend to destabilize the learning progress.

Without further elongating the training time or increasing the learning rate, I try momentum descent. Using the momentum factor of 0.5, this method works really well increasing the rate of information gain by the model. You can compare the effect of momentum descent in the following two graphs. 

On top of momentum descent, I also experiment learning rating annealing, where I perform step decaying of the learning rate over the training session. The result of this combination is not very good. Although I did not extensively test it, the higher learning rate at the beginning of the training create huge fluctuation in the loss function, and it could simply jump over the local minima. It becomes difficult to find the set of hyperparameters under which the model will converge to a small loss function. In addition, I am satisfied with the learning rate of the model with momentum descent. Adding learning rate annealing may be an overkill to this problem.

#### Without Momentum Descent
![Example]
#### With Momentum Descent
![ExampleWithMomentum]

The loss converges much faster with momentum descent


## More Interesting Curves:

![InterestingCurve1]

Probably over-trained.

![InterestingCurve2]

Local minimum detected early in the training. 

![InterestingCurve3]

Overfitting.

![InterestingCurve5]

Overfitting but following a recovery.


## Rooms for Improvement

1. Data processing

    I could zero-center and normalize the data before training.

2. Compare the result with using other activation functions

    I use ReLU as my activation function, but I could also build the ANN with other activations such as `tanh` and `sigmoid` to compare their effect on the ANN. 

3. Batch Normalization

    To stabilize the neuron's action potential in the active range of `tanh` and `sigmoid`, I can also insert batch normalization across all nodes' output.

4. Combination of Learning Rate Annealing and Momentum Annealing

    For exploration purpose, I can try to slowly decay the learning rate and at the same time grow the momentum over the training time. 