import numpy as np

class Node(object):
    """
    Base class for nodes in the network.

    Arguments:
        `inbound_nodes`: A list of nodes with edges into this node.
    """
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.value = None
        self.outbound_nodes = []
        # Keys are the inputs to this node and
        # their values are the partials of this node with
        # respect to that input.
        self.gradients = {}
        # Sets this node as an outbound node for all of
        # this node's inputs.
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError


class Input(Node):
    """
    A generic input into the network.
    """
    def __init__(self):
        # self.value is set during `topological_sort` later.
        Node.__init__(self)
        # save previous gradient change for momentum descent
        self.prev_update = 0

    def forward(self):
        pass

    def backward(self):
        # An Input node has no inputs so the gradient (derivative) is zero.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_nodes:
            self.gradients[self] += n.gradients[self]


class Linear(Node):
    """
    Represents a node that performs a linear transform: Wx + b
    """
    def __init__(self, X, W, b):
        """ Initialize a node that does linear combination
        
        Arguments:
            `X` : a (m x n) matrix where m is the batch size, 
                              and n is the number of input features
            `W` : a (n x 1) matrix where n is the number of features
            `b` : a (n x 1) matrix of floating point numbers  
        """

        # The base class (Node) constructor. Weights and bias
        # are treated like inbound nodes.
        Node.__init__(self, [X, W, b])

    def forward(self):
        """
        Performs the math behind a linear transform.
        """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize the local partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value, dtype=np.float) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the output node with respect to this node 
            # from the output node
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class ReLU(Node):
    """
    Represents a node that performs the ReLU activation function
    """
    def __init__(self, node):
        """ Initialize a node to perform ReLU activation
        
        Arguments:
            `node` : a node containg a (m x 1) array as its value, 
                     where m is the batch size.
                     Usually, it is a Linear Node
        """
        Node.__init__(self, [node])

    def _relu(self, xs):
        # return a (1 x m) array
        return [max(0, x) for x in xs]

    def forward(self):
        scores = []
        for values in self.inbound_nodes[0].value:
            scores.append(self._relu(values))
        self.value = np.array(scores)

    def backward(self):
        # Initialize the local partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value, dtype=np.float) for n in self.inbound_nodes}
        # Derivative of ReLu depends on this node's output value,
        # 0 if negative otherwise 1
        grad_relu = self.value >= 0
        for n in self.outbound_nodes:
            # Get the partial of the output node with respect to this node 
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to the input node
            # and sum gradients over all outputs
            self.gradients[self.inbound_nodes[0]] += grad_relu * grad_cost


class Sigmoid(Node):
    """
    Represents a node that performs the sigmoid activation function.
    """
    def __init__(self, node):
        # The base class constructor.
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value, dtype=np.float) for n in self.inbound_nodes}
        sigmoid = self.value
        # Sum the partial with respect to the input over all the outputs.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost


class Softmax(Node):
    def __init__(self, node):
        """ 
        The softmax of scores
        """
        Node.__init__(self, [node])
        
    def _softmax(self, scores):
        exps = np.exp(scores)
        return np.divide(exps, exps.sum())

    def stable_softmax(self, scores):
        exps = np.exp(scores - np.max(scores))
        return exps / np.sum(exps)

    def forward(self):
        result = np.array([])
        # Collect scores from each input node
        for scores in self.inbound_nodes[0].value:
            dist = self.stable_softmax(scores)
            result = np.vstack((result, dist)) if result.size else np.array([dist])        
        self.value = result

    def backward(self):
        """ 
        Stochastic gradient descent:
        sets the gradient w.r.t input nodes for each sample as a vector of size m
        where m is the batch size
        """
        # Initialize gradient
        batch_size     = self.value.shape[0]
        num_classes    = self.value.shape[1]
        s, gradient    = self.value, np.array([])
        self.gradients = {n: np.zeros((batch_size, num_classes), dtype=np.float) \
                         for n in self.inbound_nodes}    
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            for i in range(num_classes):
                dcost_input = np.zeros(batch_size)
                for j in range(num_classes):
                    s_i, s_j = s[:,i], s[:,j]
                    # Set the partial of the loss with respect to this node's inputs. 
                    dcost_input += grad_cost[:,j] * (s_i * ((j==i) - s_j))
                
                gradient = np.vstack((gradient, dcost_input)) if gradient.size else np.array([dcost_input])
            
            self.gradients[self.inbound_nodes[0]] += gradient.T


class CEL(Node):
    def __init__(self, y, p):
        """
        The cross entropy loss function.
        Should be used as the last node for a network
        
        Arguments:
            `y` : a (m x k) matrix of expected probability 
                  distribution for all the classes, 
                  where m is batch size and k is number of classes 
            `p` : a (m x k) matrix estimate probability 
                  distribution output from the network,
                  where m is batch size and k is number of classes
        """
        Node.__init__(self, [y, p])

    def sanity_check(self):
        y = self.inbound_nodes[0]
        p = self.inbound_nodes[1]
        assert y.value.shape == p.value.shape, f"Matrices of expected probability (y) and "  \
                                               f"output probability (p) must equal. " \
                                               f"y: {y.value.shape}, p: {p.value.shape}"
        self.batch_size = y.value.shape[0] 
    
    def forward(self): 
        self.sanity_check()
        y = self.inbound_nodes[0].value
        p = self.inbound_nodes[1].value
        self.value = -np.sum(np.multiply(y, np.log(p))) / self.batch_size

    def backward(self):
        """ 
        Stochastic gradient descent: 
        sets the gradient as a (m x k) matrix of gradient vector for each sample 
        where m is batch size and k is the number of classes
        """
        y = self.inbound_nodes[0].value
        p = self.inbound_nodes[1].value
        # Partial of the loss with respect to the label y
        self.gradients[self.inbound_nodes[0]] = -np.log(p)
        # Partial of the loss with respect to the prediction p
        self.gradients[self.inbound_nodes[1]] = -np.divide(y,p)


def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    Arguments:
        `feed_dict`: A dictionary where the key is a `Input` Node 
                    and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]
    # An adjacency table for all the nodes in the network
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        # Record n's out-going edges
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    # input_nodes have no incoming edges
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        # Initialize the Input node's value
        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward(graph):
    # Forward pass
    for n in graph:
        n.forward()

def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:
        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()
    # Backprop
    for n in graph[::-1]:
        n.backward()


def sgd_update_with_momentum(trainables, learning_rate=1e-2, momentum=0.5):
    """
    Updates the value of each trainable with Stochastic Gradient Descent.

    Arguments:
        `trainables`    : A list of `Input` Nodes representing weights/biases.
        `learning_rate` : The learning rate.
        `momentum`      : Ratio that 
    """
    for t in trainables:
        # NOTE: accumulative gradient over all batched samples
        # Can be compensated by a smaller learning rate
        partial = t.gradients[t]
        # Momentum Descent
        update = learning_rate * partial + (momentum * t.prev_update)
        t.value -= update
        t.prev_update = update