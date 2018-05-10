import unittest
import neuron, datapipe
import numpy as np
import annealing

# class TestNodes(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.inputNode = neuron.Input()
#         self.inputNode.value = np.array([[1, 2, 3],[4, 5, 6]])
#         self.labelNode = neuron.Input()
#         self.labelNode.value = np.array([[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])
#         self.d1 = 4
#         self.h1 = 5
#         self.d2 = 5
#         self.h2 = 3
#         self.inputSample = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]])
#         self.inputLabel  = np.array([[0.01, 0.01, 0.98], [0.01, 0.01, 0.98]])
#         self.w1 = 0.001 * np.random.randn(self.d1, self.h1)
#         self.b1  = 0.00001 + np.zeros((1,self.h1))
#         self.w2 = 0.001 * np.random.randn(self.d2, self.h2)
#         self.b2  = 0.00001 + np.zeros((1,self.h2))
    
    # def testSoftmax(self):
    #     s = neuron.Softmax(self.inputNode)
    #     s.forward()
    #     # print(s.value)
    #     return s

    # def testCEL(self):
    #     s = self.testSoftmax()
    #     loss = neuron.CEL(self.labelNode, s)
    #     # print("s's value:", s.value)
    #     loss.forward()
    #     # print(f"Lost: {loss.value}")
    
    # def testCELandSoftmaxBackward(self):
    #     s = self.testSoftmax()
    #     loss = neuron.CEL(self.labelNode, s)
    #     loss.forward()
    #     loss.backward()
    #     s.backward()
    #     # print(f"Loss gradients: {loss.gradients}")
    #     # print(f"Softmax gradients: {s.gradients}")
    #     # print(f"Expected gradients: {s.value - self.labelNode.value} \n")
    #     return 0

    # def testForward(self):
    #     X, W, b, y = neuron.Input(), neuron.Input(), neuron.Input(), neuron.Input()
    #     X.value = self.inputSample
    #     W.value = 0.01 * np.random.randn(4, 3)
    #     b.value = 0.001 + np.zeros((1,3))
    #     y.value = self.inputLabel
    #     f = neuron.Linear(X, W, b)
    #     f.forward()
    #     # print("Linear forward: ", f.value)
    #     r = neuron.ReLU(f)
    #     r.forward()
    #     # print("ReLU forward: ", r.value)
    #     s = neuron.Softmax(r)
    #     s.forward()
    #     # print("Softmax forward: ", s.value)
    #     e = neuron.CEL(y, s)
    #     e.forward()
    #     # print("Error: ", e.value)

    # def testCompleteForward(self):
    #     # Setup the different layers in the NN 
    #     # Input layer
    #     X, W, b, y = neuron.Input(), neuron.Input(), neuron.Input(), neuron.Input()
    #     f = neuron.Linear(X, W, b)
    #     a = neuron.ReLU(f)
    #     s = neuron.Softmax(a)
    #     # Output layer
    #     e = neuron.CEL(y, s)
        
    #     w_ = 0.01 * np.random.randn(4, 3)
    #     b_ = 0.001 + np.zeros((1,3))

    #     # A dictionary that sets the value of all Input nodes in the NN
    #     feed_dict = {
    #         X: self.inputSample,
    #         y: self.inputLabel,
    #         W: w_,
    #         b: b_
    #     }

    #     graph = neuron.topological_sort(feed_dict)
    #     for n in graph:
    #         n.forward()
    #     # print("Complete Forward Error: ", graph[-1].value)

    #     return (graph, X, W, b, y)

    # def testCompleteBackward(self):
    #     result_tuple = self.testCompleteForward()
    #     graph = result_tuple[0]
    #     X     = result_tuple[1]
    #     W     = result_tuple[2]
    #     b     = result_tuple[3]
    #     y     = result_tuple[4]
    #     for n in graph[::-1]:
    #         n.backward()
    #     # print("gradients: ",W.gradients)

    # def testDoubleHiddenLayer(self):
    #     # Setup the different layers in the NN 
    #     # Input layer
    #     X, W1, W2, b1, b2, y = neuron.Input(), neuron.Input(), neuron.Input(), neuron.Input(),neuron.Input(),neuron.Input()
    #     f1 = neuron.Linear(X, W1, b1)
    #     a1 = neuron.ReLU(f1)
    #     f2 = neuron.Linear(a1, W2, b2)
    #     a2 = neuron.ReLU(f2)
    #     s = neuron.Softmax(a2)
    #     # Output layer
    #     e = neuron.CEL(y, s)

    #     # A dictionary that sets the value of all Input nodes in the NN
    #     feed_dict = {
    #         X: self.inputSample,
    #         y: self.inputLabel,
    #         W1: self.w1,
    #         W2: self.w2,
    #         b1: self.b1,
    #         b2: self.b2
    #     }

    #     graph = neuron.topological_sort(feed_dict)
    #     neuron.forward_and_backward(graph)
    #     print("\n *************************************** \n")
    #     print("W1 gradients: ",W1.gradients)
    #     print("b1 gradients: ",b1.gradients)
    #     print("W2 gradients: ",W2.gradients)
    #     print("b2 gradients: ",b2.gradients)
        # print("CEL gradients :", e.gradients)
        # print("Softmax gradients: ", s.gradients)
        # print(f"Expected gradients: {s.value - self.inputLabel} \n")

# class TestDataPipe(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.filename = 'iris-data.txt'
#         self.classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#         self.input_dimension = 4
#         self.data = datapipe.Data(self.classes, self.input_dimension)

#     def testRead(self):
#         self.data.read_n_process(self.filename)
#         for index in range(len(self.data.labels)):
#             self.assertEquals(self.data.encoding_to_class(self.data.label[index]), 
#                               self.data.d[index,4])

#     def testOneHot(self):
#         print("'Iris-setosa is :", self.data.one_hot_encode('Iris-setosa'))

class TestAnnealing(unittest.TestCase):
    def testAsWhole(self):
        a = annealing.Annealing(0.01, 2, 0.90,  0.0004)
        for i in range(20):
            a.decay()
            if a.counter == 0:
                print(a.value)

if __name__ == '__main__':
    unittest.main()
        