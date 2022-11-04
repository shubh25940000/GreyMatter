import pandas as pd
import numpy as np
import math


class DeepLearning:
    def __init__(self):
        pass

    def RELU(self, x):
        if x <= 0:
            return 0
        else:
            return x

    def sigmoid(self, x):
        output = 1 / (1 + (math.e) ** x)
        return output

    def activationFunction(self, activation):
        if str(activation).lower() == "relu":
            return np.vectorize(self.RELU)
        elif str(activation).lower() == "sigmoid":
            return np.vectorize(self.sigmoid)

    def activationInput(self, X, W, B):
        A = np.dot(W, X) + B
        return A

    def costFunction(self, calculated, actual):
        return (calculated - actual) ^ 2


class Layer(DeepLearning):
    def __init__(self, type_=None, shape=None, activation=None, input_=None, bias = None):
        super().__init__()
        self.type_ = type_
        self.shape = shape
        self.activation = activation
        self.input_ = input_
        self.bias = bias

    def output(self):
        t = self.type_
        if str(t).lower() == "input":
            return np.array(self.input_)
        elif str(t).lower() == "hidden":
            if self.bias == None:
                return np.random.rand(self.shape), self.activation
            else:
                return np.random.rand(self.shape), self.activation, self.bias
        elif str(t).lower() == "output":
            return self.activation


class Train(Layer):
    def __init__(self,  layers, epochs, type_=None, shape=None, activation=None, input_=None, bias = None):
        Layer().__init__(type_=None, shape=None, activation=None, input_=None, bias = None)
        self.layers = layers
        self.epochs = epochs

    def train(self):
        pass



if __name__ == "__main__":
    inputLayer = Layer(type_ = "input", shape = [60000, 784]).output()
    hidden1 = Layer(type_ = "hidden", shape = [784, 1], activation = "RELU").output()
    hidden2 = Layer(type_="hidden", shape=[784, 1], activation = "RELU").output()
    output = Layer(type_ = "output", activation = "RELU").output()

    layers = [inputLayer, hidden1, hidden2, output]
    outputPred = Train(layers, epochs=10).train()
