import pandas as pd
import tqdm
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

    def activationOutput(self, activation, activationInput):
        return self.activationFunction(activation)(activationInput)

    def loss(self, calculated, actual):
        return (calculated - actual) ^ 2

    def optimization(self, optimizer):
        if str(optimizer).lower() == "sgd":
            pass


class Layer(DeepLearning):
    def __init__(self, type_=None, n_neurons = None, activation=None, input_=None, bias = None):
        super().__init__()
        self.type_ = type_
        self.n_neurons = n_neurons
        self.activation = activation
        self.input_ = input_
        self.bias = bias

    def output(self):
        t = self.type_
        if str(t).lower() == "input":
            return np.array(self.input_)
        elif str(t).lower() == "hidden":
            if self.bias == None:
                return self.activation, self.n_neurons
            else:
                return self.activation, self.n_neurons, self.bias
        elif str(t).lower() == "output":
            return self.activation, self.n_neurons

class Train(DeepLearning):
    def __init__(self,  layers, epochs):
        super().__init__()
        self.layers = layers
        self.epochs = epochs

    def train(self):
        X = self.layers[0]
        np.random.seed(seed=42)
        output = self.layers[-1]
        for i in range(self.epochs):
            if i == 0:
                ####Forward pass
                n_input_features = X[1] #784
                n_input_rows = X[0] #60000
                for j in self.layers[1:-1]:
                    ####Initializing random weights and bias
                    W = np.random.rand(n_input_features, j[1])
                    B = np.random.rand(n_input_rows, 1)
                    hiddeninput_ = self.activationInput(X, W, B)
                    k = self.activationOutput(j[0], hiddeninput_)
                    X = k #60000X700
                    n_input_features = X[1]
                    n_input_rows = X[0]
                W = np.random.rand(n_input_features, output[1])
                B = np.random.rand(n_input_rows, 1)
                hiddeninput_ = self.activationInput(X, W, B)
                outputCalculated =  self.activationOutput(output[0], hiddeninput_)
            else:
                pass


if __name__ == "__main__":

    train = pd.read_csv("DeepLearning/MNIST/mnist_train.csv")
    X = train.drop("label", axis=1)
    y = train["label"]
    X = X.to_numpy()
    y = y.to_numpy()
    X = X / np.max(X)
    inputLayer = Layer(type_ = "input", input_ = X).output()
    hidden1 = Layer(type_ = "hidden", n_neurons=700, activation = "RELU").output()
    hidden2 = Layer(type_="hidden", n_neurons=700, activation = "RELU").output()
    output = Layer(type_ = "output", n_neurons=10, activation = "Sigmoid").output()

    layers = [inputLayer, hidden1, hidden2, output]
    outputPred = Train(layers, epochs=10).train()
