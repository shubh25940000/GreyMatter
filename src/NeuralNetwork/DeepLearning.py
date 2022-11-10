import pandas as pd
import tqdm
import numpy as np
import math
import sympy as sym
from sympy import *


class ActivationFunctions:
    def __init__(self):
        pass

    def RELU(self, x):
        """
        Rectified linear unit
        :param x: Takes a single value. Not vectorized
        :return:
        """
        return max(0, x)

    def sigmoid(self, x):
        """
        Sigmoid
        :param x: Takes a single value. Not vectorized
        :return:
        """
        if x < -709:
            x = -708
        return 1 / (1 + (math.e) ** (-x))

    def sigmoidDeriv(self, x):
        """
        Sigmoid
        :param x: Takes a single value. Not vectorized
        :return:
        """
        #         if x < -353:
        #             x = -353
        #         expr = -(math.e ** (-x))/(1 - math.e ** (-x)) ** 2
        expr = x * (1 - x)
        return expr

    def softmax(self, x):
        """
        :param x: Input is a numpy array
        :return:
        """
        return lambda x: np.exp(x) / np.sum(np.exp(x))

    def activationFunction(self, activation, derivative=None):
        if str(activation).lower() == "relu":
            if derivative == None:
                return np.vectorize(self.RELU)
            else:
                f = lambda x: 0 if x <= 0 else 1
                return np.vectorize(f)
        elif str(activation).lower() == "sigmoid":
            if derivative == None:
                return np.vectorize(self.sigmoid)
            else:
                return np.vectorize(self.sigmoidDeriv)
        elif str(activation).lower() == "softmax":
            if derivative == None:
                return self.softmax
            else:
                expr = lambda x: (np.exp(x) / np.sum(np.exp(x)) * (1 - np.exp(x) / np.sum(np.exp(x))))
                return expr


class DeepLearning(ActivationFunctions):
    def __init__(self):
        pass

    def vectorNormalize(self, y):
        y = y / np.sqrt(y.shape[0])
        #             y[i] = (y[i] - np.min(y[i])) / (np.max(y[i]) - np.min(y[i]))
        return y

    def activationInput(self, X, W, B):
        A = np.dot(X, W) + B
        return A

    def activationOutput(self, activation, activationInput, derivative=None):
        return self.activationFunction(activation, derivative=derivative)(activationInput)

    def RSS(self, calculated, actual):
        l = (calculated - actual) ** 2
        return l

    def loss(self, X, Y):
        return np.vectorize(self.RSS)(X, Y)

    def gradientDescent(self, gradient, start, learn_rate, n_iter=50):
        v = start
        diff = - learn_rate * gradient(v)
        v += diff
        return v

    def optimization(self, optimizer):
        if str(optimizer).lower() == "sgd":
            pass

    def decodeOutput(self, x):
        shape = x.shape[0]
        return np.argmax(x, axis=1).reshape(shape, 1)

    def getPrecision(self):
        pass

    def getRecall(self):
        pass

    def getF1Score(self):
        Precision = self.getPrecision()
        Recall = self.getRecall
        F1 = 2 * Precision * Recall / (Precision + Recall)

    def getAccuracy(self, predicted, actual):
        i = 0
        c = 0
        for i in range(actual.shape[0]):
            if predicted[i][0] == actual[i][0]:
                c += 1
        return str(round(((c / actual.shape[0]) * 100), 2)) + "%"


class LossFunctions:
    def __init__(self):
        pass


class Layer(DeepLearning):
    def __init__(self, type_=None, n_neurons=None, activation=None, input_=None, bias=None):
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


class Model(DeepLearning, ActivationFunctions):
    def __init__(self, layers, learningRate, epochs, actual):
        super().__init__()
        self.layers = layers
        self.epochs = epochs
        self.actual = actual
        self.learningRate = learningRate

    def train(self):
        np.random.seed(seed=42)
        output = self.layers[-1]
        Weights = []
        Biases = []
        ActivationInput = []
        ActivationOutput = []
        for i in range(self.epochs):
            X = self.layers[0]
            if i == 0:
                ####Forward pass
                print("Forward pass in progress")
                n_input_features = X.shape[1]  # 784
                n_input_rows = X.shape[0]  # 60000
                for j in self.layers[1:-1]:
                    ####Initializing random weights and bias
                    W = self.vectorNormalize(np.random.randn(n_input_features, j[1]))
                    B = self.vectorNormalize(np.random.randn(n_input_rows, 1))
                    hiddeninput_ = self.activationInput(X, W, B)
                    ActivationInput.append(hiddeninput_)
                    k = self.activationOutput(j[0], hiddeninput_)
                    ActivationOutput.append(k)
                    X = k  # 60000X700
                    n_input_features = X.shape[1]
                    n_input_rows = X.shape[0]
                    Weights.append(W)
                    Biases.append(B)
                W = self.vectorNormalize(np.random.randn(n_input_features, output[1]))
                B = self.vectorNormalize(np.random.randn(n_input_rows, 1))
                hiddeninput_ = self.activationInput(X, W, B)
                ActivationInput.append(hiddeninput_)
                Weights.append(W)
                Biases.append(B)
                outputCalculated = self.activationOutput(output[0], hiddeninput_)
                ActivationOutput.append(outputCalculated)
                print("Forward pass completed")


            else:
                ####BackProp
                print("Back Propogation in progress")
                print("Epoch = %d" % i)
                eR = []
                #                 outputCalculated = None
                n = -1
                print("Error = " + str(np.sum((outputCalculated - self.actual) ** 2)))
                #                     plt.plot(eR.append(np.sum((outputCalculated - self.actual) ** 2)))

                layer = len(self.layers[:0:-1]) - 1
                C_ = (outputCalculated - self.actual)  #### 60000X10
                ####Initialize D:
                D = []
                D.append(C_ * self.activationOutput(activation=self.layers[n][0],
                                                    activationInput=outputCalculated, derivative=True))

                Weights[n] = Weights[n] - (self.learningRate * ActivationOutput[n - 1].T.dot(D[0]))  ####100,10
                x = 0
                for j in self.layers[-2:0:-1]:
                    print("Updating Weights and Biases for layer: %d" % layer)
                    delta = D[-1].dot(Weights[n].T)
                    delta = delta * self.activationOutput(activation=j[0],
                                                          activationInput=ActivationOutput[n - 1],
                                                          derivative=True)
                    D.append(delta)

                    if layer == 1:
                        Weights[n - 1] = Weights[n - 1] - self.learningRate * self.layers[0].T.dot(D[x + 1])
                    else:
                        Weights[n - 1] = Weights[n - 1] - self.learningRate * ActivationOutput[n - 2].T.dot(D[x + 1])

                    #                         D.append(delta)
                    n = n - 1
                    layer = layer - 1
                    x = x + 1

                ActivationInput = []
                ActivationOutput = []
                numberL = 0
                X = self.layers[0]
                for j in self.layers[1:-1]:
                    W = Weights[numberL]
                    B = Biases[numberL]

                    hiddeninput_ = self.activationInput(X, W, B)
                    ActivationInput.append(hiddeninput_)
                    k = self.activationOutput(j[0], hiddeninput_)
                    ActivationOutput.append(k)
                    X = k  # 60000X700
                    n_input_features = X.shape[1]
                    n_input_rows = X.shape[0]
                    numberL = numberL + 1
                W = Weights[-1]
                B = Biases[-1]
                hiddeninput_ = self.activationInput(X, W, B)
                ActivationInput.append(hiddeninput_)
                outputCalculated = self.activationOutput(output[0], hiddeninput_)

                ActivationOutput.append(outputCalculated)

                X = self.layers[0]

                print("Accuracy = %s" % self.getAccuracy(self.decodeOutput(outputCalculated),
                                                         self.decodeOutput(self.actual)))
        return outputCalculated, Weights, Biases


if __name__ == "__main__":

    train = pd.read_csv("DeepLearning/MNIST/mnist_train.csv")
    X = train.drop("label", axis=1)
    y = train["label"]
    X = X.to_numpy()
    y = y.to_numpy()
    X = X / np.max(X)
    inputLayer = Layer(type_ = "input", input_ = X).output()
    hidden1 = Layer(type_ = "hidden", n_neurons=100, activation = "RELU").output()
    hidden2 = Layer(type_="hidden", n_neurons=100, activation = "RELU").output()
    output = Layer(type_ = "output", n_neurons=10, activation = "RELU").output()

    layers = [inputLayer, hidden1, hidden2, output]
    outputPred, Weights, Biases = Model(layers = layers, learningRate=0.08, epochs=5, n_iter=4, actual = y).train()
