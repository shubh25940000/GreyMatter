import pandas as pd
from tqdm import tqdm as tq
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pickle

class ActivationFunctions:
    def __init__(self):
        pass

    def RELU(self, x):
        """
        Rectified linear unit
        :param x: Takes a single value. Not vectorized
        :return:
        """
        return np.maximum(0, x)

    def RELUDeriv(self, x):
        if x <= 0:
            return 0.0
        else:
            return 1.0

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
                return np.vectorize(self.RELUDeriv)
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

    def activationInput(self, X, W, B=None):
        if B:
            A = np.dot(X, W) + B
        else:
            A = np.dot(X, W)
        return A

    def activationOutput(self, activation, activationInput, derivative=None):
        return self.activationFunction(activation, derivative=derivative)(activationInput)

    def RSS(self, calculated, actual):
        l = (calculated - actual) ** 2
        return l

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
            if predicted[i] == actual[i]:
                c += 1
        return str(round(((c / actual.shape[0]) * 100), 2)) + "%"

    def predict(self, X, Weights):
        OutPutConsolidated = []
        for inp in X:
            ####Forward pass
            ActivationInput = []
            ActivationOutput = []
            X = np.atleast_2d(inp)
            n_input_features = X.shape[1]  # 784
            p = 0
            for j in Model.layers[1:-1]:
                W = Weights[p]
                hiddeninput_ = self.activationInput(X, W)
                ActivationInput.append(hiddeninput_)
                k = self.activationOutput(j[0], hiddeninput_)
                ActivationOutput.append(k)
                X = k  # 60000X700
                p = p + 1
            #                     if kk == 0:
            #                         print(X)
            W = Weights[p]
            hiddeninput_ = self.activationInput(X, W)
            ActivationInput.append(hiddeninput_)
            outputCalculated = self.activationOutput(output[0], hiddeninput_)
            #             outputCalculated = outputCalculated/np.max(outputCalculated)
            ActivationOutput.append(outputCalculated)
            OutPutConsolidated.append(np.argmax(outputCalculated))
        finalOutput = np.vstack(OutPutConsolidated)
        return finalOutput


class Layer(DeepLearning):
    def __init__(self, type_=None, n_neurons=None, activation=None, input_=None, bias=None):
        super().__init__()
        self.type_ = type_
        self.n_neurons = n_neurons
        self.activation = activation
        self.input_ = input_
        self.bias = bias

    def encode(self, x):
        edit = np.zeros(10)
        edit[x] = 1
        return edit

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
    def __init__(self, layers, learningRate, epochs, actual, X_test = None, y_test = None):
        super().__init__()
        self.layers = layers
        self.epochs = epochs
        self.actual = actual
        self.learningRate = learningRate
        self.X_test = X_test
        self.y_test = y_test

    def plotErrors(self, figure, plot, epoch, error):
        plot = plot
        new_y = error
        plot.set_xdata(range(0, epoch + 1))
        plot.set_ydata(new_y)
        plt.plot(range(0, epoch + 1), error)
        # figure.canvas.draw()
        plt.pause(0.5)
        figure.canvas.flush_events()

    def train(self, EvaluateOnTest = None):
        np.random.seed(seed=42)
        output = self.layers[-1]
        Weights = []
        Biases = []
        ####Initiate graph:
        graphx = range(1, 0 + 1)
        graphy = range(0)
        plt.ion()
        # here we are creating sub plots
        figure, ax = plt.subplots(figsize=(6, 4))
        line1, = ax.plot(graphx, graphy)
        plt.title("Error vs Epoch", fontsize=20)
        # setting x-axis label and y-axis label
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        predictedOut = None
        input_layer = self.layers[0]
        n_input_features = input_layer.shape[1]
        for j in self.layers[1:-1]:
            ####Initializing random weights and bias
            W = self.vectorNormalize(np.random.randn(n_input_features, j[1]))
            Weights.append(W)
            n_input_features = W.shape[1]
        W = self.vectorNormalize(np.random.randn(n_input_features, output[1]))
        Weights.append(W)
        print("Weights are initilized")
        E = []
        for i in range(self.epochs):
            print("Epoch = %d" % (i + 1))
            kk = 0
            OutPutConsolidated = []
            input_layer = self.layers[0]
            for inp in tq(input_layer):
                ####Forward pass
                ActivationInput = []
                ActivationOutput = []
                X = np.atleast_2d(inp)
                n_input_features = X.shape[1]  # 784
                p = 0
                for j in self.layers[1:-1]:
                    W = Weights[p]
                    hiddeninput_ = self.activationInput(X, W)
                    ActivationInput.append(hiddeninput_)
                    k = self.activationOutput(j[0], hiddeninput_)
                    ActivationOutput.append(k)
                    X = k  # 60000X700
                    p = p + 1

                W = Weights[p]
                hiddeninput_ = self.activationInput(X, W)
                ActivationInput.append(hiddeninput_)
                outputCalculated = self.activationOutput(output[0], hiddeninput_)
                ActivationOutput.append(outputCalculated)

                ####BackProp
                n = -1
                X = np.atleast_2d(inp)
                layer = len(self.layers[:0:-1]) - 1
                C_ = (outputCalculated - self.actual[kk])  #### 60000X10

                ####Initialize D:
                D = []
                D.append(C_ * self.activationOutput(activation=self.layers[n][0],
                                                    activationInput=outputCalculated, derivative=True))

                Weights[n] = Weights[n] - (self.learningRate * ActivationOutput[n - 1].T.dot(D[0]))  ####100,10
                x = 0
                for j in self.layers[-2:0:-1]:
                    delta = D[-1].dot(Weights[n].T)
                    delta = delta * self.activationOutput(activation=j[0],
                                                          activationInput=ActivationOutput[n - 1],
                                                          derivative=True)
                    D.append(delta)

                    if layer == 1:
                        Weights[n - 1] = Weights[n - 1] - (self.learningRate * X.T.dot(D[x + 1]))
                    else:
                        Weights[n - 1] = Weights[n - 1] - (self.learningRate * ActivationOutput[n - 2].T.dot(D[x + 1]))

                    n = n - 1
                    layer = layer - 1
                    x = x + 1
                OutPutConsolidated.append(outputCalculated)
                kk = kk + 1
            finalOutput = np.vstack(OutPutConsolidated)
            print("Loss = " + str(np.sum(self.RSS(finalOutput, self.actual))))
            E.append(np.sum((finalOutput - self.actual) ** 2))
            self.plotErrors(figure, line1, i, E)

            print("Accuracy on train dataset = " + str(self.getAccuracy(self.decodeOutput(finalOutput), self.decodeOutput(self.actual))))
            if EvaluateOnTest == True:
                print("Accuracy on test dataset = " + str(
                    self.getAccuracy(self.predict(self.X_test, Weights), self.decodeOutput(self.y_test))))
        return Weights



if __name__ == "__main__":

    l = Layer()
    def preprocessing(input):
        input["label"] = input["label"].apply(lambda x: l.encode(x))
        X = input.drop("label", axis=1)
        y = input["label"]
        X = X.to_numpy()
        y = y.to_numpy()
        y = np.vstack(y)
        X = X / np.max(X)
        return X, y
    train = pd.read_csv("/Users/shubhamchoudhury/Documents/Research/DeepLearning/FashionMNIST/fashion-mnist_train.csv")
    test = pd.read_csv("/Users/shubhamchoudhury/Documents/Research/DeepLearning/FashionMNIST/fashion-mnist_test.csv")

    X, y = preprocessing(train)
    X_test, y_test = preprocessing(test)
    inputLayer = Layer(type_="input", input_=X).output()
    hidden1 = Layer(type_="hidden", n_neurons=300, activation="Sigmoid").output()
    hidden2 = Layer(type_="hidden", n_neurons=100, activation="Sigmoid").output()
    output = Layer(type_="output", n_neurons=10, activation="Sigmoid").output()

    layers = [inputLayer, hidden1, hidden2, output]
    Model = Model(layers=layers, learningRate=0.01, epochs=30, actual=y, X_test=X_test, y_test=y_test)
    Weights = Model.train(EvaluateOnTest=True)

    with open("Models\FashionMNIST\Weights.pkl", "wb") as f:
        pickle.dump(Weights, f)

    with open("Models\FashionMNIST\Model.pkl", "wb") as f:
        pickle.dump(Model, f)
