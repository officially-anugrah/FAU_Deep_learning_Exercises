from copy import deepcopy
from Layers import *
from Optimization import *

class NeuralNetwork(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = Loss.CrossEntropyLoss() 
        self.input_tensor = None
        self.label_tensor = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        current_weights = self.input_tensor

        for layer in self.layers:
            current_weights = layer.forward(current_weights)
        predictions = current_weights

        return self.loss_layer.forward(predictions, self.label_tensor)

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)

        for i in range(1, len(self.layers) + 1):
            layer = self.layers[-i]
            error = layer.backward(error)

        return error

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor