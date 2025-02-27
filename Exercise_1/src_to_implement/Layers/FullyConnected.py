import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (output_size, input_size + 1))
        self.input_tensor = None
        self.gradient_weights = None
        self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        bias_term = np.ones((input_tensor.shape[0], 1))
        augmented_input = np.hstack([input_tensor, bias_term])

        return np.dot(augmented_input, self.weights.T)

    def backward(self, error_tensor):
        bias_term = np.ones((self.input_tensor.shape[0], 1))
        augmented_input = np.hstack([self.input_tensor, bias_term])
        self.gradient_weights = np.dot(error_tensor.T, augmented_input)
        
        if self._optimizer:
            sgd = self.optimizer
            self.weights = sgd.calculate_update(self.weights, self.gradient_weights)

        return np.dot(error_tensor, self.weights[:,:-1])

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value