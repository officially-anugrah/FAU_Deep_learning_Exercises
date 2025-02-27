import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        grad_input = (self.input_tensor > 0).astype(float)
        return grad_input * error_tensor
