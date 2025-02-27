import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.softmax_output = None

    def forward(self, input_tensor):
        exp_tensor = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.softmax_output = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)
        return self.softmax_output
    
    def backward(self, error_tensor):
        gradient_logits = self.softmax_output * error_tensor
        sum_gradients = gradient_logits.sum(axis=1, keepdims=True)
        gradient_logits -= self.softmax_output * sum_gradients
        return gradient_logits