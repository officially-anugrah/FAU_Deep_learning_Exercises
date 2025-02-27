import numpy as np
from Layers import Base, Initializers

class Dropout(Base.BaseLayer): 
    
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if self.testing_phase == False: 
            tensor = np.random.random(input_tensor.shape)
            tensor[tensor > (1 - self.probability)] = 1
            tensor[tensor < (1 - self.probability)] = 0
            self.tensor = tensor # Keep probabilities for the backward tensor
            return tensor*input_tensor/self.probability
        else:
            return input_tensor

    def backward(self, error_tensor):
        if self.testing_phase == False:
            return error_tensor * self.tensor / self.probability
        else:
            return error_tensor
