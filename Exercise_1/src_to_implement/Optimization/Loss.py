import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.label_tensor = None
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        loss = -np.sum(label_tensor * np.log(prediction_tensor + np.finfo(float).eps))
        return loss

    def backward(self, label_tensor):
        temp = -(1 / self.prediction_tensor) * label_tensor
        return temp