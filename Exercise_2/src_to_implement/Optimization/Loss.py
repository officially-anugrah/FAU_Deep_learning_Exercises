import numpy as np


# Cross-entropy loss: Loss function used in classification problems, when outputs of neural networks are probability
# distributions


class CrossEntropyLoss:
    def __init__(self):
        self.label_tensor = None
        self.prediction_tensor = None  # store predictions made by network

    def forward(self, prediction_tensor, label_tensor):
        # prediction_tensor: probabilities predicted by the model for each class (typically output of softmax layer)
        # labels/ ground truth labels: possibly one-hot encoded
        # the smallest representable positive number, used to avoid taking log of zero
        # compute cross-entropy loss
        # take negative log of all predicted probabilities adjusted by epsilon and multiplying corresponding albels
        # the operation is element-wise, and total loss is computed by summing over all elements
        self.prediction_tensor = prediction_tensor
        loss = -np.sum(label_tensor * np.log(prediction_tensor + np.finfo(float).eps))
        return loss

    def backward(self, label_tensor):
        # used in backpropagation
        # return gradient of the loss wrt predictions
        # def backward(self, label_tensor):
        tmp = -(1 / self.prediction_tensor) * label_tensor
        return tmp
