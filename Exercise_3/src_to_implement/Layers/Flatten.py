from .Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.trainable = False

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        batch_size = self.input_shape[0]
        tensor_flattened = input_tensor.reshape(batch_size, -1)
        return tensor_flattened

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)