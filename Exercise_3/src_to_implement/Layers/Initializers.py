import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, shape, fan_in=None, fan_out=None):
        return np.full(shape, self.value)

class UniformRandom:
    @staticmethod
    def initialize(shape, fan_in=None, fan_out=None):
        return np.random.uniform(0, 1, size=shape)

class Xavier:
    @staticmethod
    def initialize(shape, fan_in, fan_out):
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, stddev, size=shape)

class He:
    @staticmethod
    def initialize(shape, fan_in, fan_out=None):
        stddev = np.sqrt(2 / fan_in)
        return np.random.normal(0, stddev, size=shape)