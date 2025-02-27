import numpy as np

class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights

class SgdWithMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient_tensor
        return weight_tensor + self.velocity

class Adam:
    def __init__(self, learning_rate=0.001, mu=0.9, rho=0.999):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.m = None
        self.v = None
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
            self.v = np.zeros_like(weight_tensor)
        self.t += 1
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)
        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)
        return weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)