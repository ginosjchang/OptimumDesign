import numpy as np
import math
from utils.Module import Module

# Sigmoid Function
def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

# Relu Function
def relu(Z):
    return np.where(Z>0, Z, 0)

# Leaky Relu Function
def leaky_relu(Z):
    return Z * np.where(Z>0, 1, 0.1)

# Derivative of the Sigmoid function
def d_sigmoid(Z):
    return sigmoid(Z) * (1-sigmoid(Z))

# Derivative of the Relu Function
def d_relu(Z):
    return np.where(Z>0, 1, 0)

# Derivative of the Leaky Relu Function
def d_leaky_relu(Z):
    return np.where(Z>0, 1, 0.1)

class Sigmoid(Module):
    def __init__(self):
        Module.__init__(self)
    
    def forward(self, x):
        return sigmoid(x)
    
    def backward(self, x):
        x = np.multiply(x, d_sigmoid(self.input))
        return x

class LeakyRelu(Module):
    def __init__(self):
        Module.__init__(self)
    
    def forward(self, x):
        return leaky_relu(x)
    
    def backward(self, x):
        return np.multiply(x, d_leaky_relu(self.input))