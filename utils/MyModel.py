from utils.Module import Module
import numpy as np
from utils.ActivateFunction import *
from utils.Loss import *

class Dense(Module):
    def __init__(self, in_len, out_len):
        Module.__init__(self)

        self.in_len = in_len
        self.out_len = out_len
        self.n = out_len * in_len + out_len
        
        self.reset()

    def __len__(self):
        return self.n

    def reset(self):
        self.w = np.random.randn(self.out_len, self.in_len)
        self.b = np.zeros((self.out_len, 1))
    
    def forward(self, x):
        return np.dot(self.w, x) + self.b
    
    def backward(self, x):
        self.dw = np.dot(x, self.input.T)
        self.db = np.sum(x, axis = 1, keepdims = True)
        return np.dot(self.w.T, x)
    
    def grad(self):
        return np.concatenate((self.dw.reshape(-1), self.db.reshape(-1)))
    
    def parameters(self):
        return np.concatenate((self.w.reshape(-1), self.b.reshape(-1)))

    def set_param(self, x):
        self.b = x[-self.out_len:].reshape((-1, 1))
        self.w = x[:-self.out_len].reshape(self.out_len, self.in_len)

class MyModel(Module):
    def __init__(self, n_x, n_h, n_y):
        Module.__init__(self)

        self.layers = []
        self.layers.append(Dense(n_x, n_h))
        self.layers.append(LeakyRelu())
        self.layers.append(Dense(n_h, n_y))
        self.layers.append(Sigmoid())
    
    def __len__(self):
        n = 0
        for layer in self.layers:
            n += len(layer)
        return n
    
    def __str__(self):
        s = ""
        for layer in self.layers:
            s += layer.__str__()
        return s

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, Gt):
        x = d_loss_ce(self.output, Gt)
        for layer in reversed(self.layers):
            x = layer.backward(x)
    
    def grad(self):
        g = None
        for layer in self.layers:
            if layer.grad() is None: continue
            if g is None: g = layer.grad()
            else: g = np.concatenate((g, layer.grad()))
        return g

    def parameters(self):
        p = None
        for layer in self.layers:
            if layer.parameters() is None: continue
            if p is None: p = layer.parameters()
            else: p = np.concatenate((p, layer.parameters()))
        return p

    def reset(self):
        for layer in self.layers:
            layer.reset()
        return self

    def set_param(self, x):
        for layer in self.layers:
            layer.set_param(x[:len(layer)])
            x = x[len(layer):]
        return self

    def forward_param(self, x, param):
        orig = self.parameters()
        self.set_param(param)
        x = self.forward(x)
        self.set_param(orig)
        return x
