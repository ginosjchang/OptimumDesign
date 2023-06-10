import numpy as np
import math
# Loss function: (MSE)
    # input: Model output: A2 & groundtruth: Y
    # output: Loss value

def loss_mse(A2, Y):
    loss = (A2 - Y)**2
    loss = loss.mean()

    return loss

# Derivative of the Loss function
def d_loss_mse(A2, Y):
    dsq_diffs = (2 * (A2 - Y))/m
    
    return dsq_diffs

def loss_ce(A2, Y):
    loss = np.where(Y > 0.5, np.log(A2), np.log(1-A2))
    return -1*loss.mean()

def d_loss_ce(A2, Y):
    d = np.where(Y > 0.5, -1/A2, 1/(1-A2))
    return d