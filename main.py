import numpy as np
from matplotlib import pyplot as plt
import math
from utils.Loss import loss_mse, loss_ce, d_loss_ce
from utils.MyModel import MyModel
from utils.Optimizers import *
from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

parser = OptionParser()
parser.add_option("-o", "--optimizer", dest="optz", help="Optimizer select: gd, fr, nm", default="gd")
parser.add_option("-m", "--maxEpoch", dest="epochs", help="Number of maximum epoch", default=1000)
(options, args) = parser.parse_args()

logdir = r'/root/notebooks/tensorflow/logs'

# 使用 powell's conjugate direction method 收斂參數
def powell_conjugate(X, Y, loss_fn=loss_mse):
    print(f"--- Powell Conjugate Optimizer ---")
    model = MyModel(X.shape[0], 2, Y.shape[0])

    def forward_propagation_loss(param):
        model.set_param(param)
        return loss_fn(model(X), Y)

    def find_min(x1, v, h=0.1, cc=-1):
        a,b = bracket(x1,v,h=h)
        if np.sum(a) == 0.:
            return [False,None]
        opt = Gold_sec_nD(np.array([a,b]),cc=cc)
        return [True,opt]

    def bracket(x1, v, h=0.1):#from the book: Numerical methods in Engineering with Python
        c = 1.618033989
        f1 = forward_propagation_loss(x1)
        x2 = x1 + v*h
        f2 = forward_propagation_loss(x2)
        # Determine downhill direction and change sign of h if needed
        if f2 > f1:
            h = -h
            x2 = x1 + v*h
            f2 = forward_propagation_loss(x2)
            # Check if minimum between x1 - h and x1 + h
            if f2 > f1: 
                return x2, x1- v*h
        # Search loop
        for i in range (100):
            h = c* h
            x3 = x2+ v*h
            f3 = forward_propagation_loss(x3)
            if f3 > f2:
                return x1, x3
            x1 = x2
            x2 = x3
            f1 = f2
            f2 = f3
        print("The bracket did not include a minimum")
        return 0., 0.

    def Gold_sec_nD(intv: np.ndarray, cc=-1):
        '''
        use golden select method to find local minimum or maximum
        '''
        if cc == -1: # converge condition
            cc = 1.0e-4
        alf = (5**0.5 -1)/ 2 # golden ratio = 0.618... 
        a = intv[0]
        b = intv[1]
        lam= a+ (1- alf)* (b- a)  # |--------|-----|--------|
        mu = a+ alf* (b- a)       # a       lam   mu        b
        fl = forward_propagation_loss(lam)
        fm = forward_propagation_loss(mu) #At first, 2 function evaluations are needed
        iter = 0
        while float(math.dist(a,b)) > cc and iter < 500:
            iter +=1
            # n_iter = n_iter+1      
            if fl > fm:           # |--------|-----|--------|
                a = lam           # x     a(lam)  lam(mu)   b
                lam = mu
                fl = fm
                mu = a+ alf* (b- a)
                fm = forward_propagation_loss(mu)   #In the while loop, only 1 more function evalution is needed
                # optv.append(fl)
            else:
                b = mu
                mu = lam
                fm = fl
                lam = a+ (1- alf)* (b- a)
                fl = forward_propagation_loss(lam)
                # optv.append(fm)
        if forward_propagation_loss(a) < forward_propagation_loss(b): #compare f(a) with f(b), and xopt is the one with smaller func value
            xopt = a
        else:
            xopt = b
            
        return xopt

    param = model.parameters()
    f1 = forward_propagation_loss(param)
    d = np.identity(len(param))
    niter = 0
    while niter < 100:
        z = [param]
        y = [param]
        x = param
        for j in range(len(param)):
            for i in range(len(param)):
                _, param = find_min(param, d[i])
                z.append(param)
            _, param = find_min(param, z[-1]-z[0])
            y.append(param)
            if j == len(param) -1:
                continue
            np.delete(d,0)
            np.concatenate((d,np.array([z[-1]-z[0]])))
            z = [y[j+1]]
        if math.dist(y[-1],x) < 1e-4 and forward_propagation_loss(param) < 1e-2:
            break
        else:
            param = y[-1]
        niter += 1
        print(f"\rIteration {niter} Loss: {forward_propagation_loss(param)}", end="")
    print("")
    return niter, model.set_param(x)

if __name__ == '__main__':
    # The 4 training examples by columns
    X = np.array([[0, 0, 1, 1], 
                [0, 1, 0, 1]])

    # The outputs of the XOR for every example in X
    Y = np.array([[0, 1, 1, 0]])

    loss_fn = loss_mse
    model = MyModel(X.shape[0], 2, Y.shape[0])

    if options.optz == "gd":
        optz = GD(model, loss_fn = loss_fn, lr=0.1)
    elif options.optz == "fr":
        optz = FR(model, loss_fn = loss_fn)
    elif options.optz == "nm":
        optz = Nelder_Mead(model, loss_fn)
    else:
        print("Error optimizer type")
        exit()

    out_dir = os.path.join(logdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    writer = SummaryWriter(out_dir)

    for epoch in range(options.epochs):
        loss = optz.one_epoch(X, Y)
        writer.add_scalar(f'{options.optz}/Loss', loss, epoch + 1)
        print(f"\rEpoch {epoch + 1}/{options.epochs} Loss: {loss}", end="")
    print("")