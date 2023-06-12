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
import time

parser = OptionParser()
parser.add_option("-o", "--optimizer", dest="optz", help="Optimizer select: gd, fr, nm", default="gd")
parser.add_option("-m", "--maxEpoch", dest="epochs", help="Number of maximum epoch",type="int", default=1000)
parser.add_option("-t", "--testTimes", dest="testTimes",type="int", default=100)
(options, args) = parser.parse_args()

logdir = r'/root/notebooks/tensorflow/logs'

if __name__ == '__main__':
    # The 4 training examples by columns
    X = np.array([[0, 0, 1, 1], 
                [0, 1, 0, 1]])

    # The outputs of the XOR for every example in X
    Y = np.array([[0, 1, 1, 0]])

    loss_fn = loss_mse
    loss_threshold = 1e-2

    out_dir = os.path.join(logdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    writer = SummaryWriter(out_dir)
    correct_record = [0, 0, 0, 0, 0]
    conv_times = 0
    conv_record = []
    # iter_t = 0

    for c_t in range(options.testTimes):
        iter_record = True
        model = MyModel(X.shape[0], 2, Y.shape[0])

        if options.optz == "gd":
            optz_name = "GD"
            optz = GD(model, loss_fn = loss_fn, lr=0.1)
        elif options.optz == "fr":
            optz_name = "Fletcher Reeves"
            optz = FR(model, loss_fn = loss_fn)
        elif options.optz == "nm":
            optz_name = "Nelder Mead"
            optz = Nelder_Mead(model, loss_fn)
        else:
            print("Error optimizer type")
            exit()

        for epoch in range(options.epochs):
            loss = optz.one_epoch(X, Y)
            writer.add_scalar(f'{optz_name}_Loss/{c_t}', loss, epoch + 1)

            if iter_record and loss <= loss_threshold:
                iter_record = False
                conv_times +=1
                conv_record.append(epoch+1)
                writer.add_scalar(f'convergence/iter', epoch + 1, c_t)
            print(f"\r{c_t}/{options.testTimes} Epoch {epoch + 1}/{options.epochs} Loss: {loss}", end="")
        if iter_record:
            writer.add_scalar(f'convergence/iter', options.epochs + 1, c_t)
        print("")

        predict = model(X).reshape(-1)
        count = np.where(Y.reshape(-1) == np.where(predict >= 0.5, 1, 0), 1, 0).sum()
        correct_record[count] += 1

    writer.add_scalar('convergence/times', options.testTimes - conv_times, 0)
    writer.add_scalar('convergence/times', conv_times, 1)
    
    for i in range(len(correct_record)):
        writer.add_scalar(f'{optz_name}/Correct', correct_record[i], i)
    
    print("--- Covergence ---")
    print(f"\tsuccess {conv_times}\n\tfailed {options.testTimes - conv_times}")
    if len(conv_record) > 0:
        print(f"\tmean {np.mean(conv_record)}\n\t std {np.std(conv_record)}")
    print("--- Accurancy ---")
    print(f'\t{correct_record}')