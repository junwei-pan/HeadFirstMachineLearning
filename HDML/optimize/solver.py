import numpy as np
import random as rd
from HDML.util.functions import sigmoid
import time

def gradient(w, lst_data, lst_label):
    g = np.zeros(len(w))
    for index, data in enumerate(lst_data):
        mu = sigmoid(w, data)
        label = lst_label[index]
        g += data * (mu - label) 
    return g

def BGD(w, lst_data, lst_label, n_iter = 50, reg = 'l2', eta = 0.001, beta = 0.5, time_invertal = 0.0001, debug = False):
    lst_w = []
    lst_w_t = []
    time_begin = time.time()
    interval_current = time_invertal
    for n_iter in range(n_iter):
        g = gradient(w, lst_data, lst_label)
        if reg == 'l2':
            # L2 Regularization
            g += np.array(w) * beta
        else:
            pass
        w = w - eta * g
        time_elasped = time.time() - time_begin
        if time_elasped >= interval_current:
            lst_w_t.append(w)
            interval_current += time_invertal
        lst_w.append(w)
        if debug == True:
            print 'BGD solver w', w
    return w, lst_w, lst_w_t

def SGD(w, lst_data, lst_label, n_iter = 50, reg = 'l2', eta = 0.0001, beta = 0.5, time_invertal = 0.0001, debug = False):
    # On each iteration, we chose D samples from the whole dataset with replacement, where D is the number of samples.
    lst_w = []
    lst_w_t = []
    time_begin = time.time()
    interval_current = time_invertal
    for n_iter in range(n_iter):
        for j in range(len(lst_data)):
            #index = rd.randint(0, len(lst_data) - 1)
            index = j
            lst_sample = lst_data[index]
            g = gradient(w, [lst_sample], [lst_label[index]])
            if reg == 'l2':
                # L2 Regularization
                g += np.array(w) * beta
            else:
                pass
            w = w - eta * g
            if j % 1000 == 0:
                time_elasped = time.time() - time_begin
                if time_elasped >= interval_current:
                    lst_w_t.append(w)
                    interval_current += time_invertal
        lst_w.append(w)
        if debug == True:
            print 'SGD solver w', w
    return w, lst_w, lst_w_t
