import math
import numpy as np
import bigfloat
import warnings

def sigmoid(W, X):
    #print 'W', W
    #print 'X', X
    warnings.filterwarnings('always')
    try:
        res = 1 / (1 + np.exp(-1 * np.dot(W, X)))
    except Warning:
        print "W", W
        print "X", X
    return res

def sigmoid_boundary(x, W):
    return -1 * (W[0] * x + W[2]) / W[1]

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def log_likelihood_grad(X, Y, w, C=0.1):
    g = np.zeros(len(w))
    for index, data in enumerate(X):
        #mu = sigmoid(w, data)
        mu = logistic(np.dot(data, w))
        label = Y[index]
        g += data * (label - mu)
    g -= C * w
    return g

def log_likelihood(X, Y, w, C=0.1):
    likelihood = 0.0
    for index, data in enumerate(X):
        mu = logistic(np.dot(data, w))
        label = Y[index]
        likelihood += np.log(mu ** label * (1 - mu) ** (1 - label))
    return likelihood - C / 2 * np.dot(w, w)
    #return np.sum(np.log(logistic(Y * np.dot(X, w)))) - C/2 * np.dot(w, w)

def mean(lst):
    return sum(lst) * 1.0 / len(lst)
