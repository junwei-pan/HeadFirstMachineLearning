from HDML.util.functions import sigmoid, sigmoid_boundary, log_likelihood_grad, log_likelihood
from HDML.optimize.solver import BGD, SGD
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
class logistic_regression:
    def __init__(self, data, label, n_iter, eta, beta, time_invertal=0.001, debug = False):
        '''
        eta: float, learning rate
        beta: float, coefficient of regularization term
        '''
        self.n_iter = n_iter
        self.eta = eta
        self.beta = beta
        self.label = label
        self.time_invertal = time_invertal
        # Weight in different iterations
        self.lst_W = []
        self.lst_W_t = []
        self.n_feature = len(data[0])
        self.data = data
        self.data_original = data
        self.debug = debug
        # Inteception
        self.data = np.insert(self.data, self.n_feature , 1.0, axis = 1)
        self.W = np.random.rand(self.n_feature + 1)
    
    def nll(self, w):
        warnings.filterwarnings("always")
        self.likelihood = 0.0
        for index, sample in enumerate(self.data):
            mu = sigmoid(w, sample)
            label = self.label[index]
            try:
                self.likelihood -= np.log(mu ** label * (1 - mu) ** (1 - label)) 
            except Warning:
                print "NLL, w", w
        return self.likelihood +  self.beta / 2 * np.dot(w, w)

    def gradient(self, w, lst_data, lst_label):
        g = np.zeros(len(w))
        for index, data in enumerate(lst_data):
            mu = sigmoid(w, data)
            label = lst_label[index]
            g += data * (mu - label) 
        g += np.array(w) * self.beta
        return g

    def train_with_BGD(self):
        self.W, self.lst_W, self.lst_W_t = BGD(self.W, self.data, self.label, reg = 'l2', eta = self.eta, beta = self.beta, n_iter = self.n_iter, debug = self.debug)
        return self.W, self.lst_W
        '''
        for n_iter in range(self.n_iter):
            g = self.gradient(self.W, self.data, self.label)
            #self.W = [x - self.eta * y for x, y in zip(self.W, g)]
            self.W = self.W - self.eta * g
            self.lst_W.append(self.W)
        '''
    
    def train_with_SGD(self):
        try:
            self.W, self.lst_W, self.lst_W_t = SGD(self.W, self.data, self.label, reg = 'l2', eta = self.eta, beta = self.beta, n_iter = self.n_iter, debug = self.debug)
        except:
            print self.W
        return self.W, self.lst_W
        '''
        for n_iter in range(self.n_iter):
            for j in range(len(self.data)):
                index = rd.randint(0, len(self.data) - 1)
                data = self.data[index]
                g = self.gradient(self.W, [data], [self.label[index]])
                #self.W = [x - self.eta * y for x, y in zip(self.W, g)]
                self.W = self.W - self.eta * g
            self.lst_W.append(self.W)
        '''
    def coordinate_gradient_descent(self):
        for n_iter in range(self.n_iter):
            pass

    def bfgs(self):
        def f(w):
            return self.nll(w)
        
        def fprime(w):
            return self.gradient(w, self.data, self.label)

        self.W, lst_W = scipy.optimize.fmin_bfgs(f, self.W, fprime, disp=True, retall = True, maxiter = self.n_iter)
        for W in lst_W:
            print self.nll(W)
        return self.W
    
    def cg(self):
        def f(w):
            return self.nll(w)
        
        def fprime(w):
            return self.gradient(w, self.data, self.label)

        self.W, lst_W = scipy.optimize.fmin_cg(f, self.W, fprime, disp=True, retall = True, maxiter = self.n_iter)
        for W in lst_W:
            print self.nll(W)
        return self.W
    
    def tnc(self):
        def f(w):
            return self.nll(w)
        
        def fprime(w):
            return self.gradient(w, self.data, self.label)

        self.W, lst_W = scipy.optimize.fmin_tnc(f, self.W, fprime, disp=True)
        for W in lst_W:
            print self.nll(W)
        return self.W
    
    def powell(self):
        def f(w):
            return self.nll(w)
        
        self.W, lst_W = scipy.optimize.fmin_powell(f, self.W, retall = True, maxiter = self.n_iter)
        for W in lst_W:
            print self.nll(W)
        return self.W
    
    
    def test(self, test_data, label):
        n_pos = 0.0
        for i in range(len(test_data)):
            if sigmoid(self.W, test_data[i]) >= 0.5 and self.label[i] == 1:
                n_pos += 1
            elif sigmoid(self.W, test_data[i]) < 0.5 and self.label[i] == 0:
                n_pos += 1
        
        print n_pos, len(test_data), n_pos / len(test_data)
        print self.W


    def plot_data(self):
        lst_negative = [self.data_original[i] for i in range(len(self.data_original)) if self.label[i] != 1.0]
        lst_positive = [self.data_original[i] for i in range(len(self.data_original)) if self.label[i] == 1.0]
        lst_positive_x = [l[0] for l in lst_positive]
        lst_positive_y = [l[1] for l in lst_positive]
        lst_negative_x = [l[0] for l in lst_negative]
        lst_negative_y = [l[1] for l in lst_negative]
        plt.plot(lst_positive_x, lst_positive_y, 'r*')
        plt.plot(lst_negative_x, lst_negative_y, 'g*')
        #x_min = min([d[0] for d in self.data_original])
        #x_max = max([d[0] for d in self.data_original])
        #plt.plot([x_min, x_max], [sigmoid_boundary(x_min, self.W), sigmoid_boundary(x_max, self.W)], 'b-')
        plt.show()
    
    def plot_nll_vs_iteration(self):
        x = range(len(self.lst_W))  
        y = [self.nll(w) for w in self.lst_W]
        plt.clf()
        plt.plot(x, y, 'r-')
        plt.show()
    
    def plot_nll_vs_time(self, n_interval = 2000):
        MAX = len(self.lst_W_t)
        x = range(MAX)  
        y = [self.nll(w) for w in self.lst_W_t[:MAX]]
        plt.clf()
        plt.plot(x, y, 'r-')
        plt.show()
    
    def plot_decision_boundary(self):
        pass

    def set_initial_weight(self, W):
        self.W = np.append(W, [1.0])

def read_data(filename, sep=",", filt=int):

    def split_line(line):
        return line.split(sep)

    def apply_filt(values):
        return map(filt, values)

    def process_line(line):
        return apply_filt(split_line(line))

    f = open(filename)
    lines = map(process_line, f.readlines())
    # "[1]" below corresponds to x0
    X = np.array([[1] + l[1:] for l in lines])
    # "or -1" converts 0 values to -1
    Y = np.array([l[0] or -1 for l in lines])
    f.close()

    return X, Y


