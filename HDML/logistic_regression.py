from functions import sigmoid, sigmoid_boundary, log_likelihood_grad, log_likelihood, logistic
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.optimize

class logistic_regression:
    def __init__(self, data, label, n_iter, eta, beta):
        '''
        eta: float, learning rate
        beta: float, coefficient of regularization term
        '''
        self.n_iter = n_iter
        self.eta = eta
        self.beta = beta
        self.label = label
        self.n_feature = len(data[0])
        self.data = data
        self.data_original = data
        # Inteception
        self.data = np.insert(self.data, self.n_feature , 1.0, axis = 1)
        self.W = np.random.rand(self.n_feature + 1)
    
    def NLL(self, w):
        self.likelihood = 0.0
        for index, sample in enumerate(self.data):
            mu = sigmoid(w, sample)
            label = self.label[index]
            self.likelihood -= np.log(mu ** label * (1 - mu) ** (1 - label)) 
        return self.likelihood +  self.beta / 2 * np.dot(w, w)

    def gradient(self, w, lst_data, lst_label):
        g = np.zeros(len(w))
        for index, data in enumerate(lst_data):
            mu = sigmoid(w, data)
            label = lst_label[index]
            g += data * (mu - label) 
        g += np.array(w) * self.beta
        return g

    def batch_gradient_descent(self):
        for n_iter in range(self.n_iter):
            g = self.gradient(self.W, self.data, self.label)
            #self.W = [x - self.eta * y for x, y in zip(self.W, g)]
            self.W -= self.eta * g
            print self.NLL(self.W)
        return self.W

    def stochastic_gradient_descent(self):
        for n_iter in range(self.n_iter):
            for j in range(len(self.data)):
                index = rd.randint(0, len(self.data) - 1)
                data = self.data[index]
                g = self.gradient(self.W, [data], [self.label[index]])
                #self.W = [x - self.eta * y for x, y in zip(self.W, g)]
                self.W -= self.eta * g
            print self.NLL(self.W)
        return self.W
    def coordinate_gradient_descent(self):
        for n_iter in range(self.n_iter):
            pass

    def bfgs(self):
        def f(w):
            return self.NLL(w)
        
        def fprime(w):
            return self.gradient(w, self.data, self.label)

        self.W, lst_W = scipy.optimize.fmin_bfgs(f, self.W, fprime, disp=True, retall = True, maxiter = self.n_iter)
        for W in lst_W:
            print self.NLL(W)
        return self.W
    
    def cg(self):
        def f(w):
            return self.NLL(w)
        
        def fprime(w):
            return self.gradient(w, self.data, self.label)

        self.W, lst_W = scipy.optimize.fmin_cg(f, self.W, fprime, disp=True, retall = True, maxiter = self.n_iter)
        for W in lst_W:
            print self.NLL(W)
        return self.W
    
    def tnc(self):
        def f(w):
            return self.NLL(w)
        
        def fprime(w):
            return self.gradient(w, self.data, self.label)

        self.W, lst_W = scipy.optimize.fmin_tnc(f, self.W, fprime, disp=True)
        for W in lst_W:
            print self.NLL(W)
        return self.W
    
    def powell(self):
        def f(w):
            return self.NLL(w)
        
        self.W, lst_W = scipy.optimize.fmin_powell(f, self.W, retall = True, maxiter = self.n_iter)
        for W in lst_W:
            print self.NLL(W)
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
        x_min = min([d[0] for d in self.data_original])
        x_max = max([d[0] for d in self.data_original])
        plt.plot([x_min, x_max], [sigmoid_boundary(x_min, self.W), sigmoid_boundary(x_max, self.W)], 'b-')
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

def generate_synthetic_data(n_sample = 1000):
    lst_data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_sample)
    lst_label1 = [1.0 for i in range(n_sample)]
    lst_data2 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], n_sample)
    lst_label2 = [0.0 for i in range(n_sample)]
    lst_data = np.append(lst_data1, lst_data2, axis=0)
    lst_label = lst_label1 + lst_label2
    return lst_data, lst_label

W_init = np.random.rand(2)
n_iter = 20
eta = 0.0005
beta = 0.5
lst_data, lst_label = generate_synthetic_data()

print 'Batch'
lr1 = logistic_regression(lst_data, lst_label, n_iter, eta, beta)
lr1.set_initial_weight(W_init)
W = lr1.batch_gradient_descent()
#lr1.plot_data()
print W

print 'Stochastic'
lr2 = logistic_regression(lst_data, lst_label, n_iter,eta, beta)
lr2.set_initial_weight(W_init)
W = lr2.stochastic_gradient_descent()
#lr2.plot_data()
print W
'''
print 'BFGS'
lr3 = logistic_regression(lst_data, lst_label, n_iter, eta, beta)
lr3.set_initial_weight(W_init)
W = lr3.bfgs()
#lr3.plot_data()
print W
print 'CG'
lr3 = logistic_regression(lst_data, lst_label, n_iter, eta, beta)
lr3.set_initial_weight(W_init)
W = lr3.cg()
print W
print 'TNC'
lr3 = logistic_regression(lst_data, lst_label, n_iter, eta, beta)
lr3.set_initial_weight(W_init)
W = lr3.tnc()
print W
print 'powell'
lr3 = logistic_regression(lst_data, lst_label, n_iter, eta, beta)
lr3.set_initial_weight(W_init)
W = lr3.powell()
print W
'''
