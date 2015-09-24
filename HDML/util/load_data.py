import numpy as np

def generate_synthetic_data(n_sample = 1000):
    lst_data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_sample)
    lst_label1 = [1.0 for i in range(n_sample)]
    lst_data2 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], n_sample)
    lst_label2 = [0.0 for i in range(n_sample)]
    lst_data = np.append(lst_data1, lst_data2, axis=0)
    lst_label = lst_label1 + lst_label2
    return lst_data, lst_label

def read_dsv(path, delimiter = '\t', header = False):
    '''
    path: path to csv file
    header: whether the csv file has a header, default False
    '''
    X = []
    lst_name = []
    n_line = -1
    for line in open(path):
        lst = line.strip('\n').split(delimiter)
        n_line += 1
        if header:
            if n_line == 0:
                lst_name = lst
                continue
        X.append(lst)
    return np.array(X), lst_name

def read_libsvm(path):
    lst_data = []
    lst_label = []
    MAX = -1
    for line in open(path):
        lst = line.strip('\n').split('\t')
        lst_label.append(int(lst[0]))
        for feature in lst[1].split(' '):
            index = int(feature.split(':')[0])
            if index > MAX:
                MAX = index
    for line in open(path):
        l_sample = np.zeros(MAX)
        lst = line.strip('\n').split('\t')
        lst_label.append(int(lst[0]))
        for feature in lst[1].split(' '):
            ll = feature.split(':')
            index = int(ll[0])
            l_sample[index] = float(ll[1])
        lst_data.append(l_sample)
    return lst_data, lst_label

