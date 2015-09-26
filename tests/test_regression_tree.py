import HDML.tree_based_model.regression_tree as regression_tree
import numpy as np

n_sample = 100
sample_train_fea = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_sample)
sample_train_label = np.zeros([n_sample, 1])
sample_train_label.fill(-1)
sample_train_pos = np.append(sample_train_fea, sample_train_label, axis = 1)

sample_train_fea = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], n_sample)
sample_train_label = np.zeros([n_sample, 1])
sample_train_label.fill(1)
sample_train_neg = np.append(sample_train_fea, sample_train_label, axis = 1)

#sample_train = np.append(sample_train_pos, sample_train_neg, axis = 0)

#print sample_train

rt = regression_tree.regression_tree(sample_train_neg, ['fea1', 'fea2'], 10)
rt.fit()
print rt.predict(sample_train_neg)
