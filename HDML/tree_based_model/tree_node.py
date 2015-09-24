import sys
from HDML.util.functions import mean

class tree_node:
    def __init__(self, lst_sample):
        self.lst_sample = lst_sample
        self.n_sample = len(self.lst_sample)
        self.left_child = None
        self.right_child = None
        self.split_fea = None
        self.split_val = 0
        '''
        self.left_child_tmp = None
        self.right_child_tmp = None
        self.split_fea_tmp = None
        self.split_val_tmp = 0
        '''
    def try_split_node(self, criteria = 'MSE'):
        if not lst_sample:
            raise ValueError("Empty lst_sample! Can NOT split any more!")
        if criteria == "MSE":
            mse_min = sys.maxint
            self.split_fea = -1
            self.split_val = 0
            self.left_child = None
            self.right_child = None
            for i_fea in len(self.lst_sample[0] - 1):
                for sample_split in self.lst_sample:
                    lst_sample_left = []
                    lst_sample_right = []
                    flt_split = sample_split[i_fea]
                    for sample in self.lst_sample:
                        if sample[i_fea] <= flt_split:
                            lst_sample_left.append(sample)
                        else:
                            lst_sample_right.append(sample)
                    mean_left = mean([sample[-1] for sample in lst_sample_left])
                    mean_right = mean([sample[-1] for sample in lst_sample_right])
                    mse = sum([(sample[-1] - mean_left) ** 2 for sample in mean_left]) + sum([(sample[-1] - mean_right) ** 2 for sample in mean_right])
                    if mse < mse_min:
                        mse_min = mse
                        self.split_fea_tmp = i_fea
                        self.split_val_tmp = sample_split[i_fea]
                        self.left_child_tmp = tree_node(lst_sample_left)
                        self.right_child_tmp = tree_node(lst_sample_right)
            return (mse_min, self.split_fea_tmp, self.split_val_tmp, self.left_child_tmp, self.right_child_tmp)
        elif criteria == "IG":
            pass

    def split_node(self, split_fea, split_val):
        lst_sample_left = []
        lst_sample_right = []
        for sample in self.lst_sample:
            if sample[split_fea] < = split_val:
                lst_sample_left.append(sample)
            else:
                lst_sample_right.append(sample)
        self.left_child = tree_node(lst_sample_left)
        self.right_child = tree_node(lst_sample_right)

            



