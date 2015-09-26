import sys
from HDML.util.functions import mean

class tree_node:
    def __init__(self, lst_sample, lst_feature):
        self.lst_feature = lst_feature
        self.lst_sample = lst_sample
        self.n_sample = len(self.lst_sample)
        self.left_child = None
        self.right_child = None
        self.split_fea = None
        self.split_fea_val = 0
        self.split_fea_index = 0
        self.split_fea_index_tmp = 0
        self.is_leaf = True
        self.predict_val = mean([sample[-1] for sample in self.lst_sample])
        '''
        self.left_child_tmp = None
        self.right_child_tmp = None
        self.split_fea_tmp = None
        self.split_fea_val_tmp = 0
        '''
    def try_split_node(self, criteria = 'MSE'):
        if  len(self.lst_sample) <= 1:
            raise ValueError("Empty lst_sample! Can NOT split any more!")
        if criteria == "MSE":
            mse_min = sys.maxint
            self.split_fea_index = -1
            self.split_fea_val = 0
            self.left_child = None
            self.right_child = None
            for index_fea in range(len(self.lst_sample[0]) - 1):
                lst_flt_split = [sample[index_fea] for sample in self.lst_sample]
                max_flt_split = max(lst_flt_split)
                min_flt_split = min(lst_flt_split)
                set_flt_split = set()
                for sample_split in self.lst_sample:
                    lst_sample_left = []
                    lst_sample_right = []
                    flt_split = sample_split[index_fea]
                    # If the split feature value is the max or min or has been evaluated before, then skip
                    if flt_split == max_flt_split or flt_split == min_flt_split or flt_split in set_flt_split:
                        continue
                    set_flt_split.add(flt_split)
                    for sample in self.lst_sample:
                        if sample[index_fea] <= flt_split:
                            lst_sample_left.append(sample)
                        else:
                            lst_sample_right.append(sample)
                    mean_left = mean([sample[-1] for sample in lst_sample_left])
                    mean_right = mean([sample[-1] for sample in lst_sample_right])
                    mse = sum([(sample[-1] - mean_left) ** 2 for sample in lst_sample_left]) + sum([(sample[-1] - mean_right) ** 2 for sample in lst_sample_right])
                    if mse < mse_min:
                        mse_min = mse
                        self.split_fea_index_tmp = index_fea
                        self.split_fea_val_tmp = sample_split[index_fea]
                        self.left_child_tmp = tree_node(lst_sample_left, self.lst_feature)
                        self.right_child_tmp = tree_node(lst_sample_right, self.lst_feature)
            return (mse_min, self.split_fea_index_tmp, self.split_fea_val_tmp, self.left_child_tmp, self.right_child_tmp)
        elif criteria == "IG":
            pass

    def split_node(self, split_fea_index, split_fea_val):
        self.is_leaf = False
        lst_sample_left = []
        lst_sample_right = []
        for sample in self.lst_sample:
            if sample[split_fea_index] <= split_fea_val:
                lst_sample_left.append(sample)
            else:
                lst_sample_right.append(sample)
        self.left_child = tree_node(lst_sample_left, self.lst_feature)
        self.right_child = tree_node(lst_sample_right, self.lst_feature)
        self.split_fea = self.lst_feature[split_fea_index]
        self.split_fea_index = split_fea_index
        self.split_fea_val = split_fea_val

            



