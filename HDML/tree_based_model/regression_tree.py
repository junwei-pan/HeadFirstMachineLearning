from HDML.tree_based_model.tree_node import tree_node
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import xml.dom.minidom
import sys

class regression_tree:
    def __init__(self, lst_sample, lst_feature, n_node):
        self.lst_feature = lst_feature
        self.lst_sample = lst_sample
        self.n_node = n_node
        self.lst_leaf = []
        self.root = None

    def fit(self):
        current_n_node = 0
        root = tree_node(self.lst_sample, self.lst_feature)
        mse_min, split_fea_index, split_val, left_child, right_child = root.try_split_node(criteria = 'MSE')
        root.split_node(split_fea_index, split_val)
        if root.left_child.n_sample > 1:
            self.lst_leaf.append(root.left_child)
        if root.right_child.n_sample > 1:
            self.lst_leaf.append(root.right_child)
        current_n_node += 1
        while current_n_node < self.n_node:
            if not self.lst_leaf:
                raise ValueError("None leaf nodes now!")
            # Select a leaf to split among all leaves
            mse_min_global = sys.maxint
            split_node_global = None
            split_fea_index_global = None
            split_val_global = None
            
            for leaf in self.lst_leaf:
                mse_min, split_fea_index, split_val, left_child, right_child =  leaf.try_split_node(criteria = 'MSE')
                if mse_min < mse_min_global:
                    mse_min_global = mse_min
                    split_node_global = leaf
                    split_fea_index_global = split_fea_index
                    split_val_global = split_val

            split_node_global.split_node(split_fea_index_global, split_val_global)
            if split_node_global.left_child.n_sample > 1:
                self.lst_leaf.append(split_node_global.left_child)
            if split_node_global.right_child.n_sample > 1:
                self.lst_leaf.append(split_node_global.right_child)
            current_n_node += 1
        self.root = root

    def predict_one_sample(self, sample):
        # Go through the tree until the leaf node.
        current_node = self.root
        while not current_node.is_leaf:
            split_fea_index = current_node.split_fea_index
            split_fea_val = current_node.split_fea_val
            if sample[split_fea_index] <= split_fea_val:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        return current_node.predict_val

    def predict(self, lst_sample):
        lst_predict_val = []
        for sample in lst_sample:
            lst_predict_val.append(self.predict_one_sample(sample))
        return lst_predict_val
