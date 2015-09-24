from HDML.tree_based_model.tree_node import tree_node
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import xml.dom.minidom
import sys

class regression_tree:
    def __init__(self, lst_sample, n_node):
        self.lst_sample = lst_sample
        self.n_node = n_node
        self.lst_leaf = []

    def fit(self):
        current_n_node = 0
        root = tree_node(self.lst_sample)
        mse_min, split_fea, split_val, left_child, right_child = root.try_split_node(criteria = 'MSE')
        root.split_node(split_fea, split_val)
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
            split_fea_global = None
            split_val_global = None
            
            for leaf in self.lst_leaf:
                mse_min, split_fea, split_val, left_child, right_child =  leaf.try_split_node(criteria = 'MSE')
                if mse_min < mse_min_global:
                    mse_min_global = mse_min
                    split_node_global = leaf
                    split_fea_global = split_fea
                    split_val_global = split_val

            split_node_global.split_node(split_fea_global, split_val_global)
            if split_node_global.left_child.n_sample > 1:
                self.lst_leaf.append(split_node_global.left_child)
            if split_node_global.right_child.n_sample > 1:
                self.lst_leaf.append(split_node_global.right_child)
            current_n_node += 1
        return root

    




