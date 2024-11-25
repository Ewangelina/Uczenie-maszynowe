import numpy as np
from collections import Counter
import random
import string
import decisionTree

class randomForest:
    def __init__(self, X_train, y_train, tree_destination=".\\trees\\196.csv", no_trees = 10, uniformity=1, min_size=3):
        tree_prefix = tree_destination[:-4]
        trees_names = []
        for i in range(no_trees):
            curr_str = tree_prefix + "_" + str(i) + ".csv"
            trees_names.append(curr_str)
            tree = decisionTree.decisionTree(curr_str)
            tree.createTree(X_train, y_train, uniformity, min_size)
            
            

        self.trees_set = trees_names
        self.tree = decisionTree.decisionTree()

    def predict(self, row):
        results = []
        for tree_name in self.trees_set:
            res = self.tree.useTree(row, tree_name)
            results.append(res)

        print("Forest " + str(results))
        return Counter(results).most_common(1)[0][0]
        
            

    
