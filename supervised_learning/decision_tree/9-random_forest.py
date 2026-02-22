#!/usr/bin/env python3
"""
Random Forest implementation using Decision Trees with random splits.
"""

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree
import numpy as np


class Random_Forest():
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.explanatory = None

    def predict(self, explanatory):
        """
        Predict the class for each individual using majority vote
        over all trees in the forest.
        """
        all_preds = np.array([pred(explanatory) for pred in self.numpy_preds])

        def mode_1d(x):
            return np.bincount(x).argmax()

        return np.apply_along_axis(mode_1d, axis=0, arr=all_preds)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)

            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory, self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
