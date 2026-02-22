#!/usr/bin/env python3
"""Implementation of a random isolation forest."""

import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """Class to implement a forest of random isolation trees."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the random isolation forest.

        Args:
            n_trees (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            min_pop (int): Minimum population to split a node
            seed (int): Seed for random generation
        """
        self.numpy_preds = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the average depths for the input data.

        Args:
            explanatory: Explanatory data

        Returns:
            Average of predictions from all trees
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=None, verbose=0):
        """
        Trains the isolation forest on the data.

        Args:
            explanatory: Training data
            n_trees (int): Number of trees to train
            verbose (int): Verbosity level
        """
        if n_trees is None:
            n_trees = self.n_trees
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            mean_depth = np.array(depths).mean()
            mean_nodes = np.array(nodes).mean()
            mean_leaves = np.array(leaves).mean()
            print(f"""  Training finished.
    - Mean depth                     : {mean_depth}
    - Mean number of nodes           : {mean_nodes}
    - Mean number of leaves          : {mean_leaves}""")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the rows with the smallest average depths.

        Args:
            explanatory: Explanatory data
            n_suspects (int): Number of suspects to return

        Returns:
            The n_suspects rows with the smallest depths and their depths
        """
        depths = self.predict(explanatory)
        # Get indices sorted by increasing order of depths
        sorted_i = np.argsort(depths)
        # Return the first n_suspects indices (smallest depths)
        suspect_i = sorted_i[:n_suspects]
        # Return the corresponding rows and depths
        return explanatory[suspect_i], depths[suspect_i]
