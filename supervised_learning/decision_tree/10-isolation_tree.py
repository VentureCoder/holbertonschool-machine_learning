#!/usr/bin/env python3
"""Implementation of an isolation random tree."""

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    Random isolation tree for anomaly detection.

    This class implements a decision tree used in isolation algorithms
    to detect anomalies in data. The tree performs random splits on
    features to isolate anomalous data points.
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initializes a random isolation tree.

        Args:
            max_depth (int): Maximum depth of the tree (default: 10)
            seed (int): Seed for random number generation (default: 0)
            root (Node): Existing root node (default: None, creates a new node)
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        Returns a textual representation of the tree.

        Returns:
            str: String representation of the tree
        """
        return self.root.__str__()

    def depth(self):
        """
        Calculates the maximum depth of the tree.

        Returns:
            int: The maximum depth of the tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes in the tree.

        Args:
            only_leaves (bool): If True, counts only leaves (default: False)

        Returns:
            int: The number of nodes (or leaves if only_leaves=True)
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Updates the bounds of all nodes in the tree.

        Necessary for the proper functioning of predictions.
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        Retrieves all the leaves of the tree.

        Returns:
            list: List of all leaf nodes in the tree
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Updates the prediction function of the tree.

        Configures leaf indicators and creates a lambda function
        that calculates predictions by summing the contributions of each leaf.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: sum(
            leaf.indicator(A) * leaf.value
            for leaf in leaves)

    def np_extrema(self, arr):
        """
        Calculates the minimum and maximum values of an array.

        Args:
            arr (numpy.array): Array for which to find the extrema

        Returns:
            tuple: (minimum_value, maximum_value)
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Generates a random split criterion for a node.

        Randomly chooses a feature and a split threshold.

        Args:
            node (Node): The node for which to generate the split criterion

        Returns:
            tuple: (feature, threshold) - the feature index and the threshold
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Creates a child leaf node with the given sub-population.

        Args:
            node (Node): The parent node
            sub_population (numpy.array): Boolean mask of the sub-population

        Returns:
            Leaf: The created leaf node
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates a child node (non-leaf) with the given sub-population.

        Args:
            node (Node): The parent node
            sub_population (numpy.array): Boolean mask of the sub-population

        Returns:
            Node: The created child node
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively fits a node by creating its children.

        Determines whether to create leaves or continue splitting
        based on depth and population size.

        Args:
            node (Node): The node to fit
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = (node.sub_population) & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = (node.sub_population) & (
            self.explanatory[:, node.feature] <= node.threshold)

        # Is left node a leaf ?
        is_left_leaf = (node.depth >= self.max_depth - 1) or (
            left_population.sum() <= 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (node.depth >= self.max_depth - 1) or (
            right_population.sum() <= 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Trains the isolation tree on the provided explanatory data.

        Builds the full tree and prepares the prediction function.

        Args:
            explanatory (numpy.array): Training data (features)
            verbose (int): Verbosity level (0=silent, 1=display stats)
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(
            explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
