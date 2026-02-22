#!/usr/bin/env python3
"""
Decision tree indicator module.

This module defines Node, Leaf, and Decision_Tree classes and implements
indicator functions used for prediction.
"""

import numpy as np


class Node:
    """Class representing an internal node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def update_bounds_below(self):
        """Update lower and upper bounds for subtree."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = dict(self.lower)
            child.upper = dict(self.upper)

            if child is self.left_child:
                child.upper[self.feature] = min(
                    child.upper.get(self.feature, np.inf),
                    self.threshold
                )
            else:
                child.lower[self.feature] = max(
                    child.lower.get(self.feature, -np.inf),
                    self.threshold
                )

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Create indicator function based on node bounds.
        """

        def is_large_enough(x):
            # True if x[:, feature] > lower bound for all constrained features
            if len(self.lower) == 0:
                return np.ones(x.shape[0], dtype=bool)
            return np.all(
                np.array([
                    np.greater(x[:, key], self.lower[key])
                    for key in self.lower.keys()
                ]),
                axis=0
            )

        def is_small_enough(x):
            # True if x[:, feature] <= upper bound for all constrained features
            if len(self.upper) == 0:
                return np.ones(x.shape[0], dtype=bool)
            return np.all(
                np.array([
                    np.less_equal(x[:, key], self.upper[key])
                    for key in self.upper.keys()
                ]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )


class Leaf(Node):
    """Class representing a leaf in a decision tree."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def update_bounds_below(self):
        """Leaf has no children."""
        pass

    def update_indicator(self):
        """Leaf uses Node's indicator update."""
        super().update_indicator()


class Decision_Tree:
    """Class representing a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def update_bounds(self):
        """Update bounds for all nodes."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return all leaves in the tree."""
        leaves = []

        def walk(node):
            if node.is_leaf:
                leaves.append(node)
            else:
                walk(node.left_child)
                walk(node.right_child)

        walk(self.root)
        return leaves
