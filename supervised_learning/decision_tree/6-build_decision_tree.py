#!/usr/bin/env python3
"""
Decision tree prediction module.

This module defines Node, Leaf, and Decision_Tree classes and implements
efficient prediction methods.
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
        """Update bounds for subtree."""
        if self.is_root:
            self.upper = {i: np.inf for i in self.lower.keys()}
            self.lower = {i: -np.inf for i in self.upper.keys()}

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
        """Create indicator function for node."""
        def is_large_enough(x):
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

    def pred(self, x):
        """Recursive prediction for one sample."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


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
        """Use parent indicator method."""
        super().update_indicator()

    def pred(self, x):
        """Return leaf value."""
        return self.value


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
        """Return all leaves."""
        leaves = []

        def walk(node):
            if node.is_leaf:
                leaves.append(node)
            else:
                walk(node.left_child)
                walk(node.right_child)

        walk(self.root)
        return leaves

    def update_predict(self):
        """Create vectorized predict function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.sum(
            np.array([leaf.value * leaf.indicator(A) for leaf in leaves]),
            axis=0
        )

    def pred(self, x):
        """Recursive prediction for one sample."""
        return self.root.pred(x)
