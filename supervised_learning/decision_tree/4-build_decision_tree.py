#!/usr/bin/env python3
"""
Decision tree bounds module.

This module defines Node, Leaf, and Decision_Tree classes and implements
methods to propagate feature bounds down the tree.
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
        """
        Recursively update lower and upper bounds for all nodes below.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            # Copy parent's bounds
            child.lower = dict(self.lower)
            child.upper = dict(self.upper)

            # Update bounds according to split
            if child is self.left_child:
                # Left child: x_feature <= threshold
                child.upper[self.feature] = min(
                    child.upper.get(self.feature, np.inf),
                    self.threshold
                )
            else:
                # Right child: x_feature > threshold
                child.lower[self.feature] = max(
                    child.lower.get(self.feature, -np.inf),
                    self.threshold
                )

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()


class Leaf(Node):
    """Class representing a leaf in a decision tree."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def update_bounds_below(self):
        """Leaf has no children; bounds already set."""
        pass


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
        """Update bounds for the entire tree."""
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
