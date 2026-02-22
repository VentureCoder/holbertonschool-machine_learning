#!/usr/bin/env python3
"""
Decision tree module.

This module defines Node, Leaf, and Decision_Tree classes and provides
methods to retrieve all leaves of a decision tree.
"""

import numpy as np


class Node:
    """Class representing an internal node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialize a Node.

        Args:
            feature (int): Feature index.
            threshold (float): Threshold value.
            left_child (Node): Left child.
            right_child (Node): Right child.
            is_root (bool): True if root node.
            depth (int): Depth of node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def get_leaves_below(self):
        """
        Return all leaves below this node.

        Returns:
            list: List of Leaf objects.
        """
        left_leaves = self.left_child.get_leaves_below()
        right_leaves = self.right_child.get_leaves_below()
        return left_leaves + right_leaves


class Leaf(Node):
    """Class representing a leaf node in a decision tree."""

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.

        Args:
            value: Predicted value.
            depth (int): Depth of leaf.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def get_leaves_below(self):
        """
        Return this leaf as a list.

        Returns:
            list: List containing this leaf.
        """
        return [self]


class Decision_Tree:
    """Class representing a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a Decision_Tree.

        Args:
            max_depth (int): Maximum depth.
            min_pop (int): Minimum population.
            seed (int): Random seed.
            split_criterion (str): Split criterion.
            root (Node): Root node.
        """
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

    def get_leaves(self):
        """
        Return all leaves of the decision tree.

        Returns:
            list: List of Leaf objects.
        """
        return self.root.get_leaves_below()
