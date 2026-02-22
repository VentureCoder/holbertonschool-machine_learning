#!/usr/bin/env python3
"""
Decision tree module.

This module defines the Node, Leaf, and Decision_Tree classes and includes
methods for computing properties of a decision tree such as the maximum depth.
"""

import numpy as np


class Node:
    """Class representing an internal node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialize a Node.

        Args:
            feature (int): Feature index used for splitting.
            threshold (float): Threshold value for the split.
            left_child (Node): Left child node.
            right_child (Node): Right child node.
            is_root (bool): True if this node is the root.
            depth (int): Depth of this node in the tree.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Return the maximum depth found below this node.

        Returns:
            int: Maximum depth in the subtree rooted at this node.
        """
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)


class Leaf(Node):
    """Class representing a leaf node in a decision tree."""

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf node.

        Args:
            value: Predicted value at the leaf.
            depth (int): Depth of this leaf in the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of this leaf.

        Returns:
            int: Depth of the leaf.
        """
        return self.depth


class Decision_Tree:
    """Class representing a decision tree model."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a Decision_Tree.

        Args:
            max_depth (int): Maximum depth allowed for the tree.
            min_pop (int): Minimum population for a node to be split.
            seed (int): Random seed.
            split_criterion (str): Criterion for splitting.
            root (Node): Root node of the tree.
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

    def depth(self):
        """
        Return the maximum depth of the decision tree.

        Returns:
            int: Maximum depth of the tree.
        """
        return self.root.max_depth_below()
