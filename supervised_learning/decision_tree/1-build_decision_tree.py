#!/usr/bin/env python3
"""
Decision tree module.

This module defines the Node, Leaf, and Decision_Tree classes and provides
methods to compute properties such as tree depth and number of nodes.
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

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this node.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes in the subtree.
        """
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )

        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count


class Leaf(Node):
    """Class representing a leaf node in a decision tree."""

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.

        Args:
            value: Predicted value at the leaf.
            depth (int): Depth of this leaf in the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this leaf.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Always returns 1 for a leaf.
        """
        return 1


class Decision_Tree:
    """Class representing a decision tree model."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a Decision_Tree.

        Args:
            max_depth (int): Maximum depth allowed.
            min_pop (int): Minimum population to split a node.
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

    def count_nodes(self, only_leaves=False):
        """
        Count nodes in the decision tree.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
