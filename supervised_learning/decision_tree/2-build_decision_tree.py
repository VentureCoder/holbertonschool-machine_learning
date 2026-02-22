#!/usr/bin/env python3
"""
Decision tree printing module.

This module defines Node, Leaf, and Decision_Tree classes and implements
string representations to display the structure of a decision tree.
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
            is_root (bool): True if node is root.
            depth (int): Depth of the node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def left_child_add_prefix(self, text):
        """
        Add prefix formatting for the left child branch.
        """
        lines = text.split("\n")
        new_text = "    +--->" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |   " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Add prefix formatting for the right child branch.
        """
        lines = text.split("\n")
        new_text = "    +--->" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "        " + x + "\n"
        return new_text

    def __str__(self):
        """
        Return a string representation of this node and its subtree.
        """
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = (
                f"-> node [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )

        left_str = self.left_child.__str__()
        right_str = self.right_child.__str__()

        text += "\n" + self.left_child_add_prefix(left_str)
        text += self.right_child_add_prefix(right_str)
        return text.rstrip("\n")


class Leaf(Node):
    """Class representing a leaf in a decision tree."""

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.

        Args:
            value: Leaf prediction value.
            depth (int): Depth of the leaf.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Return string representation of a leaf."""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Class representing a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize Decision_Tree.

        Args:
            max_depth (int): Max tree depth.
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

    def __str__(self):
        """Return string representation of the decision tree."""
        return self.root.__str__()
