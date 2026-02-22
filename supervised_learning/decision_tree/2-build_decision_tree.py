#!/usr/bin/env python3
"""
Build a decision tree.
"""
import numpy as np


class Node:
    """Class representing an internal node of a decision tree"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a node"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculates the max depth below the current node"""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes below the current node"""
        return (self.left_child.count_nodes_below(only_leaves=only_leaves) +
                self.right_child.count_nodes_below(only_leaves=only_leaves) +
                (0 if only_leaves else 1))

    def left_child_add_prefix(self, text):
        """Add prefix for left child."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix for right child."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return an ASCII representation."""
        if self.is_root:
            out = (f"root [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        else:
            out = (f"node [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")

        if self.left_child:
            out += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            out += self.right_child_add_prefix(str(self.right_child))
        return out


class Leaf(Node):
    """Class representing a leaf of a decision tree"""

    def __init__(self, value, depth=None):
        """Initializes a leaf"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes below the current node"""
        return 1

    def __str__(self):
        """Returns the string representation of the leaf"""
        return "leaf [value={}]".format(self.value)


class Decision_Tree():
    """Class representing a decision tree"""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes a decision tree"""
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
        """Calculates the depth of the decision tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts the number of nodes in the decision tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Returns the string representation of the decision tree"""
        return self.root.__str__()
