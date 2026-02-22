#!/usr/bin/env python3
"""
Decision tree printing module.
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

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "+---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "| " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "+---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "  " + x + "\n"
        return new_text

    def __str__(self):
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = (
                f"node [feature={self.feature}, threshold={self.threshold}]"
            )

        left_str = self.left_child.__str__()
        right_str = self.right_child.__str__()

        left_prefixed = self.left_child_add_prefix(left_str)
        right_prefixed = self.right_child_add_prefix(right_str)

        return text + "\n" + left_prefixed + right_prefixed.rstrip("\n")


class Leaf(Node):
    """Class representing a leaf in a decision tree."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        return f"leaf [value={self.value}]"


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

    def __str__(self):
        return self.root.__str__()
