#!/usr/bin/env python3
"""
Trainable Decision Tree implementation.
"""

import numpy as np


class Node:
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

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def pred(self, x):
        return self.value


class Decision_Tree:
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

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            vals = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(vals)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype=bool)
        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory, self.target)}""")

    def fit_node(self, node):
        # Stop criteria
        y = self.target[node.sub_population]
        if (node.depth == self.max_depth or
                y.size < self.min_pop or
                np.unique(y).size == 1):
            node.is_leaf = True
            node.value = np.bincount(y).argmax()
            return

        node.feature, node.threshold = self.split_criterion(node)

        Xf = self.explanatory[:, node.feature]
        left_population = node.sub_population & (Xf > node.threshold)
        right_population = node.sub_population & (Xf <= node.threshold)

        def is_leaf(sub_pop):
            y_sub = self.target[sub_pop]
            return (sub_pop.sum() < self.min_pop or
                    node.depth + 1 == self.max_depth or
                    np.unique(y_sub).size == 1)

        if is_leaf(left_population):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        if is_leaf(right_population):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        y_sub = self.target[sub_population]
        value = np.bincount(y_sub).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def update_predict(self):
        leaves = self.get_leaves()
        self.predict = lambda A: np.array(
            [self.pred(x) for x in A]
        )

    def pred(self, x):
        return self.root.pred(x)

    def get_leaves(self):
        leaves = []

        def walk(n):
            if isinstance(n, Leaf):
                leaves.append(n)
            else:
                walk(n.left_child)
                walk(n.right_child)

        walk(self.root)
        return leaves

    def depth(self):
        return self.max_depth

    def count_nodes(self, only_leaves=False):
        leaves = self.get_leaves()
        if only_leaves:
            return len(leaves)
        return 2 * len(leaves) - 1

    def accuracy(self, test_explanatory, test_target):
        return np.mean(self.predict(test_explanatory) == test_target)
