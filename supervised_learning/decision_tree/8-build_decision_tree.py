#!/usr/bin/env python3
"""
Decision Tree with Gini impurity splitting criterion.
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

    # ---------- Random split (baseline) ----------
    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            vals = self.explanatory[:, feature][node.sub_population]
            fmin, fmax = self.np_extrema(vals)
            diff = fmax - fmin
        x = self.rng.uniform()
        thr = (1 - x) * fmin + x * fmax
        return feature, thr

    # ---------- Gini splitting ----------
    def possible_thresholds(self, node, feature):
        values = np.unique(self.explanatory[:, feature][node.sub_population])
        if values.size <= 1:
            return np.array([])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        # data restricted to node
        idx = node.sub_population
        Xf = self.explanatory[:, feature][idx]           # (n,)
        y = self.target[idx]                             # (n,)
        thresholds = self.possible_thresholds(node, feature)  # (t,)
        if thresholds.size == 0:
            return 0.0, np.inf

        classes = np.unique(y)
        C = classes.size
        # One-hot for classes: (n, C)
        Yk = (y[:, None] == classes[None, :])

        # Left mask for each threshold: (n, t)
        L = Xf[:, None] > thresholds[None, :]

        # Count per class in left: (t, C)
        left_counts = (L[:, :, None] & Yk[:, None, :]).sum(axis=0)
        # Count per class in right: (t, C)
        right_counts = ((~L)[:, :, None] & Yk[:, None, :]).sum(axis=0)

        # totals
        left_tot = left_counts.sum(axis=1)   # (t,)
        right_tot = right_counts.sum(axis=1) # (t,)
        tot = left_tot + right_tot           # (t,)

        # Gini for left and right
        with np.errstate(divide='ignore', invalid='ignore'):
            p_left = left_counts / left_tot[:, None]
            p_right = right_counts / right_tot[:, None]

            gini_left = 1.0 - np.nansum(p_left ** 2, axis=1)
            gini_right = 1.0 - np.nansum(p_right ** 2, axis=1)

        gini_split = (left_tot / tot) * gini_left + (right_tot / tot) * gini_right

        j = np.nanargmin(gini_split)
        return thresholds[j], gini_split[j]

    def Gini_split_criterion(self, node):
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])], dtype=float)
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    # ---------- Training ----------
    def fit(self, explanatory, target, verbose=0):
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

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
        y = self.target[node.sub_population]
        if (node.depth == self.max_depth or
                y.size < self.min_pop or
                np.unique(y).size == 1):
            node.is_leaf = True
            node.value = np.bincount(y).argmax()
            return

        node.feature, node.threshold = self.split_criterion(node)

        Xf = self.explanatory[:, node.feature]
        left_pop = node.sub_population & (Xf > node.threshold)
        right_pop = node.sub_population & (Xf <= node.threshold)

        def is_leaf_child(sub_pop):
            ys = self.target[sub_pop]
            return (sub_pop.sum() < self.min_pop or
                    node.depth + 1 == self.max_depth or
                    np.unique(ys).size == 1)

        if is_leaf_child(left_pop):
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        if is_leaf_child(right_pop):
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        y = self.target[sub_population]
        value = np.bincount(y).argmax()
        leaf = Leaf(value)
        leaf.depth = node.depth + 1
        leaf.sub_population = sub_population
        return leaf

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    # ---------- Prediction ----------
    def update_predict(self):
        self.predict = lambda A: np.array([self.pred(x) for x in A])

    def pred(self, x):
        return self.root.pred(x)

    # ---------- Utils ----------
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
        L = len(self.get_leaves())
        return L if only_leaves else (2 * L - 1)

    def accuracy(self, test_explanatory, test_target):
        return np.mean(self.predict(test_explanatory) == test_target)
