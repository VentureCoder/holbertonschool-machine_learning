#!/usr/bin/env python3
"""Implementation of a simple decision tree."""

import numpy as np


class Node:
    """Class representing an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes an internal node of the decision tree."""
        self.feature = feature              # Index of the feature for split
        self.threshold = threshold          # Split threshold for the feature
        self.left_child = left_child        # Left subtree (Node or Leaf)
        self.right_child = right_child      # Right subtree (Node or Leaf)
        self.is_leaf = False                # Indicates if this node is a leaf
        self.is_root = is_root              # Indicates if this node is the root
        self.sub_population = None
        self.depth = depth                  # Depth of the node in the tree

    def max_depth_below(self):
        """Returns the maximum depth.

        Recursive method that counts the max depth of a tree
        starting from a given node

        Returns:
            int: The largest value of the recursion results
        """
        # Empty list to retrieve the max depth of each branch
        result = []
        # If the left child is not empty
        if self.left_child is not None:
            # Recursion in the left child and append the result
            result.append(self.left_child.max_depth_below())
        if self.right_child is not None:  # Same for the right child
            result.append(self.right_child.max_depth_below())

        if not result:
            return 0

        return max(result)  # Returns the largest result in the list

    def count_nodes_below(self, only_leaves=False):
        """Counts the child nodes of a given node.

        Args:
            only_leaves (bool, optional):
            indicates if the return should be the number of nodes or leaves.
            Defaults to False.

        Returns:
            int: number of child nodes or leaves
        """
        count = 0

        # Traverse the left branches recursively
        # as long as the targeted node is not None
        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)

        # Traverse the right branches recursively
        # as long as the targeted node is not None
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        # If we want to count only leaves
        if only_leaves:
            # A node is a leaf if it has no left or right children
            if self.left_child is None and self.right_child is None:
                return 1  # This node is a leaf
            else:
                # Otherwise, return the sum of the leaves
                # found in the subtrees
                return count

        # If self.left_child/right_child == None then add 1
        # to the number of nodes found
        return 1 + count

    def __str__(self):
        """Displays the node and its children as a string."""
        s = f"-> node [feature={self.feature}, threshold={self.threshold}]"
        if self.is_root is True:
            s = f"root [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child is not None:
            s += "\n" + self.left_child_add_prefix(str(self.left_child))

        if self.right_child is not None:
            s += self.right_child_add_prefix(str(self.right_child))

        return s

    def left_child_add_prefix(self, text):
        """Adds a prefix to display the left subtree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Adds a prefix to display the right subtree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def get_leaves_below(self):
        """
        Returns the list of all leaves under this node (recursive).

        Returns:
            list: List of descendant leaf objects.
        """
        result = []
        if self.left_child is not None:
            result.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            result.extend(self.right_child.get_leaves_below())

        if self.left_child is None and self.right_child is None:
            return [self]

        return result

    def update_bounds_below(self):
        """
        Recursively updates the bounds (upper/lower) for each node.

        Initializes the bounds at the root, then propagates the split 
        constraints to each child according to the branch (left/right).
        """
        if self.is_root:
            self.upper = {0: np.inf}  # {0: +inf}
            self.lower = {0: -1 * np.inf}  # {0: -inf}
            # Root: -inf < X < +inf

        for child in [self.left_child, self.right_child]:
            if child is None:
                continue

            # Copy parent bounds (upper & lower)
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

            if child == self.left_child:
                # Left child: feature > threshold
                # Update lower bound
                child.lower[self.feature] = self.threshold
            else:
                # Right child: feature <= threshold
                # Update upper bound
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()  # Recursion

    def update_indicator(self):
        """Updates the indicator attribute with a function.

        which checks if each sample respects the defined bounds.
        """
        def is_large_enough(x):
            """Check if all features in the input array `x`.

            are greater than their corresponding lower bounds.

            Args:
                x (np.ndarray): A 2D numpy array of shape
                (n_samples, n_features), where each row represents
                a sample and each column corresponds to a feature.

            Returns:
                np.ndarray: A boolean array of shape (n_samples,)
                indicating for each sample whether all its features are greater
                than the specified lower bounds in `self.lower`.
            """
            return np.all(np.array([
                np.greater(
                    x[:, key], self.lower[key])  # x[:, key] > self.lower[key]
                for key in list(self.lower.keys())]),
                          axis=0)  # column axis

        def is_small_enough(x):
            """Check if all features in the input array `x`.

            are less or equal than their corresponding upper bounds.

            Args:
                x (np.ndarray): A 2D numpy array of shape
                (n_samples, n_features), where each row represents
                a sample and each column corresponds to a feature.

            Returns:
                np.ndarray: A boolean array of shape (n_samples,)
                indicating for each sample whether all its features
                are less or equal than the specified upper
                bounds in `self.upper`.
            """
            return np.all(np.array([
                np.less_equal(
                    x[:, key], self.upper[key])  # x[:, key] <= self.upper[key]
                for key in list(self.upper.keys())]),
                          axis=0)  # column axis

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x),
                      is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Method that follows the decisions made at this node level.

        If the feature value of x is greater than the threshold, 
        it goes to the left child, otherwise to the right child, 
        and asks that child to continue the prediction.

        Args:
            x (array-like): A sample to predict.

        Returns:
            The prediction returned by the leaf reached.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Class representing a leaf of the decision tree."""

    def __init__(self, value, depth=None):
        """Initializes a leaf of the decision tree."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf.

        Returns:
            int: Depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes or leaves."""
        return 1

    def get_leaves_below(self):
        """
        Returns a list containing this leaf.

        Returns:
            list: List containing only this leaf.
        """
        return [self]

    def __str__(self):
        """
        Returns a readable representation of the leaf for display.

        Returns:
            str: Description of the leaf with its value.
        """
        return f"-> leaf [value={self.value}]"

    def update_bounds_below(self):
        """
        Method present for compatibility with Node.

        Does nothing for a leaf.
        """
        pass

    def pred(self, x):
        """When a sample reaches a leaf, the prediction is simple.

        It returns its own value

        Args:
            x (Any): The input sample for which the prediction is to be made.

        Returns:
            Any: The predicted value stored in the leaf node.
        """
        return self.value


class Decision_Tree():
    """Main class for the decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes a decision tree."""
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
        Returns the maximum depth of the tree.

        Returns:
            int: Maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts the nodes or leaves of the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a readable representation of the tree (root).

        Returns:
            str: Display of the tree root.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Returns the list of all the leaves of the tree.

        Returns:
            list: List of the leaf objects of the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Updates the bounds (upper/lower) for the whole tree.

        Starts the propagation from the root.
        """
        self.root.update_bounds_below()

    def pred(self, x):
        """Predicts the value for a sample x using the tree."""
        return self.root.pred(x)

    def update_predict(self):
        """Updates the prediction function of the tree according to the leaves."""
        self.update_bounds()
        leaves = self.get_leaves()  # Retrieve the leaves
        for leaf in leaves:  # Iterate through the leaves
            leaf.update_indicator()
        self.predict = lambda A: sum(
            leaf.indicator(A) * leaf.value
            for leaf in leaves)

    def fit(self, explanatory, target, verbose=0):
        """
        Trains the decision tree on the training data.

        Initializes the explanatory and target attributes, 
        creates the initial sub-population, then recursively builds 
        the tree by distributing individuals into nodes 
        according to split criteria, until the leaves. 
        Updates the prediction function and displays 
        diagnostic information if verbose=1.

        Reminder: an individual is an "object/observation/line of a dataset" 
        subject to the tree distributions.
        """
        # Choose the rule to split the nodes #
        if self.split_criterion == "random":  # Randomly
            self.split_criterion = self.random_split_criterion
        else:  # Gini criterion
            self.split_criterion = self.Gini_split_criterion  # \!
        # --- --- --- --- --- --- --- --- --- #
        # Store the training data in the object #
        self.explanatory = explanatory
        self.target = target
        # --- --- --- --- --- --- --- --- --- #
        # Create an array of True the same size as target #
        # This means that initially, all individuals pass through the root #
        self.root.sub_population = np.ones_like(self.target, dtype='bool')
        # --- --- --- --- --- --- --- --- --- #
        # Takes the current node, takes the selected split, #
        # separates individuals between left and right children, #
        # recursively calls fit_node to continue until the leaves. #
        # At each step, the sub_population of child nodes are updated #
        # to know who is still present. #
        self.fit_node(self.root)     # <--- to be defined later
        # --- --- --- --- --- --- --- --- --- #
        # Updates the tree prediction function based on the constructed leaves. #
        self.update_predict()     # <--- defined in the previous task
        # --- --- --- --- --- --- --- --- --- #
        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(
                self.explanatory, self.target)}""")

# --- --- --- --- --- --- --- --- --- --- --- --- --- #

    def np_extrema(self, arr):
        """Returns the minimum and maximum of a numpy array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Randomly determines a feature and a split threshold for a node.

        Selects a column (feature) randomly among the available features, 
        then chooses a random threshold between the minimum and maximum 
        value of this feature for the sub-population of the given node.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

# --- --- --- --- --- --- --- --- --- --- --- --- --- #

    def fit_node(self, node):
        """Recursively builds the tree from the given node.

        Determines the feature and split threshold for the node, 
        separates the sub-population into left and right branches, 
        then creates either a leaf or a child node for each branch 
        according to the stopping criteria (depth, size, purity).
        Recursively calls fit_node on non-leaf child nodes.
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = (node.sub_population) & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = (node.sub_population) & (
            self.explanatory[:, node.feature] <= node.threshold)

        # Is left node a leaf ?
        is_left_leaf = ((node.depth >= self.max_depth - 1) or (
            left_population.sum() < self.min_pop) or (
                np.unique(self.target[left_population]).size == 1))

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = ((node.depth >= self.max_depth - 1) or (
            right_population.sum() < self.min_pop) or (
                np.unique(self.target[right_population]).size == 1))

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf for the given sub-population
        with the majority class as the value.
        """
        # Calculates the majority class among the targets
        # of individuals in the sub-population (mode)
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal node for the given sub-population
        and updates its depth.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculates the model accuracy on a test dataset."""
        return np.sum(np.equal(self.predict(
            test_explanatory), test_target)) / test_target.size

    def possible_thresholds(self, node, feature):
        """
        Calculates the possible thresholds for a given feature in a node.

        Thresholds are the midpoints between each pair of unique values 
        of the considered feature, for the node's sub-population.

        Args:
            node (Node): The current node.
            feature (int): The index of the considered feature.

        Returns:
            np.ndarray: Array of possible thresholds.
        """
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[: -1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        Calculates the best threshold for a feature according to the Gini index.

        For each possible threshold, calculates the Gini impurity for the 
        left and right sub-populations, then returns the threshold that 
        minimizes the weighted average of the impurities.

        Args:
            node (Node): The current node.
            feature (int): The index of the considered feature.

        Returns:
            tuple: (optimal threshold, minimum Gini value)
        """
        n = node.sub_population.sum()  # number of individuals

        threshold = self.possible_thresholds(node, feature)
        t = threshold.size

        feature_value = self.explanatory[node.sub_population, feature]

        classes = np.unique(self.target[node.sub_population])
        c = classes.size

        class_matrix = (self.target[node.sub_population][
            :, None] == classes[None, :])

        threshold_matrix = feature_value[:, None] > threshold[None, :]

        Left_F = threshold_matrix[:, :, None] & class_matrix[:, None, :]

        left_count = Left_F.sum(axis=0)
        left_totals = left_count.sum(axis=1, keepdims=True)

        left_proportions = np.where(
            left_totals > 0, left_count / left_totals, 0)
        left_gini = 1 - np.sum(left_proportions ** 2, axis=1)

        Right_F = (~threshold_matrix[:, :, None]) & class_matrix[:, None, :]

        right_count = Right_F.sum(axis=0)
        right_totals = right_count.sum(axis=1, keepdims=True)

        right_proportions = np.where(
            right_totals > 0, right_count / right_totals, 0)
        right_gini = 1 - np.sum(right_proportions ** 2, axis=1)

        left_weights = left_totals[:, 0] / n
        right_weights = right_totals[:, 0] / n
        gini_average = left_weights * left_gini + right_weights * right_gini

        min_idx = np.argmin(gini_average)

        return threshold[min_idx], gini_average[min_idx]

    def Gini_split_criterion(self, node):
        """
        Determines the best feature and the best threshold according to Gini.

        Tests all features and selects the one that gives the lowest 
        Gini impurity after the split.

        Args:
            node (Node): The current node.

        Returns:
            tuple: (feature index, optimal threshold)
        """
        X = np.array([self.Gini_split_criterion_one_feature(
            node, i) for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
