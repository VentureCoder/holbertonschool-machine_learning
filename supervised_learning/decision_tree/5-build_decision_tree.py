#!/usr/bin/env python3
"""Implementation of a simple decision tree."""

import numpy as np


class Node:
    """Class representing an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes an internal node of a decision tree."""
        self.feature = feature              # Index of the feature for split
        self.threshold = threshold          # Seuil de split pour la feature
        self.left_child = left_child        # Sous-arbre gauche (Node ou Leaf)
        self.right_child = right_child      # Sous-arbre droit (Node ou Leaf)
        self.is_leaf = False                # Indique si ce nœud est une leaf
        self.is_root = is_root              # Indique si ce nœud est la racine
        self.sub_population = None
        self.depth = depth                  # Profondeur du nœud dans l'arbre

    def max_depth_below(self):
        """Returns the maximum depth.

        """
        # Liste vide pour récupérer la profondeur max de chaque branche
        result = []
        # Si l'enfant gauche n'est pas vide
        if self.left_child is not None:
            # Recursion dans l'enfant gauche et append le resultat
            result.append(self.left_child.max_depth_below())
        if self.right_child is not None:  # Pareil pour la droite
            result.append(self.right_child.max_depth_below())

        if not result:
            return 0

        return max(result)  # Renvoie le plus grand résultat de la liste

    def count_nodes_below(self, only_leaves=False):
        """Counts the child nodes of a given node.

        """
        count = 0

        # Parcourt les branches gauche récursivement
        # tant que le node visé n'est pas None
        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)

        # Parcourt les branches droite récursivement
        # tant que le node visé n'est pas None
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        # Si on souhaite compter uniquement les feuilles
        if only_leaves:
            # Un nœud est une feuille s'il n'a pas d'enfants gauche ni droit
            if self.left_child is None and self.right_child is None:
                return 1  # Ce nœud est une feuille
            else:
                # Sinon, retourne la somme des feuilles
                # trouvées dans les sous-arbres
                return count

        # Si self.left_child/right_child == None alors ajoute 1
        # au nombre de node trouvé
        return 1 + count

    def get_leaves_below(self):
        """
        Returns the list of all leaves below this node (recursive).

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
        Updates the bounds (upper/lower) for each node recursively.

        """
        if self.is_root:
            self.upper = {0: np.inf}  # {0: +∞}
            self.lower = {0: -1 * np.inf}  # {0: -∞}
            # Racine : -∞ < X < +∞

        for child in [self.left_child, self.right_child]:
            if child is None:
                continue

            # Copie les bornes du parent (upper & lower)
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

            if child == self.left_child:
                # Enfant gauche : feature <= threshold
                # MAJ de la borne inférieure
                child.lower[self.feature] = self.threshold
            else:
                # Enfant droit : feature > threshold
                # MAJ de la borne supérieure
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()  # Récursion

    def update_indicator(self):
        """
        Updates the `indicator` attribute with a function that checks
        if each sample respects the defined bounds.
        """
        def is_large_enough(x):
            """Check if all features in the input array `x`.
            """
            return np.all(np.array([
                np.greater(
                    x[:, key], self.lower[key])  # x[:, key] > self.upper[key]
                for key in list(self.lower.keys())]),
                          axis=0)  # axe des colonnes

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
                          axis=0)  # axe des colonnes

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x),
                      is_small_enough(x)]), axis=0)


class Leaf(Node):
    def __init__(self, value, depth=None):
        """
        Initializes a leaf node.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes or leaves.
        """
        return 1

    def get_leaves_below(self):
        """
        Returns a list containing this leaf.
        """
        return [self]

    def __str__(self):
        """
        Returns a human-readable representation of the leaf for display.
        """
        return f"-> leaf [value={self.value}]"

    def update_bounds_below(self):
        """
        Method present for compatibility with Node.

        Does nothing for a leaf.
        """
        pass


class Decision_Tree():
    """
    Main class for the decision tree.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes a decision tree.
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
        Returns the maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes or leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a human-readable representation of the tree (root).

        Returns:
            str: Affichage de la racine de l'arbre.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Returns the list of all leaves of the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Updates the bounds (upper/lower) for the entire tree.

        Launches the propagation from the root.
        """
        self.root.update_bounds_below()
