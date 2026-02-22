#!/usr/bin/env python3
"""Implementation of a simple decision tree."""

import numpy as np


class Node:
    """Represents an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes an internal node of the decision tree."""
        self.feature = feature              # Index of the feature for splitting
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

    def __str__(self):
        s = f"-> node [feature={self.feature}, threshold={self.threshold}]"
        if self.is_root is True:
            s = f"root [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child is not None:
            s += "\n" + self.left_child_add_prefix(str(self.left_child))

        if self.right_child is not None:
            s += self.right_child_add_prefix(str(self.right_child))

        return s

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("       "+x) + "\n"
        return (new_text)

    def get_leaves_below(self):
        result = []
        if self.left_child is not None:
            result.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            result.extend(self.right_child.get_leaves_below())

        if self.left_child is None and self.right_child is None:
            return [self]

        return result

    def update_bounds_below(self):
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
        def is_large_enough(x):
           
            return np.all(np.array([
                np.greater(
                    x[:, key], self.lower[key])  # x[:, key] > self.upper[key]
                for key in list(self.lower.keys())]),
                          axis=0)  # axe des colonnes

        def is_small_enough(x):
           
            return np.all(np.array([
                np.less_equal(
                    x[:, key], self.upper[key])  # x[:, key] <= self.upper[key]
                for key in list(self.upper.keys())]),
                          axis=0)  # axe des colonnes

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x),
                      is_small_enough(x)]), axis=0)

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
       
        return [self]

    def __str__(self):
       
        return f"-> leaf [value={self.value}]"

    def update_bounds_below(self):
        pass

    def pred(self, x):
      
        return self.value


class Decision_Tree():

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialise un arbre de décision."""
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
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        
        return self.root.__str__()

    def get_leaves(self):
       
        return self.root.get_leaves_below()

    def update_bounds(self):
       
        self.root.update_bounds_below()

    def pred(self, x):
        return self.root.pred(x)

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()  # Récupères les feuilles
        for leaf in leaves:  # Parcourt les feuilles
            leaf.update_indicator()
        self.predict = lambda A: sum(
            leaf.indicator(A) * leaf.value
            for leaf in leaves)
