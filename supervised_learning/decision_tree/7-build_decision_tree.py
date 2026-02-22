#!/usr/bin/env python3
"""Implémentation d'un arbre de décision simple."""

import numpy as np


class Node:
    """Classe représentant un nœud interne d'un arbre de décision."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialise un nœud interne de l'arbre de décision."""
        self.feature = feature              # Index de la feature pour split
        self.threshold = threshold          # Seuil de split pour la feature
        self.left_child = left_child        # Sous-arbre gauche (Node ou Leaf)
        self.right_child = right_child      # Sous-arbre droit (Node ou Leaf)
        self.is_leaf = False                # Indique si ce nœud est une leaf
        self.is_root = is_root              # Indique si ce nœud est la racine
        self.sub_population = None
        self.depth = depth                  # Profondeur du nœud dans l'arbre

    def max_depth_below(self):
        """Retourne la profondeur maximale.

        Méthode récursive qui compte la profondeur max d'un arbre
        à partir d'un nœud donné

        Returns:
            int: La plus grande valeur des résultats de la récursion
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
        """Compte les nodes enfant d'un node donné.

        Args:
            only_leaves (bool, optional):
            indique si le retour dois être le nombre de node ou de feuille.
            Defaults to False.

        Returns:
            int: nombre de node enfant ou de feuilles
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

    def __str__(self):
        """Affiche le nœud et ses enfants sous forme de chaîne."""
        s = f"-> node [feature={self.feature}, threshold={self.threshold}]"
        if self.is_root is True:
            s = f"root [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child is not None:
            s += "\n" + self.left_child_add_prefix(str(self.left_child))

        if self.right_child is not None:
            s += self.right_child_add_prefix(str(self.right_child))

        return s

    def left_child_add_prefix(self, text):
        """Ajoute un préfixe pour afficher le sous-arbre gauche."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Ajoute un préfixe pour afficher le sous-arbre droit."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("       "+x) + "\n"
        return (new_text)

    def get_leaves_below(self):
        """
        Retourne la liste de toutes les feuilles sous ce nœud (récursif).

        Returns:
            list: Liste des objets feuilles descendants.
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
        Met à jour récursivement les bornes (upper/lower) pour chaque nœud.

        Initialise les bornes à la racine, puis propage les contraintes
        de split à chaque enfant selon la branche (gauche/droite).
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
        """MAJ l'attribut `indicator` avec une fonction.

        qui vérifie si chaque échantillon respecte les bornes définies.
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

    def pred(self, x):
        """Méthode qui suit les décisions prises au niveau de ce nœud.

        Si la valeur de la caractéristique de x est supérieure
        au seuil (threshold), elle se dirige vers l'enfant gauche,
        sinon vers l'enfant droit, et demande à cet enfant de continuer
        la prédiction.

        Args:
            x (array-like): Un échantillon à prédire.

        Returns:
            La prédiction retournée par la feuille atteinte.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Classe représentant une feuille de l'arbre de décision."""

    def __init__(self, value, depth=None):
        """Initialise une feuille de l'arbre de décision."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Retourne la profondeur de la feuille.

        Returns:
            int: Profondeur de la feuille.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Compte le nombre de nœuds ou de feuilles."""
        return 1

    def get_leaves_below(self):
        """
        Retourne une liste contenant cette feuille.

        Returns:
            list: Liste contenant uniquement cette feuille.
        """
        return [self]

    def __str__(self):
        """
        Retourne une représentation lisible de la feuille pour l'affichage.

        Returns:
            str: Description de la feuille avec sa valeur.
        """
        return f"-> leaf [value={self.value}]"

    def update_bounds_below(self):
        """
        Méthode présente pour compatibilité avec Node.

        Ne fait rien pour une feuille.
        """
        pass

    def pred(self, x):
        """Quand un échantillon arrive à une feuille, la prédiction est simple.

        Elle retourne sa propre value

        Args:
            x (Any): The input sample for which the prediction is to be made.

        Returns:
            Any: The predicted value stored in the leaf node.
        """
        return self.value


class Decision_Tree():
    """Classe principale pour l'arbre de décision."""

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
        """
        Retourne la profondeur maximale de l'arbre.

        Returns:
            int: Profondeur maximale de l'arbre.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Compte les nœuds ou feuilles de l'arbre."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Retourne une représentation lisible de l'arbre (racine).

        Returns:
            str: Affichage de la racine de l'arbre.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Retourne la liste de toutes les feuilles de l'arbre.

        Returns:
            list: Liste des objets feuilles de l'arbre.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Met à jour les bornes (upper/lower) pour tout l'arbre.

        Lance la propagation à partir de la racine.
        """
        self.root.update_bounds_below()

    def pred(self, x):
        """Prédit la valeur pour un échantillon x en utilisant l'arbre."""
        return self.root.pred(x)

    def update_predict(self):
        """MAJ la fonction de prédiction de l'arbre selon les feuilles."""
        self.update_bounds()
        leaves = self.get_leaves()  # Récupères les feuilles
        for leaf in leaves:  # Parcourt les feuilles
            leaf.update_indicator()
        self.predict = lambda A: sum(
            leaf.indicator(A) * leaf.value
            for leaf in leaves)

    def fit(self, explanatory, target, verbose=0):
        """
        Entraîne l’arbre de décision sur les données d’entraînement.

        Initialise les attributs explanatory et target,
        crée la sous-population initiale, puis construit récursivement
        l’arbre en répartissant les individus dans les nœuds
        selon les critères de split, jusqu’aux feuilles.
        Met à jour la fonction de prédiction
        et affiche des informations de diagnostic si verbose=1.

        Rappel: un individus c'est un "objet/observation/ligne d'un dataset"
        soumis aux répartitions de l'arbre.
        """
        # Choisit la règle pour couper les nœuds #
        if self.split_criterion == "random":  # Aléatoirement
            self.split_criterion = self.random_split_criterion
        else:  # Critère Gini
            self.split_criterion = self.Gini_split_criterion  # \!
        # --- --- --- --- --- --- --- --- --- #
        # Stocke les données d'entrainement dans l'objet #
        self.explanatory = explanatory
        self.target = target
        # --- --- --- --- --- --- --- --- --- #
        # Créer un tableau de True de la même taille que target #
        # Ça veut dire qu'au départ, tous les individus passent par la racine #
        self.root.sub_population = np.ones_like(self.target, dtype='bool')
        # --- --- --- --- --- --- --- --- --- #
        # Prend le nœud courant prend le split selectionné, #
        # sépare les individus entre fils gauche et droit,#
        # appelle récursivement fit_node pour continuer jusqu'aux feuilles.#
        # À chaque étape, les sub_population des nœuds enfants sont MAJ #
        # pour savoir qui est encore présent. #
        self.fit_node(self.root)     # <--- to be defined later
        # --- --- --- --- --- --- --- --- --- #
        # Met à jour la fonction de prédiction de l'arbre #
        # en fonction des feuilles construites. #
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
        """Retourne le minimum et le maximum d’un tableau numpy."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Détermine aléatoirement une feature et un seuil de split pour un nœud.

        Sélectionne une colonne (feature) au hasard parmi les
        features disponibles,
        puis choisit un seuil aléatoire entre la valeur
        minimale et maximale de cette feature
        pour la sous-population du nœud donné.
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
        """Construit récursivement l’arbre à partir du nœud donné.

        Détermine la feature et le seuil de split pour le nœud,
        sépare la sous-population
        en branches gauche et droite, puis crée soit une feuille soit
        un nœud enfant pour chaque branche
        en fonction des critères d’arrêt (profondeur, taille, pureté).
        Appelle récursivement fit_node
        sur les nœuds enfants non-feuilles.
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
        """Crée une feuille pour la sous-population donnée.

        avec la classe majoritaire comme valeur.
        """
        # Calcule la classe majoritaire parmi les cibles
        # des individus de la sous-population (mode)
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Crée un nœud interne pour la sous-population donnée.

        et met à jour sa profondeur.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculate la précision du modèle sur un jeu de données de test."""
        return np.sum(np.equal(self.predict(
            test_explanatory), test_target)) / test_target.size
