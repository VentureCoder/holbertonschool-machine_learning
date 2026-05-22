#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    if type(X) is not np.ndarray or type(k) is not int or k <= 0:
        return None

    if len(X.shape) != 2:
        return None

    n, d = X.shape
    if n == 0:
        return None
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    centroids = np.random.uniform(mins, maxs, size=(k, d))
    return centroids
