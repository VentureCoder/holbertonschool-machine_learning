#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def variance(X, C):
    """Calculate Variance"""
    if type(X) is not np.ndarray or type(C) is not np.ndarray:
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    n, d = X.shape
    k, d_C = C.shape
    if d != d_C:
        return None

    distances = np.linalg.norm(X[:, np.newaxis, :] -
                               C[np.newaxis, :, :], axis=2)
    clss = np.argmin(distances, axis=1)
    diff = X - C[clss]
    sq_dist = np.sum(diff ** 2, axis=1)
    var = np.sum(sq_dist)
    return var
