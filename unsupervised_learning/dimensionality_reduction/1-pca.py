#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def pca(X, ndim):
    """Principal Component Analysis"""
    X = X - np.mean(X, axis=0)
    _, __, Vt = np.linalg.svd(X)
    return np.matmul(X, Vt.T[..., :ndim])
