#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def pca(X, var=0.95):
    """Principal Component Analysis"""
    U, S, Vt = np.linalg.svd(X)

    total_var_explain = 0
    idx = 0

    normal_S = S / np.sum(S)

    for var_explain in normal_S:
        total_var_explain += var_explain
        idx += 1
        if total_var_explain >= var:
            break

    return Vt.T[..., :idx]
