#!/usr/bin/env python3
"""
13-cats_got_your_tongue module
Defines a function that concatenates numpy arrays along a given axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two numpy arrays along a specified axis."""
    return np.concatenate((mat1, mat2), axis=axis)
