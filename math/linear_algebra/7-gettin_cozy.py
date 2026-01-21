#!/usr/bin/env python3
"""
7-gettin_cozy module
Defines a function that concatenates two 2D matrices along a given axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along the specified axis."""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        new_matrix = []
        for r1, r2 in zip(mat1, mat2):
            new_matrix.append(r1[:] + r2[:])
        return new_matrix

    return None
