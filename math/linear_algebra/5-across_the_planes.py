#!/usr/bin/env python3
"""
5-across_the_planes module
Defines a function that adds two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise and returns a new matrix."""
    if len(mat1) != len(mat2):
        return None

    result = []
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
        result.append([a + b for a, b in zip(row1, row2)])

    return result
