#!/usr/bin/env python3
"""
8-ridin_bareback module
Defines a function that performs matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication and returns a new matrix."""
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for row in mat1:
        new_row = []
        for col in zip(*mat2):
            new_row.append(sum(a * b for a, b in zip(row, col)))
        result.append(new_row)

    return result
