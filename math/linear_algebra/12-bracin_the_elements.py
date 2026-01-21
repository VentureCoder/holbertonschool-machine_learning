#!/usr/bin/env python3
"""
12-bracin_the_elements module
Defines a function that performs element-wise operations using NumPy.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication,
    and division on two numpy arrays or array and scalar.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
