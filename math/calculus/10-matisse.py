#!/usr/bin/env python3
"""
10-matisse module
Contains a function that computes the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly (list): list of coefficients

    Returns:
        list: derivative polynomial
        None: if poly is not valid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power in range(1, len(poly)):
        derivative.append(poly[power] * power)

    if all(val == 0 for val in derivative):
        return [0]

    return derivative
