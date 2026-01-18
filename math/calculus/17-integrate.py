#!/usr/bin/env python3
"""
17-integrate module
Contains a function to compute the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): List of coefficients where index represents power of x.
        C (int): Integration constant.

    Returns:
        list: Coefficients of the integrated polynomial, or None if invalid.
    """
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    if len(poly) == 0:
        return None

    result = [C]

    for i, coef in enumerate(poly):
        if not isinstance(coef, (int, float)):
            return None

        value = coef / (i + 1)

        if value.is_integer():
            value = int(value)

        result.append(value)

    # Remove trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
