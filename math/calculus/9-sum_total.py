#!/usr/bin/env python3
"""
9-sum_total module
Contains a function that computes the sum of i squared from 1 to n.
"""


def summation_i_squared(n):
    """
    Calculates the sum of the squares from 1 to n.

    Args:
        n (int): stopping condition

    Returns:
        int: sum of squares from 1 to n
        None: if n is not a valid number
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
