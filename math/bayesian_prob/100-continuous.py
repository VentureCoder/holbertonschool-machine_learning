#!/usr/bin/env python3
"""
Calculates the posterior probability that p is within a given range
using a continuous Beta posterior
"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that p is in [p1, p2]
    given x successes out of n trials
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    a = x + 1
    b = n - x + 1

    return special.betainc(a, b, p2) - special.betainc(a, b, p1)
