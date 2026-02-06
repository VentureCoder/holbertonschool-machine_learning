#!/usr/bin/env python3
"""
Poisson distribution class
"""

import math


class Poisson:
    """Class representing a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the PMF value for k successes
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        return (math.e ** (-self.lambtha) *
                (self.lambtha ** k) /
                math.factorial(k))
