#!/usr/bin/env python3
"""
Poisson distribution class
"""


class Poisson:
    """Class representing a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution

        data: list of values to estimate lambtha
        lambtha: expected number of occurrences
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

        k: number of successes
        Returns: PMF value
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        return (2.718281828459045 ** (-self.lambtha) *
                (self.lambtha ** k) / factorial)
