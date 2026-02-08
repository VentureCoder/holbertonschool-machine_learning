#!/usr/bin/env python3
"""
Exponential distribution
"""


class Exponential:
    """Exponential distribution class"""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list) or len(data) < 2:
                raise ValueError("data must be a list with multiple values")
            mean = float(sum(data) / len(data))
            self.lambtha = 1 / mean

    def pdf(self, x):
        """
        Calculates the PDF value for x
        """
        if x < 0:
            return 0

        e = 2.7182818285

        return self.lambtha * (e ** (-self.lambtha * x))
