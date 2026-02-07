#!/usr/bin/env python3
"""
Normal distribution class
"""


class Normal:
    """Class representing a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = len(data)
            mean = sum(data) / n
            var = sum((x - mean) ** 2 for x in data) / n

            self.mean = float(mean)
            self.stddev = float(var ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the PDF value for a given x-value
        """
        pi = 3.141592653589793
        coef = 1 / (self.stddev * (2 * pi) ** 0.5)
        exp = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coef * (2.718281828459045 ** exp)
