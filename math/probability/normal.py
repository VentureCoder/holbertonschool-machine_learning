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

    def cdf(self, x):
        """
        Calculates the CDF value for a given x-value
        """
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)

        # Abramowitz and Stegun approximation of erf
        t = 1 / (1 + 0.3275911 * abs(z))
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        erf = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1)
                   * t * (2.718281828459045 ** (-z * z)))

        if z < 0:
            erf = -erf

        return 0.5 * (1 + erf)
