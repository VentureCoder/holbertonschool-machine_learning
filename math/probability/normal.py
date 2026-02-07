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

    def pdf(self, x):
        """
        Calculates the PDF value for x
        """
        pi = 3.1415926536
        e = 2.7182818285
        z = (x - self.mean) / self.stddev
        return (1 / (self.stddev * (2 * pi) ** 0.5)) * e ** (-0.5 * z ** 2)

    def cdf(self, x):
        """
        Calculates the CDF value for x
        """
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)

        # Approximate erf(z) using Taylor series
        erf = 0
        pi = 3.1415926536

        for n in range(10):
            num = (-1) ** n * z ** (2 * n + 1)
            den = (self._factorial(n) * (2 * n + 1))
            erf += num / den

        erf *= 2 / pi ** 0.5
        return 0.5 * (1 + erf)

    def _factorial(self, n):
        """Computes factorial of n"""
        fact = 1
        for i in range(1, n + 1):
            fact *= i
        return fact
