#!/usr/bin/env python3
"""
Normal distribution
"""


class Normal:
    """Normal distribution class"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Class constructor"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list) or len(data) < 2:
                raise ValueError("data must be a list with multiple values")
            self.mean = sum(data) / len(data)

            var = 0
            for x in data:
                var += (x - self.mean) ** 2
            self.stddev = (var / len(data)) ** 0.5

    def cdf(self, x):
        """
        Calculates the CDF value for x
        """
        pi = 3.1415926536

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        erf = 0
        for n in range(50):
            # factorial of n
            fact = 1
            for i in range(1, n + 1):
                fact *= i

            erf += ((-1) ** n) * (z ** (2 * n + 1)) / (fact * (2 * n + 1))

        erf *= (2 / (pi ** 0.5))

        return 0.5 * (1 + erf)
