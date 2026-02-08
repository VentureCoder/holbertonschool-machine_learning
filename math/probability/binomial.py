#!/usr/bin/env python3
"""
Binomial distribution
"""


class Binomial:
    """Binomial distribution class"""

    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            var = 0
            for x in data:
                var += (x - mean) ** 2
            var /= len(data)

            # Estimate p first
            p = 1 - (var / mean)

            # Estimate n, then round
            n = round(mean / p)

            # Recalculate p using rounded n
            p = mean / n

            self.n = int(n)
            self.p = float(p)
