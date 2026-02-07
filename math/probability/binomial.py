#!/usr/bin/env python3
"""
Binomial distribution class
"""


class Binomial:
    """Class representing a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution
        """
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
            p = mean / max(data)
            n = round(mean / p)
            p = mean / n

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """
        Calculates the PMF for k successes
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        n = self.n
        p = self.p

        fact_n = 1
        for i in range(1, n + 1):
            fact_n *= i

        fact_k = 1
        for i in range(1, k + 1):
            fact_k *= i

        fact_nk = 1
        for i in range(1, n - k + 1):
            fact_nk *= i

        comb = fact_n / (fact_k * fact_nk)

        return comb * (p ** k) * ((1 - p) ** (n - k))
