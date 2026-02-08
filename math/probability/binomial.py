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

            p = 1 - (var / mean)
            n = round(mean / p)
            p = mean / n

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """
        Calculates the PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        # factorial n
        fact_n = 1
        for i in range(1, self.n + 1):
            fact_n *= i

        # factorial k
        fact_k = 1
        for i in range(1, k + 1):
            fact_k *= i

        # factorial (n - k)
        fact_nk = 1
        for i in range(1, self.n - k + 1):
            fact_nk *= i

        comb = fact_n / (fact_k * fact_nk)

        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))
