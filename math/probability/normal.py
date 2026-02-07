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
