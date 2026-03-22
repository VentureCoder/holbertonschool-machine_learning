#!/usr/bin/env python3
"""Comment of Function"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Early Stopping"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count == patience, count
