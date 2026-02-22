#!/usr/bin/env python3
"""
4-f1_score module
Defines a function that calculates the F1 score for each class.
"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class.

    Args:
        confusion (np.ndarray): confusion matrix of shape
            (classes, classes)

    Returns:
        np.ndarray: F1 score for each class of shape (classes,)
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)

    f1 = 2 * (prec * sens) / (prec + sens)

    return f1
