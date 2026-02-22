#!/usr/bin/env python3
"""
1-sensitivity module
Defines a function that calculates the sensitivity for each class.
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall) for each class.

    Args:
        confusion (np.ndarray): confusion matrix of shape
            (classes, classes)

    Returns:
        np.ndarray: sensitivity for each class of shape (classes,)
    """
    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # Total actual positives per class = sum of each row
    actual_positives = np.sum(confusion, axis=1)

    # Sensitivity = TP / (TP + FN) = TP / actual positives
    sensitivity_values = true_positives / actual_positives

    return sensitivity_values
