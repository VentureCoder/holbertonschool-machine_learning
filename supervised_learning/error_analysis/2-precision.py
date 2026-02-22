#!/usr/bin/env python3
"""
2-precision module
Defines a function that calculates the precision for each class.
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class.

    Args:
        confusion (np.ndarray): confusion matrix of shape
            (classes, classes)

    Returns:
        np.ndarray: precision for each class of shape (classes,)
    """
    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # Total predicted positives per class = sum of each column
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP) = TP / predicted positives
    precision_values = true_positives / predicted_positives

    return precision_values
