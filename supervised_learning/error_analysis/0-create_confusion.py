#!/usr/bin/env python3
"""
0-create_confusion module
Defines a function that creates a confusion matrix.
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels (np.ndarray): one-hot encoded true labels
            of shape (m, classes)
        logits (np.ndarray): one-hot encoded predicted labels
            of shape (m, classes)

    Returns:
        np.ndarray: confusion matrix of shape (classes, classes)
    """
    classes = labels.shape[1]

    confusion = np.zeros((classes, classes))

    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1

    return confusion
