#!/usr/bin/env python3
"""Converts selected DataFrame values to a NumPy array."""


import pandas as pd


def array(df):
    """Selects the last 10 rows of High and Close columns as a NumPy array.

    Args:
        df (pandas.DataFrame): Input DataFrame containing High and Close columns.

    Returns:
        numpy.ndarray: Array of the selected values.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
