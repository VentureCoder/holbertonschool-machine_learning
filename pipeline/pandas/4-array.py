#!/usr/bin/env python3
"""Converts selected DataFrame values to a NumPy array."""


def array(df):
    """Returns last 10 rows of High and Close columns as a NumPy array.

    Args:
        df (pandas.DataFrame): Input DataFrame with High and Close columns.

    Returns:
        numpy.ndarray: Array of selected values.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
