#!/usr/bin/env python3
"""Sorts a DataFrame by High price in descending order."""


def high(df):
    """Sorts the DataFrame by the High column in descending order.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Sorted DataFrame.
    """
    return df.sort_values(by="High", ascending=False)
