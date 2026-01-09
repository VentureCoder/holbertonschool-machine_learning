#!/usr/bin/env python3
"""Sorts a DataFrame in reverse order and transposes it."""


def flip_switch(df):
    """Reverses row order and transposes the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: The transformed DataFrame.
    """
    return df.iloc[::-1].T
