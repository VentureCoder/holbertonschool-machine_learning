#!/usr/bin/env python3
"""Removes rows with NaN values in the Close column."""


def prune(df):
    """Removes entries where the Close value is NaN.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame without NaN Close values.
    """
    return df.dropna(subset=["Close"])
