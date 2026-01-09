#!/usr/bin/env python3
"""Slices selected columns and rows from a DataFrame."""


def slice(df):
    """Extracts columns and selects every 60th row.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: The sliced DataFrame.
    """
    cols = ["High", "Low", "Close", "Volume_(BTC)"]
    return df[cols].iloc[::60]
