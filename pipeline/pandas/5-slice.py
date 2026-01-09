#!/usr/bin/env python3
"""Slices specific columns and rows from a pandas DataFrame."""


import pandas as pd


def slice(df):
    """Extracts selected columns and every 60th row from a DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: The sliced DataFrame.
    """
    cols = ["High", "Low", "Close", "Volume_(BTC)"]
    return df[cols].iloc[::60]
