#!/usr/bin/env python3
"""Computes descriptive statistics for a DataFrame."""


import pandas as pd


def analyze(df):
    """Computes descriptive statistics excluding the Timestamp column.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing descriptive statistics.
    """
    return df.drop(columns=["Timestamp"]).describe()
