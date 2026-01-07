#!/usr/bin/env python3
"""Renames and formats timestamp data in a pandas DataFrame."""


import pandas as pd


def rename(df):
    """Renames the Timestamp column to Datetime and formats the DataFrame.

    The function converts timestamp values to datetime values and
    displays only the Datetime and Close columns.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a Timestamp column.

    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
