#!/usr/bin/env python3
"""Fills missing values in a DataFrame according to given rules."""


def fill(df):
    """Cleans and fills missing values in the DataFrame.

    - Drops the Weighted_Price column
    - Fills missing Close values using forward fill
    - Fills missing Open, High, Low using Close from the same row
    - Sets missing Volume columns to 0

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].ffill()

    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
