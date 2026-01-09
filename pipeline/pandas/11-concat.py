#!/usr/bin/env python3
"""Concatenates Bitstamp and Coinbase data with labeled keys."""


index = __import__('10-index').index


def concat(df1, df2):
    """Concatenates two DataFrames with timestamp indexing.

    Args:
        df1 (pandas.DataFrame): Coinbase DataFrame.
        df2 (pandas.DataFrame): Bitstamp DataFrame.

    Returns:
        pandas.DataFrame: Concatenated DataFrame.
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[:1417411920]

    pd = __import__('pandas')
    return pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"],
    )
