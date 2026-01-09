#!/usr/bin/env python3
"""Creates a hierarchical DataFrame indexed by timestamp first."""


import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """Concatenates two DataFrames with a hierarchical index.

    Args:
        df1 (pandas.DataFrame): Coinbase DataFrame.
        df2 (pandas.DataFrame): Bitstamp DataFrame.

    Returns:
        pandas.DataFrame: Hierarchically indexed DataFrame.
    """
    df1 = index(df1)
    df2 = index(df2)

    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    df = pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"],
    )

    df = df.swaplevel(0, 1)
    df = df.sort_index()

    return df
