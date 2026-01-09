#!/usr/bin/env python3
"""Sets the Timestamp column as the DataFrame index."""


def index(df):
    """Sets the Timestamp column as the index.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame indexed by Timestamp.
    """
    return df.set_index("Timestamp")
