#!/usr/bin/env python3
"""Loads data from a file into a pandas DataFrame."""


import pandas as pd


def from_file(filename, delimiter):
    """Loads data from a file as a pandas DataFrame.

    Args:
        filename (str): Path to the file to load.
        delimiter (str): Column separator.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(filename, delimiter=delimiter)
