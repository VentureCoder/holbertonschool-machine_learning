#!/usr/bin/env python3
"""Creates a pandas DataFrame from a NumPy array."""




def from_numpy(array):
    """Creates a DataFrame from a NumPy ndarray.

    Columns are labeled alphabetically in uppercase.

    Args:
        array (numpy.ndarray): The array to convert.

    Returns:
        pandas.DataFrame: The newly created DataFrame.
    """
    cols = [chr(i) for i in range(ord("A"), ord("A") + array.shape[1])]
    return pd.DataFrame(array, columns=cols)
