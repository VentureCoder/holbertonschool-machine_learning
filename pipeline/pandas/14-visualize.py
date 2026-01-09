#!/usr/bin/env python3
"""Script to visualize the pd.DataFrame."""


import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


df = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',
    ',',
)

# Remove the Weighted_Price column
df = df.drop(columns=["Weighted_Price"])

# Rename Timestamp to Date
df = df.rename(columns={"Timestamp": "Date"})

# Convert timestamp values to datetime
df["Date"] = pd.to_datetime(df["Date"], unit="s")

# Set Date as index
df = df.set_index("Date")

# Fill missing Close values with previous row
df["Close"] = df["Close"].ffill()

# Fill missing High, Low, Open with same-row Close
df["High"] = df["High"].fillna(df["Close"])
df["Low"] = df["Low"].fillna(df["Close"])
df["Open"] = df["Open"].fillna(df["Close"])

# Fill missing Volume values with 0
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

# Keep data from 2017 onward
df = df.loc["2017-01-01":]

# Resample daily with required aggregations
df = df.resample("D").agg(
    {
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
    }
)

# Display the transformed DataFrame
print(df)

# Plot the data
df.plot()
plt.show()
