#!/usr/bin/env python3
"""Plots a stacked bar graph of fruit quantities per person."""


import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plots a stacked bar chart."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    people = ["Farrah", "Fred", "Felicia"]
    x = np.arange(3)

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.bar(x, apples, width=0.5, color="red", label="apples")
    ax.bar(
        x,
        bananas,
        width=0.5,
        bottom=apples,
        color="yellow",
        label="bananas",
    )
    ax.bar(
        x,
        oranges,
        width=0.5,
        bottom=apples + bananas,
        color="#ff8000",
        label="oranges",
    )
    ax.bar(
        x,
        peaches,
        width=0.5,
        bottom=apples + bananas + oranges,
        color="#ffe5b4",
        label="peaches",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(people)
    ax.set_ylabel("Quantity of Fruit")
    ax.set_ylim(0, 80)
    ax.set_yticks(list(range(0, 81, 10)))
    ax.set_title("Number of Fruit per Person")
    ax.legend(loc="upper right")

    plt.show()
