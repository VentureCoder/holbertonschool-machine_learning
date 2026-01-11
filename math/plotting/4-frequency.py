#!/usr/bin/env python3
"""
4-frequency.py
Plots a histogram of student grades
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades where:
    - x-axis labeled Grades
    - y-axis labeled Number of Students
    - bins every 10 units
    - title is Project A
    - bars outlined in black
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xlim((0, 100))
    plt.xticks(np.arange(0, 101, 10))
    plt.ylim((0, 30))
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.show()
