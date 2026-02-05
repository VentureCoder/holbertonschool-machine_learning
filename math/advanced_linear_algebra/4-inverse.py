#!/usr/bin/env python3
"""
4-inverse module
Defines a function that calculates the inverse of a matrix.
"""


def determinant(matrix):
    """Helper function to calculate determinant."""
    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        sub = [row[:col] + row[col+1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub)

    return det


def minor(matrix):
    """Helper function to calculate minor matrix."""
    n = len(matrix)

    if n == 1:
        return [[1]]

    minors = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            sub = [r[:j] + r[j+1:] for k, r in enumerate(matrix) if k != i]
            row_minors.append(determinant(sub))
        minors.append(row_minors)

    return minors


def cofactor(matrix):
    """Helper function to calculate cofactor matrix."""
    minors = minor(matrix)

    cofactors = []
    for i in range(len(minors)):
        row = []
        for j in range(len(minors)):
            row.append(((-1) ** (i + j)) * minors[i][j])
        cofactors.append(row)

    return cofactors


def adjugate(matrix):
    """Helper function to calculate adjugate matrix."""
    cof = cofactor(matrix)
    n = len(cof)

    adj = []
    for j in range(n):
        row = []
        for i in range(n):
            row.append(cof[i][j])
        adj.append(row)

    return adj


def inverse(matrix):
    """Calculates the inverse of a matrix."""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)

    inv = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(adj[i][j] / det)
        inv.append(row)

    return inv
