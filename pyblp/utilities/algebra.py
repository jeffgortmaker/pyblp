"""Algebraic routines."""

import warnings

import numpy as np
import scipy.linalg


def multiply_tensor_and_matrix(a, b):
    """Multiply a 3D tensor with a 2D matrix in a loop to exploit speed gains from optimized 2D multiplication."""
    (n1, n2, _), (_, n3) = a.shape, b.shape
    multiplied = np.zeros((n1, n2, n3), a.dtype)
    for i in range(n1):
        multiplied[i] = a[i] @ b
    return multiplied


def multiply_matrix_and_tensor(a, b):
    """Multiply a 2D matrix with a 3D tensor in a loop to exploit speed gains from optimized 2D multiplication."""
    (n2, _), (n1, _, n3) = a.shape, b.shape
    multiplied = np.zeros((n1, n2, n3), a.dtype)
    for i in range(n1):
        multiplied[i] = a @ b[i]
    return multiplied


def precisely_solve(a, b):
    """Attempt to precisely solve a system of equations."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            solved = scipy.linalg.solve(a, b) if b.size > 0 else b
            successful = True
    except (ValueError, scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
        solved = np.full_like(b, np.nan)
        successful = False
    return solved, successful


def precisely_invert(x):
    """Attempt to precisely invert a matrix."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            inverted = scipy.linalg.inv(x) if x.size > 0 else x
            successful = True
    except (ValueError, scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
        inverted = np.full_like(x, np.nan)
        successful = False
    return inverted, successful


def approximately_solve(a, b):
    """Attempt to solve a system of equations with decreasingly precise replacements for the inverse."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            solved = scipy.linalg.solve(a, b) if b.size > 0 else b
            replacement = None
    except:
        inverse, replacement = approximately_invert(a)
        solved = inverse @ b
    return solved, replacement


def approximately_invert(x):
    """Attempt to invert a matrix with decreasingly precise replacements for the inverse."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            inverted = scipy.linalg.inv(x) if x.size > 0 else x
            replacement = None
    except ValueError:
        inverted = np.full_like(x, np.nan)
        replacement = "null values"
    except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
        try:
            inverted = scipy.linalg.pinv(x)
            replacement = "its Moore-Penrose pseudo inverse"
        except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            inverted = np.diag(1 / np.diag(x))
            replacement = "inverted diagonal terms because the Moore-Penrose pseudo inverse could not be computed"
    return inverted, replacement
