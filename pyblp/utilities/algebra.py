"""Algebraic routines."""

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.linalg

from .basics import Array
from .. import options


def precisely_identify_collinearity(x: Array) -> Tuple[Array, bool]:
    """Compute the QR decomposition of a matrix and identify which diagonal elements of the upper diagonal matrix are
    within absolute and relative tolerances.
    """
    collinear = np.zeros(x.shape[1], np.bool)
    successful = True
    if x.size > 0 and min(options.collinear_atol, options.collinear_rtol) > 0:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                r = scipy.linalg.qr(x, mode='r')[0]
                collinear = np.abs(r.diagonal()) < options.collinear_atol + options.collinear_rtol * x.std(axis=0)
        except (ValueError, scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            successful = False
    return collinear, successful


def precisely_identify_psd(x: Array) -> Tuple[bool, bool]:
    """Compute the SVD of a matrix and use it to identify whether the matrix is PSD with absolute and relative
    tolerances.
    """
    psd = successful = True
    if x.size > 0 and np.isfinite([options.psd_atol, options.psd_rtol]).any():
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                _, s, v = scipy.linalg.svd(x)
                psd = np.allclose((v.T * s) @ v, x, atol=options.psd_atol, rtol=options.psd_rtol)
        except (ValueError, scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            psd = successful = False
    return psd, successful


def precisely_compute_eigenvalues(x: Array) -> Tuple[Array, bool]:
    """Compute the eigenvalues of a real symmetric matrix."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            eigenvalues = scipy.linalg.eigvalsh(x) if x.size > 0 else x.flatten()
            successful = True
    except (ValueError, scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
        eigenvalues = np.full_like(np.diag(x), np.nan)
        successful = False
    return eigenvalues, successful


def precisely_solve(a: Array, b: Array) -> Tuple[Array, bool]:
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


def precisely_invert(x: Array) -> Tuple[Array, bool]:
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


def approximately_solve(a: Array, b: Array) -> Tuple[Array, Optional[str]]:
    """Attempt to solve a system of equations with decreasingly precise replacements for the inverse."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            solved = scipy.linalg.solve(a, b) if b.size > 0 else b
            replacement = None
    except Exception:
        inverse, replacement = approximately_invert(a)
        solved = inverse @ b
    return solved, replacement


def approximately_invert(x: Array) -> Tuple[Array, Optional[str]]:
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
