"""Algebraic routines."""

import itertools
from typing import Callable, List, Optional, Tuple
import warnings

import numpy as np
import scipy.linalg

from .basics import Array
from .. import options


def compute_condition_number(x: Array) -> float:
    """Compute the condition number of a square matrix."""
    if x.size == 0:
        return 0
    if not np.isfinite(x).all():
        return np.nan
    try:
        return np.linalg.cond(x.astype(np.float64))
    except scipy.linalg.LinAlgError:
        return np.nan


def precisely_identify_singularity(x: Array) -> Tuple[Array, bool, Array]:
    """Compute the condition number of a matrix to identify whether it is nearly singular."""
    singular = False
    successful = True
    condition = np.nan
    if options.singular_tol < np.inf:
        condition = compute_condition_number(x)
        successful = not np.isnan(condition)
        singular = successful and condition > options.singular_tol

    return singular, successful, condition


def precisely_identify_collinearity(x: Array) -> Tuple[Array, bool]:
    """Compute the QR decomposition of a matrix and identify which diagonal elements of the upper diagonal matrix are
    within absolute and relative tolerances.
    """
    collinear = np.zeros(x.shape[1], np.bool_)
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
    inverted = np.full_like(x, np.nan)
    replacement = None
    if x.size > 0:
        # collect the different inversion methods
        methods: List[Tuple[Callable, Optional[str]]] = []
        if options.pseudo_inverses:
            methods.append((scipy.linalg.pinv, None))
        else:
            methods.extend([(scipy.linalg.inv, None), (scipy.linalg.pinv, "its Moore-Penrose pseudo inverse")])
        methods.append((
            lambda y: np.diag(1 / y.diagonal()),
            "inverted diagonal terms because the Moore-Penrose pseudo-inverse could not be computed"
        ))

        # use the different methods to invert the matrix
        for invert, replacement in methods:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    inverted = invert(x)
                replacement = None
                break
            except ValueError:
                replacement = "null values"
                break
            except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
                pass

    return inverted, replacement


def duplication_matrix(n: int) -> Array:
    """Construct the unique matrix D, which for any n x n symmetric matrix A satisfies D * vech(A) = vec(A). See
    Definition 3.2a in Magnus and Neudecker (1980).
    """
    L = elimination_matrix(n)
    K = commutation_matrix(n)
    D = L.T + K @ L.T - L.T @ L @ K @ L.T
    return D


def elimination_matrix(n: int) -> Array:
    """Construct the unique matrix L, which for any n x n symmetric matrix A satisfies L * vec(A) = vech(A). See
    Definition 3.1b in Magnus and Neudecker (1980).
    """
    L = np.zeros((n * (n + 1) // 2, n**2), np.int64)

    for i in range(n):
        for j in range(i + 1):
            u = np.zeros((n * (n + 1) // 2, 1), np.int64)
            u[j * n + i - j * (j + 1) // 2] = 1

            E = np.zeros((n, n), np.int64)
            E[i, j] = 1

            L += u @ vec(E)[None, :]

    return L


def commutation_matrix(n: int) -> Array:
    """Construct the unique matrix K, which for any n x n symmetric matrix A satisfies K * vec(A) = vec(A'). See
    Definition 2.1b in Magnus and Neudecker (1980).
    """
    K = np.zeros((n**2, n**2), np.int64)

    for i, j in itertools.product(range(n), repeat=2):
        E = np.zeros((n, n), np.int64)
        E[i, j] = 1

        K += np.kron(E, E.T)

    return K


def vec(x: Array) -> Array:
    """Ravel a matrix A in Fortran order to construct vec(A)."""
    return np.ravel(x, order='F')


def vech(x: Array) -> Array:
    """Ravel the lower triangle of a square matrix A in Fortran order to construct vech(A)."""
    return x.T[np.triu_indices_from(x)]


def vech_to_lower(x: Array, n: int) -> Array:
    """Convert vech(A) into the lower triangular n x n matrix A."""
    A = np.zeros((n, n), dtype=x.dtype)
    A[np.triu_indices(n)] = x.flat
    return A.T


def vech_to_full(x: Array, n: int) -> Array:
    """Convert vech(A) into the full, symmetric n x n matrix A."""
    A = vech_to_lower(x, n)
    A[np.triu_indices(n, k=1)] = A.T[np.triu_indices(n, k=1)]
    return A
