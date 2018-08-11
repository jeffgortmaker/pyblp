"""Standard statistical routines."""

import warnings
import functools

import numpy as np
import scipy.linalg

from .. import exceptions


class IV(object):
    """Simple model for generalized instrumental variables estimation."""

    def __init__(self, X, Z, W):
        self.errors = set()
        self.X = X
        self.Z = Z
        self.W = W
        self.covariances, approximation = approximate_invert((self.X.T @ self.Z) @ self.W @ (self.Z.T @ self.X))
        if approximation:
            self.errors.add(functools.partial(exceptions.LinearParameterCovariancesInversionError, approximation))

    def estimate(self, y, compute_residuals=True):
        """Estimate parameters and compute residuals."""
        parameters = self.covariances @ (self.X.T @ self.Z) @ self.W @ (self.Z.T @ y)
        return (parameters, y - self.X @ parameters) if compute_residuals else parameters


def compute_gmm_se(u, Z, W, jacobian, se_type, clustering_ids):
    """Use an error term, instruments, a weighting matrix, and the Jacobian of the error term with respect to parameters
    to estimate GMM standard errors. Return a set of any errors.
    """
    errors = set()

    # compute the Jacobian of the sample moments with respect to all parameters
    G = Z.T @ jacobian

    # attempt to compute the covariance matrix
    covariances, approximation = approximate_invert(G.T @ W @ G)
    if approximation:
        errors.add(functools.partial(exceptions.GMMParameterCovariancesInversionError, approximation))

    # compute the robust covariance matrix and extract standard errors
    covariances, approximation = approximate_invert(G.T @ W @ G)
    with np.errstate(invalid='ignore'):
        if se_type != 'unadjusted':
            g = u * Z
            S = compute_gmm_moment_covariances(g, se_type, clustering_ids)
            covariances = covariances @ G.T @ W @ S @ W @ G @ covariances
        se = np.sqrt(np.c_[covariances.diagonal()])

    # handle null values
    if np.isnan(se).any():
        errors.add(exceptions.InvalidCovariancesError)
    return se, errors


def compute_2sls_weights(Z):
    """Use instruments to compute a 2SLS weighting matrix."""
    errors = set()

    # attempt to compute the weighting matrix
    W, approximation = approximate_invert(Z.T @ Z)
    if approximation:
        errors.add(functools.partial(exceptions.GMMMomentCovariancesInversionError, approximation))
    return W, errors


def compute_gmm_weights(u, Z, center_moments, se_type, clustering_ids):
    """Use an error term and instruments to compute a GMM weighting matrix. Return a set of any errors."""
    errors = set()

    # compute and center the sample moments
    g = u * Z
    if center_moments:
        g -= g.mean(axis=0)

    # attempt to compute the weighting matrix
    W, approximation = approximate_invert(compute_gmm_moment_covariances(g, se_type, clustering_ids))
    if approximation:
        errors.add(functools.partial(exceptions.GMMMomentCovariancesInversionError, approximation))

    # handle null values
    if np.isnan(W).any():
        errors.add(exceptions.InvalidWeightsError)
    return W, errors


def compute_gmm_moment_covariances(g, se_type, clustering_ids):
    """Compute covariances between moment conditions."""
    if se_type == 'clustered' and clustering_ids.shape[1] > 0:
        return sum(g[clustering_ids.flat == c].T @ g[clustering_ids.flat == c] for c in np.unique(clustering_ids))
    return g.T @ g


def solve(a, b):
    """Solve a system of equations and return a solution of nans if anything goes wrong."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return scipy.linalg.solve(a, b)
        except (ValueError, scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            return np.full_like(b, np.nan)


def invert(x):
    """Invert a matrix and return a solution of nans if anything goes wrong."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return scipy.linalg.inv(x)
        except (ValueError, scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            return np.full_like(x, np.nan)


def approximate_solve(a, b):
    """Attempt to solve a system of equations with decreasingly precise inversion functions. Along with the solution,
    return a description of any approximation that was used.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return scipy.linalg.solve(a, b), None
        except ValueError:
            return np.full_like(b, np.nan), None
        except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            inverse, approximation = invert(a)
            return inverse @ b, approximation


def approximate_invert(x):
    """Attempt to invert a matrix with decreasingly precise inversion functions. Along with the inverted matrix, return
    a description of any approximation that was used.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return scipy.linalg.inv(x), None
        except ValueError:
            return np.full_like(x, np.nan), None
        except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            try:
                return scipy.linalg.pinv(x), "computing the Moore-Penrose pseudo inverse"
            except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
                return (
                    np.diag(1 / np.diag(x)),
                    "inverting only the variance terms, since the Moore-Penrose pseudo inverse could not be computed"
                )
