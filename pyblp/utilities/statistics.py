"""Standard statistical routines."""

import numpy as np
import scipy.linalg

from .. import exceptions


class IV(object):
    """Simple model for generalized instrumental variables estimation."""

    def __init__(self, X, Z, W):
        """Pre-compute projection and covariance matrices, which will be used during estimation. Store any errors."""
        self.errors = set()
        self.X = X
        self.projection = Z @ W @ Z.T
        self.covariances, approximation = invert(X.T @ self.projection @ X)
        if approximation:
            self.errors.add(lambda: exceptions.LinearParameterCovariancesInversionError(approximation))

    def compute_parameters(self, Y):
        """Estimate parameters."""
        return self.covariances @ self.X.T @ self.projection @ Y

    def compute_residuals(self, Y, parameters):
        """Estimate residuals."""
        return Y - self.X @ parameters


def compute_gmm_se(u, Z, W, jacobian, se_type):
    """Use an error term, instruments, a weighting matrix, and the Jacobian of the error term with respect to parameters
    to estimate GMM standard errors. Return a set of any errors.
    """
    errors = set()

    # compute the Jacobian of the sample moments with respect to all parameters
    G = Z.T @ jacobian

    # attempt to compute the covariance matrix
    covariances, approximation = invert(G.T @ W @ G)
    if approximation:
        errors.add(lambda: exceptions.GMMParameterCovariancesInversionError(approximation))

    # compute the robust covariance matrix and extract standard errors
    covariances, approximation = invert(G.T @ W @ G)
    with np.errstate(invalid='ignore'):
        if se_type == 'robust':
            covariances = covariances @ G.T @ W @ Z.T @ (np.diagflat(u) ** 2) @ Z @ W @ G @ covariances
        se = np.sqrt(np.c_[covariances.diagonal()])

    # handle null values
    if np.isnan(se).any():
        errors.add(exceptions.InvalidCovariancesError)
    return se, errors


def compute_gmm_weights(u, Z, center_moments):
    """Use an error term and instruments to compute a GMM weighting matrix. Return a set of any errors."""
    errors = set()

    # compute and center the sample moments
    g = u * Z
    if center_moments:
        g -= g.mean(axis=0)

    # attempt to compute the weighting matrix
    W, approximation = invert(g.T @ g)
    if approximation:
        errors.add(lambda: exceptions.GMMMomentCovariancesInversionError(approximation))

    # handle null values
    if np.isnan(W).any():
        errors.add(exceptions.InvalidWeightsError)
    return W, errors


def invert(matrix):
    """Attempt to invert a matrix with decreasingly precise inversion functions. The first attempt is with a
    standard inversion function; the second, with the Moore-Penrose pseudo inverse; the third, with simple diagonal
    inversion. Along with the inverted matrix, return a description of any approximation used.
    """
    try:
        return scipy.linalg.inv(matrix), None
    except ValueError:
        return np.full_like(matrix, np.nan), None
    except scipy.linalg.LinAlgError:
        try:
            return scipy.linalg.pinv(matrix), "computing the Moore-Penrose pseudo inverse"
        except scipy.linalg.LinAlgError:
            return (
                np.diag(1 / np.diag(matrix)),
                "inverting only the variance terms, since the Moore-Penrose pseudo inverse could not be computed"
            )
