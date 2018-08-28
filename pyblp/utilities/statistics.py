"""Standard statistical routines."""

import numpy as np

from .. import exceptions
from .basics import Groups
from .algebra import approximately_invert


class IV(object):
    """Simple model for generalized instrumental variables estimation."""

    def __init__(self, X, Z, W):
        """Store data and pre-compute covariances."""
        self.X = X
        self.Z = Z
        self.W = W

        # attempt to pre-compute covariances
        covariances_inverse = (self.X.T @ self.Z) @ self.W @ (self.Z.T @ self.X)
        self.covariances, replacement = approximately_invert(covariances_inverse)

        # store any errors
        self.errors = []
        if replacement:
            self.errors.append(exceptions.LinearParameterCovariancesInversionError(covariances_inverse, replacement))

    def estimate(self, y, compute_residuals=True):
        """Estimate parameters and optionally compute residuals."""
        parameters = self.covariances @ (self.X.T @ self.Z) @ self.W @ (self.Z.T @ y)
        if compute_residuals:
            return parameters, y - self.X @ parameters
        return parameters


def compute_gmm_se(u, Z, W, jacobian, covariance_type, clustering_ids):
    """Use an error term, instruments, a weighting matrix, and the Jacobian of the error term with respect to parameters
    to estimate GMM standard errors.
    """
    errors = []
    N = u.size

    # compute the Jacobian of the sample moments with respect to all parameters
    G = Z.T @ jacobian / N

    # attempt to compute the covariance matrix
    covariances_inverse = G.T @ W @ G
    covariances, replacement = approximately_invert(covariances_inverse)
    if replacement:
        errors.append(exceptions.GMMParameterCovariancesInversionError(covariances_inverse, replacement))

    # compute the robust covariance matrix
    if covariance_type != 'unadjusted':
        with np.errstate(invalid='ignore'):
            g = u * Z
            S = compute_gmm_moment_covariances(g, covariance_type, clustering_ids)
            covariances = covariances @ G.T @ W @ S @ W @ G @ covariances

    # compute standard errors and handle null values
    with np.errstate(invalid='ignore'):
        se = np.sqrt(np.c_[covariances.diagonal()] / N)
    if np.isnan(se).any():
        errors.append(exceptions.InvalidParameterCovariancesError())
    return se, errors


def compute_2sls_weights(Z):
    """Use instruments to compute a 2SLS weighting matrix."""
    errors = []

    # attempt to compute the weighting matrix
    S = Z.T @ Z
    W, replacement = approximately_invert(S)
    if replacement:
        errors.append(exceptions.GMMMomentCovariancesInversionError(S, replacement))
    return W, errors


def compute_gmm_weights(u, Z, center_moments, covariance_type, clustering_ids):
    """Use an error term and instruments to compute a GMM weighting matrix."""
    errors = []

    # compute moment conditions or their analogues for the given covariance type
    if u.size == 0:
        g = Z
    elif covariance_type == 'unadjusted':
        g = np.sqrt(compute_gmm_error_variance(u)) * Z
    else:
        g = u * Z
        if center_moments:
            g -= g.mean(axis=0)

    # attempt to compute the weighting matrix
    S = compute_gmm_moment_covariances(g, covariance_type, clustering_ids)
    W, replacement = approximately_invert(S)
    if replacement:
        errors.append(exceptions.GMMMomentCovariancesInversionError(S, replacement))

    # handle null values
    if np.isnan(W).any():
        errors.append(exceptions.InvalidMomentCovariancesError())
    return W, errors


def compute_gmm_error_variance(u):
    """Compute the variance of an error term."""
    return np.cov(u.flatten(), bias=True)


def compute_gmm_moment_covariances(g, covariance_type='unadjusted', clustering_ids=None):
    """Compute covariances between moment conditions."""
    N, M = g.shape
    if covariance_type != 'clustered':
        S = g.T @ g / N
    else:
        assert clustering_ids is not None
        S = np.zeros((M, M))
        for q_c in Groups(clustering_ids).sum(g):
            q_c = np.c_[q_c]
            S += q_c @ q_c.T / N
    return S
