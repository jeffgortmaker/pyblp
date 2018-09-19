"""Standard statistical routines."""

from typing import List, Optional, Tuple, Union

import numpy as np

from .algebra import approximately_invert
from .basics import Array, Error, Groups
from .. import exceptions


class IV(object):
    """Simple model for generalized instrumental variables estimation."""

    X: Array
    Z: Array
    W: Array
    covariances: Array
    errors: List[Error]

    def __init__(self, X: Array, Z: Array, W: Array) -> None:
        """Store data and pre-compute covariances."""
        self.X = X
        self.Z = Z
        self.W = W

        # attempt to pre-compute covariances
        product = self.Z.T @ self.X
        covariances_inverse = product.T @ self.W @ product
        self.covariances, replacement = approximately_invert(covariances_inverse)

        # store any errors
        self.errors: List[Error] = []
        if replacement:
            self.errors.append(exceptions.LinearParameterCovariancesInversionError(covariances_inverse, replacement))

    def estimate(self, y: Array, residuals: bool = True) -> Union[Tuple[Array, Array], Array]:
        """Estimate parameters and optionally compute residuals."""
        parameters = self.covariances @ (self.X.T @ self.Z) @ self.W @ (self.Z.T @ y)
        if residuals:
            return parameters, y - self.X @ parameters
        return parameters


def compute_gmm_se(
        u: Array, Z: Array, W: Array, jacobian: Array, se_type: str, step: int,
        clustering_ids: Optional[Array] = None) -> Tuple[Array, List[Error]]:
    """Use an error term, instruments, a weighting matrix, and the Jacobian of the error term with respect to parameters
    to estimate GMM standard errors.
    """
    errors: List[Error] = []

    # if this is the first step, an unadjusted weighting matrix needs to be computed in order to properly scale
    #   unadjusted standard errors (other standard error types will be scaled properly)
    if se_type == 'unadjusted' and step == 1:
        W, W_errors = compute_gmm_weights(u, Z, 'unadjusted')
        errors.extend(W_errors)

    # compute the Jacobian of the sample moments with respect to all parameters
    N = u.size
    G = Z.T @ jacobian / N

    # attempt to compute the covariance matrix
    covariances_inverse = G.T @ W @ G
    covariances, replacement = approximately_invert(covariances_inverse)
    if replacement:
        errors.append(exceptions.GMMParameterCovariancesInversionError(covariances_inverse, replacement))

    # compute the robust covariance matrix
    if se_type != 'unadjusted':
        with np.errstate(invalid='ignore'):
            g = u * Z
            S = compute_gmm_moment_covariances(g, se_type, clustering_ids)
            covariances = covariances @ G.T @ W @ S @ W @ G @ covariances

    # compute standard errors and handle null values
    with np.errstate(invalid='ignore'):
        se = np.sqrt(np.c_[covariances.diagonal()] / N)
    if np.isnan(se).any():
        errors.append(exceptions.InvalidParameterCovariancesError())
    return se, errors


def compute_2sls_weights(Z: Array) -> Tuple[Array, List[Error]]:
    """Use instruments to compute a 2SLS weighting matrix."""
    errors: List[Error] = []
    S = Z.T @ Z
    W, replacement = approximately_invert(S)
    if replacement:
        errors.append(exceptions.GMMMomentCovariancesInversionError(S, replacement))
    return W, errors


def compute_gmm_weights(
        u: Array, Z: Array, W_type: str, center_moments: bool = True,
        clustering_ids: Optional[Array] = None) -> Tuple[Array, List[Error]]:
    """Use an error term and instruments to compute a GMM weighting matrix."""
    errors: List[Error] = []

    # compute moment conditions or their analogues for the given covariance type
    if u.size == 0:
        g = Z
    elif W_type == 'unadjusted':
        g = np.sqrt(compute_gmm_error_variance(u)) * Z
    else:
        g = u * Z
        if center_moments:
            g -= g.mean(axis=0)

    # attempt to compute the weighting matrix
    S = compute_gmm_moment_covariances(g, W_type, clustering_ids)
    W, replacement = approximately_invert(S)
    if replacement:
        errors.append(exceptions.GMMMomentCovariancesInversionError(S, replacement))

    # handle null values
    if np.isnan(W).any():
        errors.append(exceptions.InvalidMomentCovariancesError())
    return W, errors


def compute_gmm_error_variance(u: Array) -> Array:
    """Compute the variance of an error term."""
    return np.cov(u.flatten(), bias=True)


def compute_gmm_moment_covariances(g: Array, covariance_type: str, clustering_ids: Optional[Array] = None) -> Array:
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
