"""Standard statistical routines."""

from typing import List, Tuple

import numpy as np
import scipy.linalg

from .algebra import approximately_invert
from .basics import Array, Error, Groups
from .. import exceptions, options


class IV(object):
    """Simple model for generalized instrumental variables estimation."""

    covariances: Array
    errors: List[Error]

    def __init__(self, X_list: List[Array], Z_list: List[Array], W: Array) -> None:
        """Pre-compute covariances."""

        # stack matrices
        X = scipy.linalg.block_diag(*X_list)
        Z = scipy.linalg.block_diag(*Z_list)

        # attempt to pre-compute covariances
        product = Z.T @ X
        covariances_inverse = product.T @ W @ product
        self.covariances, replacement = approximately_invert(covariances_inverse)

        # store any errors
        self.errors: List[Error] = []
        if replacement:
            self.errors.append(exceptions.LinearParameterCovariancesInversionError(covariances_inverse, replacement))

    def estimate(
            self, X_list: List[Array], Z_list: List[Array], W: Array, y_list: List[Array]) -> (
            Tuple[List[Array], List[Array]]):
        """Estimate parameters and compute residuals."""

        # stack matrices
        X = scipy.linalg.block_diag(*X_list)
        Z = scipy.linalg.block_diag(*Z_list)
        y = np.vstack(y_list)

        # estimate the model
        parameters = self.covariances @ (X.T @ Z) @ W @ (Z.T @ y)
        residuals = y - X @ parameters

        # split the parameters and residuals into lists
        parameters_list = np.split(parameters, [x.shape[1] for x in X_list[:-1]], axis=0)
        residuals_list = np.split(residuals, len(X_list), axis=0)
        return parameters_list, residuals_list


def compute_2sls_weights(Z_list: List[Array]) -> Tuple[Array, List[Error]]:
    """Use instruments to compute a 2SLS weighting matrix."""
    errors: List[Error] = []
    Z = scipy.linalg.block_diag(*Z_list)
    S = Z.T @ Z
    W, replacement = approximately_invert(S)
    if replacement:
        errors.append(exceptions.GMMMomentCovariancesInversionError(S, replacement))
    return W, errors


def compute_gmm_weights(
        u_list: List[Array], Z_list: List[Array], W_type: str, clustering_ids: Array, center_moments: bool) -> (
        Tuple[Array, List[Error]]):
    """Compute a GMM weighting matrix."""
    errors: List[Error] = []
    S = compute_gmm_moment_covariances(u_list, Z_list, W_type, clustering_ids, center_moments)
    W, replacement = approximately_invert(S)
    if replacement:
        errors.append(exceptions.GMMMomentCovariancesInversionError(S, replacement))
    if np.isnan(W).any():
        errors.append(exceptions.InvalidMomentCovariancesError())
    return W, errors


def compute_gmm_moment_covariances(
        u_list: List[Array], Z_list: List[Array], covariance_type: str, clustering_ids: Array,
        center_moments: bool = False) -> Array:
    """Compute covariances between moment conditions."""

    # count dimensions
    N = u_list[0].shape[0]

    # compute the moment covariances
    if covariance_type == 'unadjusted':
        pairs = list(zip(u_list, Z_list))
        S = np.block([[compute_gmm_error_covariance(u1, u2) * (Z1.T @ Z2) for u2, Z2 in pairs] for u1, Z1 in pairs]) / N
    else:
        g = compute_gmm_moments(u_list, Z_list)
        if center_moments:
            g -= g.mean(axis=0)
        if covariance_type == 'clustered':
            g = Groups(clustering_ids).sum(g)
        S = sum(np.c_[g_n] @ np.c_[g_n].T for g_n in g) / N

    # enforce shape and symmetry
    return np.c_[(S + S.T) / 2]


def compute_gmm_parameter_covariances(
        jacobian_list: List[Array], u_list: List[Array], Z_list: List[Array], W: Array, se_type: str,
        clustering_ids: Array, update_W: bool) -> Tuple[Array, List[Error]]:
    """Estimate GMM parameter covariances."""
    errors: List[Error] = []

    # optionally update the weighting matrix
    if update_W:
        W, W_errors = compute_gmm_weights(u_list, Z_list, se_type, clustering_ids, center_moments=False)
        errors.extend(W_errors)

    # attempt to compute the covariance matrix
    G_bar = compute_gmm_moments_jacobian_mean(jacobian_list, Z_list)
    covariances_inverse = G_bar.T @ W @ G_bar
    covariances, replacement = approximately_invert(covariances_inverse)
    if replacement:
        errors.append(exceptions.GMMParameterCovariancesInversionError(covariances_inverse, replacement))

    # compute the robust covariance matrix
    with np.errstate(invalid='ignore'):
        S = compute_gmm_moment_covariances(u_list, Z_list, se_type, clustering_ids)
        covariances = covariances @ G_bar.T @ W @ S @ W @ G_bar @ covariances

    # enforce shape and symmetry
    return np.c_[(covariances + covariances.T) / 2], errors


def compute_gmm_error_covariance(u1: Array, u2: Array) -> Array:
    """Compute the covariance between two error terms."""
    return np.cov(u1.flatten(), u2.flatten(), bias=True)[0][1]


def compute_gmm_moments(u_list: List[Array], Z_list: List[Array]) -> Array:
    """Compute GMM moment conditions."""

    # count dimensions
    N = u_list[0].shape[0]
    M = sum(Z.shape[1] for Z in Z_list)

    # compute the moments
    g = np.zeros((N, M), options.dtype)
    for n in range(N):
        Z_n = scipy.linalg.block_diag(*(Z[n] for Z in Z_list))
        u_n = np.vstack(u[n] for u in u_list)
        g[n] = (Z_n.T @ u_n).flat
    return g


def compute_gmm_moments_mean(u_list: List[Array], Z_list: List[Array]) -> Array:
    """Compute GMM moment conditions, averaged across observations."""

    # count dimensions
    N = u_list[0].shape[0]
    M = sum(Z.shape[1] for Z in Z_list)

    # compute the moment means
    g_bar = np.zeros((M, 1))
    for n in range(N):
        Z_n = scipy.linalg.block_diag(*(Z[n] for Z in Z_list))
        u_n = np.vstack(u[n] for u in u_list)
        g_bar += Z_n.T @ u_n / N
    return g_bar


def compute_gmm_moments_jacobian_mean(jacobian_list: List[Array], Z_list: List[Array]) -> Array:
    """Compute the Jacobian of GMM moment conditions with respect to parameters, averaged across observations."""

    # count dimensions
    N, P = jacobian_list[0].shape
    M = sum(Z.shape[1] for Z in Z_list)

    # compute the Jacobian means
    G_bar = np.zeros((M, P), options.dtype)
    for n in range(N):
        Z_n = scipy.linalg.block_diag(*(Z[n] for Z in Z_list))
        jacobian_n = np.vstack(j[n] for j in jacobian_list)
        G_bar += Z_n.T @ jacobian_n / N
    return G_bar
