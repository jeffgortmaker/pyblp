"""Standard statistical routines."""

from typing import List, Tuple

import numpy as np
import scipy.linalg

from .algebra import approximately_invert, duplication_matrix, elimination_matrix
from .basics import Array, Error, Groups
from .. import exceptions


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
            self, X_list: List[Array], Z_list: List[Array], W: Array, y_list: List[Array], jacobian_list: List[Array],
            convert_jacobians: bool) -> Tuple[List[Array], List[Array], List[Array]]:
        """Estimate parameters and compute residuals. Optionally convert Jacobians of y into Jacobians of residuals."""

        # stack matrices
        X = scipy.linalg.block_diag(*X_list)
        Z = scipy.linalg.block_diag(*Z_list)
        y = np.vstack(y_list)

        # estimate the model
        XZ = X.T @ Z
        parameters = self.covariances @ XZ @ W @ (Z.T @ y)
        residuals = y - X @ parameters

        # split the parameters and residuals into lists
        parameters_list = np.split(parameters, [x.shape[1] for x in X_list[:-1]], axis=0)
        residuals_list = np.split(residuals, len(X_list), axis=0)

        # optionally convert Jacobians
        if convert_jacobians:
            jacobian = (np.eye(y.size) - X @ self.covariances @ XZ @ W @ Z.T) @ np.vstack(jacobian_list)
            jacobian_list = np.split(jacobian, len(jacobian_list), axis=0)

        return parameters_list, residuals_list, jacobian_list


def compute_gmm_weights(S: Array) -> Tuple[Array, List[Error]]:
    """Compute a GMM weighting matrix."""
    errors: List[Error] = []

    # invert the matrix and handle any errors
    W, replacement = approximately_invert(S)
    if replacement:
        errors.append(exceptions.GMMMomentCovariancesInversionError(S, replacement))
    if np.isnan(W).any():
        errors.append(exceptions.InvalidMomentCovariancesError())

    # enforce shape and symmetry
    return np.c_[W + W.T] / 2, errors


def compute_gmm_moment_covariances(
        u_list: List[Array], Z_list: List[Array], covariance_type: str, clustering_ids: Array,
        center_moments: bool) -> Array:
    """Compute covariances between moments."""

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
        S = g.T @ g / N

    # enforce shape and symmetry
    return np.c_[S + S.T] / 2


def compute_gmm_parameter_covariances(W: Array, S: Array, mean_G: Array, se_type: str) -> Tuple[Array, List[Error]]:
    """Estimate GMM parameter covariances."""
    errors: List[Error] = []

    # attempt to compute the covariance matrix
    covariances_inverse = mean_G.T @ W @ mean_G
    covariances, replacement = approximately_invert(covariances_inverse)
    if replacement:
        errors.append(exceptions.GMMParameterCovariancesInversionError(covariances_inverse, replacement))

    # compute the robust covariance matrix
    if se_type != 'unadjusted':
        with np.errstate(invalid='ignore'):
            covariances = covariances @ mean_G.T @ W @ S @ W @ mean_G @ covariances

    # enforce shape and symmetry
    return np.c_[covariances + covariances.T] / 2, errors


def compute_gmm_error_covariance(u1: Array, u2: Array) -> Array:
    """Compute the covariance between two error terms."""
    return np.cov(u1.flatten(), u2.flatten(), bias=True)[0][1]


def compute_gmm_moments(u_list: List[Array], Z_list: List[Array]) -> Array:
    """Compute GMM moments."""
    return np.hstack([u * Z for u, Z in zip(u_list, Z_list)])


def compute_gmm_moments_mean(u_list: List[Array], Z_list: List[Array]) -> Array:
    """Compute GMM moments, averaged across observations."""
    return np.c_[compute_gmm_moments(u_list, Z_list).mean(axis=0)]


def compute_gmm_moments_jacobian_mean(jacobian_list: List[Array], Z_list: List[Array]) -> Array:
    """Compute the Jacobian of GMM moments with respect to parameters, averaged across observations."""
    return np.concatenate([j[:, None] * Z[:, :, None] for j, Z in zip(jacobian_list, Z_list)], axis=1).mean(axis=0)


def compute_sigma_squared_vector_covariances(sigma: Array, sigma_vector_covariances: Array) -> Array:
    """Use the delta method to transform the asymptotic covariance matrix of vech(sigma) into the asymptotic covariance
    matrix of vech(sigma * sigma') where sigma is a lower triangular Cholesky root of parameters. See Section 10.5.4 in
    the Handbook of Matrices (Lutkepohl and Lutkepohl).
    """
    if sigma_vector_covariances.size == 0:
        return sigma_vector_covariances

    k = sigma.shape[0]
    L = elimination_matrix(k)
    D = duplication_matrix(k)
    I = np.eye(k)
    jacobian = 2 * scipy.linalg.pinv(D) @ np.kron(sigma, I) @ L.T
    return jacobian @ sigma_vector_covariances @ jacobian.T
