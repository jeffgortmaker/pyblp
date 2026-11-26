"""Standard statistical routines."""

import itertools
from typing import List, Tuple

import numpy as np
import scipy.linalg

from .algebra import approximately_invert, duplication_matrix, elimination_matrix, precisely_identify_psd
from .basics import Array, Error, Groups, interact_ids
from .. import exceptions


class IV(object):
    """Simple model for generalized instrumental variables estimation."""

    covariances: Array
    errors: List[Error]

    def __init__(self, X_list: List[Array], Z_list: List[Array], W: Array) -> None:
        """Pre-compute covariances."""

        # stack matrices
        X = np.vstack(X_list)
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
        self,
        X_list: List[Array],
        Z_list: List[Array],
        W: Array,
        y_list: List[Array],
        jacobian_list: List[Array],
        convert_jacobians: bool,
    ) -> Tuple[Array, List[Array], List[Array]]:
        """Estimate parameters and compute residuals. Optionally convert Jacobians of y into Jacobians of residuals."""

        # stack matrices
        X = np.vstack(X_list)
        Z = scipy.linalg.block_diag(*Z_list)
        y = np.vstack(y_list)

        # estimate the model
        XZ = X.T @ Z
        parameters = self.covariances @ XZ @ W @ (Z.T @ y)
        residuals = y - X @ parameters

        # split the parameters and residuals into lists
        residuals_list = np.split(residuals, len(y_list), axis=0)

        # optionally convert Jacobians (making sure to never create a full square matrix)
        if convert_jacobians:
            jacobian = np.vstack(jacobian_list)
            with np.errstate(under='ignore'):
                jacobian -= X @ (self.covariances @ (XZ @ (W @ (Z.T @ jacobian))))
            jacobian_list = np.split(jacobian, len(jacobian_list), axis=0)

        return parameters, residuals_list, jacobian_list


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
    u_list: List[Array],
    Z_list: List[Array],
    covariance_type: str,
    clustering_ids: Array,
    center_moments: bool,
) -> Tuple[Array, List[Error]]:
    """Compute covariances between moments."""
    errors: List[Error] = []

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
            S, errors = compute_clustered_moment_covariances(g, clustering_ids)
        else:
            S = g.T @ g / N

    # enforce shape and symmetry
    return np.c_[S + S.T] / 2, errors


def compute_clustered_moment_covariances(g: Array, clustering_ids: Array) -> Tuple[Array, List[Error]]:
    """Compute clustered covariances between moments, which are multi-way clustered if there are multiple columns of
    clustering IDs.
    """
    errors: List[Error] = []

    # count dimensions
    N = g.shape[0]
    D = clustering_ids.shape[1]

    # combine covariances clustered by the intersection of each nonempty subset of clustering dimensions with the
    #   inclusion-exclusion formula of Cameron, Gelbach, and Miller (2011), adding subsets with an odd number of
    #   dimensions and subtracting the others (with one dimension, these are just the one-way clustered covariances)
    S = np.zeros((g.shape[1], g.shape[1]), g.dtype)
    for size in range(1, D + 1):
        for subset in itertools.combinations(range(D), size):
            g_subset = Groups(interact_ids(*(clustering_ids[:, d] for d in subset))).sum(g)
            S += (-1)**(size + 1) * (g_subset.T @ g_subset)
    S /= N

    # unlike one-way clustered covariances, multi-way clustered covariances need not be PSD, so replace any negative
    #   eigenvalues with zeros, which is the adjustment that Cameron, Gelbach, and Miller (2011) suggest for parameter
    #   covariances, made here so that the matrix can also be used to update the weighting matrix
    if D > 1:
        psd, successful = precisely_identify_psd(S)
        if successful and not psd:
            eigenvalues, eigenvectors = scipy.linalg.eigh(S)
            S = (eigenvectors * np.clip(eigenvalues, 0, None)) @ eigenvectors.T
            errors.append(exceptions.ClusteredMomentCovariancesEigenvaluesError(eigenvalues))

    return S, errors


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


def compute_gmm_parameter_sensitivity(W: Array, mean_G: Array) -> Tuple[Array, List[Error]]:
    """Compute the sensitivity measure of Andrews, Gentzkow, and Shapiro (2017)."""
    errors: List[Error] = []

    # attempt to compute the covariance matrix
    covariances_inverse = mean_G.T @ W @ mean_G
    covariances, replacement = approximately_invert(covariances_inverse)
    if replacement:
        errors.append(exceptions.GMMParameterCovariancesInversionError(covariances_inverse, replacement))

    # compute the sensitivity measure
    sensitivity = -covariances @ mean_G.T @ W
    return sensitivity, errors


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
