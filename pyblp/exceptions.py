"""BLP-specific exceptions."""

import collections
from typing import List, Sequence

import numpy as np

from .utilities.basics import Array, Error, format_number


class _MultipleReversionError(Error):
    """Reversion of problematic elements."""

    _bad: int
    _total: int

    def __init__(self, bad_indices: Array) -> None:
        """Store element counts."""
        self._bad = bad_indices.sum()
        self._total = bad_indices.size

    def __str__(self) -> str:
        """Supplement the error with the counts."""
        return f"{super().__str__()} Number of reverted elements: {self._bad} out of {self._total}."


class _InversionError(Error):
    """Problems with inverting a matrix."""

    _condition: float

    def __init__(self, matrix: Array) -> None:
        """Compute condition number of the matrix."""
        self._condition = np.nan if not np.isfinite(matrix).all() else np.linalg.cond(matrix.astype(np.float64))

    def __str__(self) -> str:
        """Supplement the error with the condition number."""
        return f"{super().__str__()} Condition number: {format_number(self._condition)}."


class _InversionReplacementError(_InversionError):
    """Problems with inverting a matrix led to the use of a replacement such as an approximation."""

    _replacement: str

    def __init__(self, matrix: Array, replacement: str) -> None:
        """Store the replacement description."""
        super().__init__(matrix)
        self._replacement = replacement

    def __str__(self) -> str:
        """Supplement the error with the description."""
        return f"{super().__str__()} The inverse was replaced with {self._replacement}."


class MultipleErrors(Error):
    """Multiple errors that occurred around the same time."""

    _errors: List[Error]

    def __new__(cls, errors: Sequence[Error]) -> Exception:
        """Defer to the class of a singular error."""
        if len(errors) == 1:
            return next(iter(errors))
        return super().__new__(cls)

    def __init__(self, errors: Sequence[Error]) -> None:
        """Store distinct errors."""
        self._errors = list(collections.OrderedDict.fromkeys(errors))

    def __str__(self) -> str:
        """Combine all the error messages."""
        return "\n".join(str(e) for e in self._errors)


class NonpositiveCostsError(Error):
    """Encountered nonpositive marginal costs in a log-linear specification.

    This problem can sometimes be mitigated by bounding costs from below, choosing more reasonable initial parameter
    values, setting more conservative parameter bounds, or using a linear costs specification.

    """


class InvalidParameterCovariancesError(Error):
    """Failed to compute standard errors because of invalid estimated covariances of GMM parameters."""


class InvalidMomentCovariancesError(Error):
    """Failed to compute a weighting matrix because of invalid estimated covariances of GMM moments."""


class DeltaFloatingPointError(Error):
    r"""Encountered floating point issues when computing :math:`\delta`.

    This problem is often due to prior problems, overflow, or nonpositive shares, and can sometimes be mitigated by
    choosing smaller initial parameter values, setting more conservative bounds, rescaling data, removing outliers,
    changing the floating point precision, or using different optimization, iteration, or integration configurations.

    """


class XiByThetaJacobianFloatingPointError(Error):
    r"""Encountered floating point issues when computing the Jacobian of :math:`\xi` (equivalently, of :math:`\delta`)
    with respect to :math:`\theta`.

    This problem is often due to prior problems, overflow, or nonpositive shares, and can sometimes be mitigated by
    choosing smaller initial parameter values, setting more conservative bounds, rescaling data, removing outliers,
    changing the floating point precision, or using different optimization, iteration, or integration configurations.

    """


class CostsFloatingPointError(Error):
    """Encountered floating point issues when computing marginal costs.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by choosing smaller initial
    parameter values, setting more conservative bounds, rescaling data, removing outliers, changing the floating point
    precision, or using different optimization or cost configurations.

    """


class OmegaByThetaJacobianFloatingPointError(Error):
    r"""Encountered floating point issues when computing the Jacobian of :math:`\omega` (equivalently, of transformed
    marginal costs) with respect to :math:`\theta`.

    This problem is often due to prior problems or overflow, and can sometimes be mitigated by choosing smaller initial
    parameter values, setting more conservative bounds, rescaling data, removing outliers, changing the floating point
    precision, or using different optimization or cost configurations.

    """


class SyntheticPricesFloatingPointError(Error):
    """Encountered floating point issues when computing synthetic prices.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by making sure that the
    specified parameters are reasonable. For example, the parameters on prices should generally imply a downward sloping
    demand curve.

    """


class SyntheticSharesFloatingPointError(Error):
    """Encountered floating point issues when computing synthetic shares.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by making sure that the
    specified parameters are reasonable. For example, the parameters on prices should generally imply a downward sloping
    demand curve.

    """


class EquilibriumPricesFloatingPointError(Error):
    """Encountered floating point issues when computing equilibrium prices.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by rescaling data, removing
    outliers, or changing the floating point precision.

    """


class EquilibriumSharesFloatingPointError(Error):
    """Encountered floating point issues when computing equilibrium shares.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by rescaling data, removing
    outliers, or changing the floating point precision.

    """


class AbsorptionConvergenceError(Error):
    """An iterative de-meaning procedure failed to converge when absorbing fixed effects.

    This problem can sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the
    fixed point tolerance, configuring other iteration settings, or choosing less complicated sets of fixed effects.

    """


class ThetaConvergenceError(Error):
    """The optimization routine failed to converge.

    This problem can sometimes be mitigated by choosing more reasonable initial parameter values, setting more
    conservative bounds, or configuring other optimization settings.

    """


class DeltaConvergenceError(Error):
    r"""The fixed point computation of :math:`\delta` failed to converge.

    This problem can sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the
    fixed point tolerance, choosing more reasonable initial parameter values, setting more conservative bounds, or using
    different iteration or optimization configurations.

    """


class SyntheticPricesConvergenceError(Error):
    """The fixed point computation of synthetic prices failed to converge.

    This problem can sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the
    fixed point tolerance, configuring other iteration settings, or making sure the specified parameters are reasonable.
    For example, the parameters on prices should generally imply a downward sloping demand curve.

    """


class EquilibriumPricesConvergenceError(Error):
    """The fixed point computation of equilibrium prices failed to converge.

    This problem can sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the
    fixed point tolerance, or configuring other iteration settings.

    """


class ObjectiveReversionError(Error):
    """Reverted a problematic GMM objective value."""


class GradientReversionError(_MultipleReversionError):
    """Reverted problematic elements in the GMM objective gradient."""


class DeltaReversionError(_MultipleReversionError):
    r"""Reverted problematic elements in :math:`\delta`."""


class CostsReversionError(_MultipleReversionError):
    """Reverted problematic marginal costs."""


class XiByThetaJacobianReversionError(_MultipleReversionError):
    r"""Reverted problematic elements in the Jacobian of :math:`\xi` (equivalently, of :math:`\delta`) with respect to
    :math:`\theta`.

    """


class OmegaByThetaJacobianReversionError(_MultipleReversionError):
    r"""Reverted problematic elements in the Jacobian of :math:`\omega` (equivalently, of transformed marginal costs)
    with respect to :math:`\theta`.

    """


class AbsorptionInversionError(_InversionError):
    """Failed to invert the :math:`A` matrix from :ref:`references:Somaini and Wolak (2016)` when absorbing two-way
    fixed effects.

    The formulated fixed effects may be highly collinear.

    """


class HessianEigenvaluesError(_InversionError):
    """Failed to compute eigenvalues for the GMM objective's Hessian matrix."""


class FittedValuesInversionError(_InversionReplacementError):
    """Failed to invert an estimated covariance when computing fitted values.

    There are probably collinearity issues.

    """


class SharesByXiJacobianInversionError(_InversionReplacementError):
    r"""Failed to invert a Jacobian of shares with respect to :math:`\xi` when computing the Jacobian of :math:`\xi`
    (equivalently, of :math:`\delta`) with respect to :math:`\theta`.

    """


class IntraFirmJacobianInversionError(_InversionReplacementError):
    r"""Failed to invert an intra-firm Jacobian of shares with respect to prices when computing :math:`\eta`."""


class LinearParameterCovariancesInversionError(_InversionReplacementError):
    """Failed to invert an estimated covariance matrix of linear parameters.

    One or more data matrices may be highly collinear.

    """


class GMMParameterCovariancesInversionError(_InversionReplacementError):
    """Failed to invert an estimated covariance matrix of GMM parameters.

    One or more data matrices may be highly collinear.

    """


class GMMMomentCovariancesInversionError(_InversionReplacementError):
    """Failed to invert an estimated covariance matrix of GMM moments.

    One or more data matrices may be highly collinear.

    """
