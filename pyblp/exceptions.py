"""BLP-specific exceptions."""

import collections
from typing import Any, List, Sequence

from .utilities.basics import Error, NumericalError, MultipleReversionError, InversionError, InversionReplacementError


class MultipleErrors(Error):
    """Multiple errors that occurred around the same time."""

    _errors: List[Error]

    def __new__(cls, errors: Sequence[Error]) -> Any:
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


class NonpositiveSyntheticCostsError(Error):
    """Encountered nonpositive synthetic marginal costs in a log-linear specification.

    This problem can sometimes be mitigated by more reasonable initial parameter values or using a linear costs
    specification.

    """


class InvalidParameterCovariancesError(Error):
    """Failed to compute standard errors because of invalid estimated covariances of GMM parameters."""


class InvalidMomentCovariancesError(Error):
    """Failed to compute a weighting matrix because of invalid estimated covariances of GMM moments."""


class DeltaNumericalError(NumericalError):
    r"""Encountered a numerical error when computing :math:`\delta`.

    This problem is often due to prior problems, overflow, or nonpositive shares, and can sometimes be mitigated by
    choosing smaller initial parameter values, setting more conservative bounds, rescaling data, removing outliers,
    changing the floating point precision, or using different optimization, iteration, or integration configurations.

    """


class CostsNumericalError(NumericalError):
    """Encountered a numerical error when computing marginal costs.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by choosing smaller initial
    parameter values, setting more conservative bounds, rescaling data, removing outliers, changing the floating point
    precision, or using different optimization or cost configurations.

    """


class MicroMomentsNumericalError(NumericalError):
    """Encountered a numerical error when computing micro moments.

    This problem is often due to prior problems, overflow, or nonpositive shares, and can sometimes be mitigated by
    choosing smaller initial parameter values, setting more conservative bounds, rescaling data, removing outliers,
    changing the floating point precision, or using different optimization, iteration, or integration configurations.

    """


class XiByThetaJacobianNumericalError(NumericalError):
    r"""Encountered a numerical error when computing the Jacobian of :math:`\xi` (equivalently, of :math:`\delta`)
    with respect to :math:`\theta`.

    This problem is often due to prior problems, overflow, or nonpositive shares, and can sometimes be mitigated by
    choosing smaller initial parameter values, setting more conservative bounds, rescaling data, removing outliers,
    changing the floating point precision, or using different optimization, iteration, or integration configurations.

    """


class OmegaByThetaJacobianNumericalError(NumericalError):
    r"""Encountered a numerical error when computing the Jacobian of :math:`\omega` (equivalently, of transformed
    marginal costs) with respect to :math:`\theta`.

    This problem is often due to prior problems or overflow, and can sometimes be mitigated by choosing smaller initial
    parameter values, setting more conservative bounds, rescaling data, removing outliers, changing the floating point
    precision, or using different optimization or cost configurations.

    """


class MicroMomentsByThetaJacobianNumericalError(NumericalError):
    r"""Encountered a numerical error when computing the Jacobian of micro moments with respect to :math:`\theta`."""


class MicroMomentCovariancesNumericalError(NumericalError):
    """Encountered a numerical error when computing micro moment covariances."""


class SyntheticPricesNumericalError(NumericalError):
    """Encountered a numerical error when computing synthetic prices.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by making sure that the
    specified parameters are reasonable. For example, the parameters on prices should generally imply a downward sloping
    demand curve.

    """


class SyntheticSharesNumericalError(NumericalError):
    """Encountered a numerical error when computing synthetic shares.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by making sure that the
    specified parameters are reasonable. For example, the parameters on prices should generally imply a downward sloping
    demand curve.

    """


class SyntheticDeltaNumericalError(NumericalError):
    r"""Encountered a numerical error when computing the synthetic :math:`\delta`.

    This problem is often due to prior problems, overflow, or nonpositive shares, and can sometimes be mitigated by
    making sure that the specified parameters are reasonable.

    """


class SyntheticCostsNumericalError(NumericalError):
    """Encountered a numerical error when computing synthetic marginal costs.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by making sure that the
    specified parameters are reasonable.

    """


class SyntheticMicroMomentsNumericalError(NumericalError):
    """Encountered a numerical error when computing synthetic micro moments."""


class EquilibriumRealizationNumericalError(NumericalError):
    """Encountered a numerical error when solving for a realization of equilibrium prices and shares."""


class XiByThetaJacobianRealizationNumericalError(NumericalError):
    r"""Encountered a numerical error when computing a realization of the Jacobian of :math:`\xi` (equivalently, of
    :math:`\delta`) with respect to :math:`\theta`.

    """


class OmegaByThetaJacobianRealizationNumericalError(NumericalError):
    r"""Encountered a numerical error when computing a realization of the Jacobian of :math:`\omega` (equivalently, of
    transformed marginal costs) with respect to :math:`\theta`.

    """


class PostEstimationNumericalError(NumericalError):
    """Encountered a numerical error when computing a post-estimation output."""


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


class SyntheticDeltaConvergenceError(Error):
    r"""The fixed point computation of the synthetic :math:`\delta` failed to converge.

    This problem can sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the
    fixed point tolerance, choosing more reasonable parameter values, or using a different iteraiton configuration.

    """


class EquilibriumPricesConvergenceError(Error):
    """The fixed point computation of equilibrium prices failed to converge.

    This problem can sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the
    fixed point tolerance, or configuring other iteration settings.

    """


class ObjectiveReversionError(Error):
    """Reverted a problematic GMM objective value."""


class GradientReversionError(MultipleReversionError):
    """Reverted problematic elements in the GMM objective gradient."""


class DeltaReversionError(MultipleReversionError):
    r"""Reverted problematic elements in :math:`\delta`."""


class CostsReversionError(MultipleReversionError):
    """Reverted problematic marginal costs."""


class MicroMomentsReversionError(MultipleReversionError):
    """Reverted problematic micro moments."""


class XiByThetaJacobianReversionError(MultipleReversionError):
    r"""Reverted problematic elements in the Jacobian of :math:`\xi` (equivalently, of :math:`\delta`) with respect to
    :math:`\theta`.

    """


class OmegaByThetaJacobianReversionError(MultipleReversionError):
    r"""Reverted problematic elements in the Jacobian of :math:`\omega` (equivalently, of transformed marginal costs)
    with respect to :math:`\theta`.

    """


class MicroMomentsByThetaJacobianReversionError(MultipleReversionError):
    r"""Reverted problematic elements in the Jacobian of micro moments with respect to :math:`\theta`."""


class AbsorptionInversionError(InversionError):
    """Failed to invert the :math:`A` matrix from :ref:`references:Somaini and Wolak (2016)` when absorbing two-way
    fixed effects.

    The formulated fixed effects may be highly collinear.

    """


class HessianEigenvaluesError(InversionError):
    """Failed to compute eigenvalues for the GMM objective's (reduced) Hessian matrix."""


class FittedValuesInversionError(InversionReplacementError):
    """Failed to invert an estimated covariance when computing fitted values.

    There are probably collinearity issues.

    """


class SharesByXiJacobianInversionError(InversionReplacementError):
    r"""Failed to invert a Jacobian of shares with respect to :math:`\xi` when computing the Jacobian of :math:`\xi`
    (equivalently, of :math:`\delta`) with respect to :math:`\theta`.

    """


class IntraFirmJacobianInversionError(InversionReplacementError):
    r"""Failed to invert an intra-firm Jacobian of shares with respect to prices when computing :math:`\eta`."""


class LinearParameterCovariancesInversionError(InversionReplacementError):
    """Failed to invert an estimated covariance matrix of linear parameters.

    One or more data matrices may be highly collinear.

    """


class GMMParameterCovariancesInversionError(InversionReplacementError):
    """Failed to invert an estimated covariance matrix of GMM parameters.

    One or more data matrices may be highly collinear.

    """


class GMMMomentCovariancesInversionError(InversionReplacementError):
    """Failed to invert an estimated covariance matrix of GMM moments.

    One or more data matrices may be highly collinear.

    """


class WaldInversionError(InversionReplacementError):
    """Failed to invert the matrix in the Wald statistic expression."""
