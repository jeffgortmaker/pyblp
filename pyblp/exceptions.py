"""Exceptions that are specific to the BLP problem."""

import collections

import numpy as np

from .utilities import output


class _Error(Exception):
    """Common error functionality."""

    def __hash__(self):
        """Hash this instance such that in collections it is indistinguishable from others with the same message."""
        return hash((type(self).__name__, str(self)))

    def __eq__(self, other):
        """Defer to hashes."""
        return hash(self) == hash(other)

    def __repr__(self):
        """Defer to the string representation."""
        return str(self)


class _MultipleReversionError(_Error):
    """Reversion of problematic elements."""

    def __init__(self, bad_indices):
        """Store a message about the number of non-finite elements."""
        self.reverted = f"{bad_indices.sum()} out of {bad_indices.size}"


class _InversionError(_Error):
    """Problems with inverting a matrix."""

    def __init__(self, matrix):
        """Compute and format the condition number of the matrix."""
        self.condition = output.format_number(np.nan if not np.isfinite(matrix).all() else np.linalg.cond(matrix))


class _InversionReplacementError(_InversionError):
    """Problems with inverting a matrix led to the use of a replacement such as an approximation."""

    def __init__(self, matrix, replacement):
        """Store the replacement description."""
        super().__init__(matrix)
        self.replacement = replacement


class MultipleErrors(_Error):
    """Multiple errors that occurred around the same time."""

    def __new__(cls, errors):
        """Defer to the class of a singular error."""
        if len(errors) == 1:
            return next(iter(errors))
        return super().__new__(cls)

    def __init__(self, errors):
        """Store distinct errors."""
        self.errors = list(collections.OrderedDict.fromkeys(errors))

    def __str__(self):
        """Combine all the error messages."""
        return "\n".join(str(e) for e in self.errors)


class LargeInitialParametersError(_Error):
    """Large initial nonlinear parameters encountered."""

    def __str__(self):
        return (
            "The specified initial nonlinear parameters are likely to give rise to overflow during choice probability "
            "computation. Consider choosing smaller initial values, rescaling data, removing outliers, or changing "
            "options.dtype."
        )


class NonpositiveSharesError(_Error):
    """Nonpositive shares encountered during delta computation."""

    def __str__(self):
        return (
            "Encountered nonpositive shares when computing delta. This problem can sometimes be mitigated by changing "
            "initial parameter values, setting more conservative bounds, using a different integration configuration, "
            "or using a nonlinear fixed point formulation."
        )


class NonpositiveCostsError(_Error):
    """Nonpositive marginal costs encountered in a log-linear specification."""

    def __str__(self):
        return (
            "Encountered nonpositive marginal costs in a log-linear specification. This problem can sometimes be "
            "mitigated by bounding costs from below, choosing more reasonable initial parameter values, setting more "
            "conservative parameter bounds, or using a linear costs specification."
        )


class InvalidParameterCovariancesError(_Error):
    """Failure to compute standard errors because of invalid estimated covariances of GMM parameters."""

    def __str__(self):
        return "Failed to compute standard errors because of invalid estimated covariances of GMM parameters."


class InvalidMomentCovariancesError(_Error):
    """Failure to compute a weighting matrix because of invalid estimated covariances of GMM moments."""

    def __str__(self):
        return "Failed to compute a weighting matrix because of invalid estimated covariances of GMM moments."


class DeltaFloatingPointError(_Error):
    """Floating point problems with delta computation."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing delta or its Jacobian with respect to theta. This "
            "problem is often due to prior problems or overflow and can sometimes be mitigated by choosing smaller "
            "initial parameter values, setting more conservative bounds, rescaling data, removing outliers, or "
            "changing options.dtype."
        )


class CostsFloatingPointError(_Error):
    """Floating point problems with marginal cost computation."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing marginal costs or their Jacobian with respect to theta. "
            "This problem is often due to prior problems or overflow and can sometimes be mitigated by choosing "
            "smaller initial parameter values, setting more conservative bounds, rescaling data, removing outliers, or "
            "changing options.dtype."
        )


class SyntheticPricesFloatingPointError(_Error):
    """Floating point problems with synthetic price computation."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing synthetic prices. This problem is often due to prior "
            "problems or overflow and can sometimes be mitigated by making sure that the specified parameters are "
            "reasonable. For example, the parameters on prices should generally imply a downward sloping demand curve."
        )


class BertrandNashPricesFloatingPointError(_Error):
    """Floating point problems with Bertrand-Nash price computation."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing Bertrand-Nash prices. This problem is often due to prior "
            "problems or overflow and can sometimes be mitigated by rescaling data, removing outliers, or changing "
            "options.dtype."
        )


class AbsorptionConvergenceError(_Error):
    """Convergence problems with iterative de-meaning."""

    def __str__(self):
        return (
            "An iterative de-meaning procedure failed to converge when absorbing fixed effects. This problem can "
            "sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the fixed "
            "point tolerance, configuring other iteration settings, or choosing less complicated sets of fixed effects."
        )


class ThetaConvergenceError(_Error):
    """Convergence problems with theta optimization."""

    def __str__(self):
        return (
            "The optimization routine failed to converge. This problem can sometimes be mitigated by choosing more "
            "reasonable initial parameter values, setting more conservative bounds, or configuring other optimization "
            "settings."
        )


class DeltaConvergenceError(_Error):
    """Convergence problems with the fixed point computation of delta."""

    def __str__(self):
        return (
            "The fixed point computation of delta failed to converge. This problem can sometimes be mitigated by "
            "increasing the maximum number of fixed point iterations, increasing the fixed point tolerance, or "
            "configuring other iteration settings."
        )


class SyntheticPricesConvergenceError(_Error):
    """Convergence problems with the fixed point computation of synthetic prices."""

    def __str__(self):
        return (
            "The fixed point computation of synthetic prices failed to converge. This problem can sometimes be "
            "mitigated by increasing the maximum number of fixed point iterations, increasing the fixed point "
            "tolerance, configuring other iteration settings, or making sure the specified parameters are reasonable. "
            "For example, the parameters on prices should generally imply a downward sloping demand curve."
        )


class BertrandNashPricesConvergenceError(_Error):
    """Convergence problems with the fixed point computation of Bertrand-Nash prices."""

    def __str__(self):
        return (
            "The fixed point computation of Bertrand-Nash prices failed to converge. This problem can sometimes be "
            "mitigated by increasing the maximum number of fixed point iterations, increasing the fixed point "
            "tolerance, or configuring other iteration settings."
        )


class ObjectiveReversionError(_Error):
    """Reversion of a problematic objective value."""

    def __str__(self):
        return "Reverted a problematic GMM objective value."


class GradientReversionError(_MultipleReversionError):
    """Reversion of problematic elements in the gradient."""

    def __str__(self):
        return f"Number of problematic elements in the GMM objective gradient that were reverted: {self.reverted}."


class DeltaReversionError(_MultipleReversionError):
    """Reversion of problematic elements in delta."""

    def __str__(self):
        return f"Number of problematic elements in delta that were reverted: {self.reverted}."


class CostsReversionError(_MultipleReversionError):
    """Reversion of problematic marginal costs."""

    def __str__(self):
        return f"Number of problematic marginal costs that were reverted: {self.reverted}."


class XiJacobianReversionError(_MultipleReversionError):
    """Reversion of problematic elements in the Jacobian of xi with respect to theta."""

    def __str__(self):
        return (
            f"Number of problematic elements in the Jacobian of xi (equivalently, of delta) with respect to theta that "
            f"were reverted: {self.reverted}."
        )


class OmegaJacobianReversionError(_MultipleReversionError):
    """Reversion of problematic elements in the Jacobian of omega with respect to theta."""

    def __str__(self):
        return (
            f"Number of problematic elements in the Jacobian of omega (equivalently, of transformed marginal costs) "
            f"with respect to theta that were reverted: {self.reverted}."
        )


class AbsorptionInversionError(_InversionError):
    """Problems with inversion of the A matrix in Somaini and Wolak (2016)."""

    def __str__(self):
        return (
            f"Failed to invert the A matrix from Somaini and Wolak (2016) when absorbing two-way fixed effects. Its "
            f"condition number is {self.condition}. The formulated fixed effects may be highly collinear."
        )


class SharesByXiJacobianInversionError(_InversionReplacementError):
    """Problems with inversion of the Jacobian of shares with respect to xi."""

    def __str__(self):
        return (
            f"Failed to invert a Jacobian of shares with respect to xi when computing the Jacobian of xi "
            f"(equivalently, of delta) with respect to theta. Its condition number is {self.condition} and its inverse "
            f"was replaced with {self.replacement}."
        )


class IntraFirmJacobianInversionError(_InversionReplacementError):
    """Problems with inversion of the intra-firm Jacobian of shares with respect to prices."""

    def __str__(self):
        return (
            f"Failed to invert an intra-firm Jacobian of shares with respect to prices when computing eta. Its "
            f"condition number is {self.condition} and its inverse was replaced with {self.replacement}."
        )


class LinearParameterCovariancesInversionError(_InversionReplacementError):
    """Problems with inversion of a covariance matrix of linear IV parameters."""

    def __str__(self):
        return (
            f"Failed to invert an estimated covariance matrix of linear IV parameters. Its condition number is "
            f"{self.condition} and its inverse was replaced with {self.replacement}. One or more data matrices may be "
            f"highly collinear."
        )


class GMMParameterCovariancesInversionError(_InversionReplacementError):
    """Problems with inversion of a covariance matrix of GMM parameters."""

    def __str__(self):
        return (
            f"Failed to invert an estimated covariance matrix of GMM parameters. Its condition number is "
            f"{self.condition} and its inverse was replaced with {self.replacement}. One or more data matrices may be "
            f"highly collinear."
        )


class GMMMomentCovariancesInversionError(_InversionReplacementError):
    """Problems with inversion of a covariance matrix of GMM moments."""

    def __str__(self):
        return (
            f"Failed to invert an estimated covariance matrix of GMM moments. Its condition number is {self.condition} "
            f"and its inverse was replaced with {self.replacement}. One or more data matrices may be highly collinear."
        )
