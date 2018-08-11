"""Exceptions that are specific to the BLP problem."""


class LargeInitialParametersError(Exception):
    """Large initial nonlinear parameters are likely to give rise to overflow during probability computation."""

    def __str__(self):
        return (
            "The specified initial nonlinear parameters are likely to give rise to overflow during choice probability "
            "computation. Consider choosing smaller initial values, rescaling data, removing outliers, or changing "
            "options.dtype."
        )


class ObjectiveReversionError(Exception):
    """Reversion of a problematic objective value."""

    def __str__(self):
        return "Reverted a problematic GMM objective value."


class _MultipleReversionError(Exception):
    """Reversion of problematic elements."""

    def __init__(self, reverted):
        self.reverted = reverted


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


class DeltaFloatingPointError(Exception):
    """Floating point issues when computing delta."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing delta or its Jacobian with respect to theta. This "
            "problem is often due to overflow and can sometimes be mitigated by choosing smaller initial parameter "
            "values, setting more conservative bounds, rescaling data, removing outliers, or changing options.dtype."
        )


class CostsFloatingPointError(Exception):
    """Floating point issues when computing marginal costs."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing marginal costs or their Jacobian with respect to theta. "
            "This problem is often due to overflow and can sometimes be mitigated by choosing smaller initial "
            "parameter values, setting more conservative bounds, rescaling data, removing outliers, or changing "
            "options.dtype."
        )


class SyntheticPricesFloatingPointError(Exception):
    """Floating point issues when computing synthetic prices."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing synthetic prices. This problem is often due to overflow "
            "and can sometimes be mitigated by making sure that the specified parameters are reasonable. For example, "
            "the parameters on prices should generally imply a downward sloping demand curve."
        )


class ChangedPricesFloatingPointError(Exception):
    """Floating point issues when computing changed prices."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing changed prices. This problem is often due to overflow "
            "and can sometimes be mitigated by rescaling data."
        )


class AbsorptionConvergenceError(Exception):
    """Convergence issues when iteratively demeaning a matrix to absorb fixed effects."""

    def __str__(self):
        return (
            "An iterative de-meaning procedure failed to converge when absorbing fixed effects. This problem can "
            "sometimes be mitigated by choosing less complicated sets of fixed effects or by configuring de-meaning "
            "iteration options."
        )


class AbsorptionInversionError(Exception):
    """Problems with inversion of the A matrix in Somaini and Wolak (2016) when absorbing two-way fixed effects."""

    def __str__(self):
        return (
            "Encountered a singular matrix when absorbing two-way fixed effects. Specifically, the A matrix from "
            "Somaini and  Wolak (2016) is singular. There may be multicollinearity in the formulated two-way fixed "
            "effecdts."
        )


class ThetaConvergenceError(Exception):
    """Convergence issues when optimizing over theta."""

    def __str__(self):
        return (
            "The optimization routine failed to converge. This problem can sometimes be mitigated by choosing more "
            "reasonable initial parameter values, setting more conservative bounds, or configuring other optimization "
            "settings."
        )


class DeltaConvergenceError(Exception):
    """Convergence issues with the fixed point computation of delta."""

    def __str__(self):
        return (
            "The fixed point computation of delta failed to converge. This problem can sometimes be mitigated by "
            "increasing the maximum number of fixed point iterations or reducing the fixed point tolerance."
        )


class SyntheticPricesConvergenceError(Exception):
    """Convergence issues with the fixed point computation of synthetic prices."""

    def __str__(self):
        return (
            "The fixed point computation of synthetic prices failed to converge. This problem can sometimes be "
            "mitigated by increasing the maximum number of fixed point iterations, reducing the fixed point tolerance, "
            "or making sure that the specified parameters are reasonable. For example, the parameters on prices should "
            "generally imply a downward sloping demand curve."
        )


class ChangedPricesConvergenceError(Exception):
    """Convergence issues with the fixed point computation of changed prices."""

    def __str__(self):
        return (
            "The fixed point computation of changed prices failed to converge. This problem can sometimes be mitigated "
            "by increasing the maximum number of fixed point iterations or reducing the fixed point tolerance."
        )


class CostsSingularityError(Exception):
    """Singular matrix encountered when computing marginal costs."""

    def __str__(self):
        return (
            "Encountered a singular intra-firm Jacobian of shares with respect to prices when computing marginal "
            "costs. This problem can sometimes by mitigated by changing initial parameter values or setting more "
            "conservative bounds."
        )


class NonpositiveSharesError(Exception):
    """Nonpositive shares encountered when computing delta."""

    def __str__(self):
        return (
            "Encountered nonpositive shares when computing delta. This problem can sometimes be mitigated by changing "
            "initial parameter values, setting more conservative bounds, using a different integration specification, "
            "or using a nonlinear fixed point formulation."
        )


class NonpositiveCostsError(Exception):
    """Nonpositive marginal costs encountered when recovering gamma in a log-linear specification."""

    def __str__(self):
        return (
            "Encountered nonpositive marginal costs when recovering gamma in a log-linear specification. This problem "
            "can sometimes be mitigated by bounding costs from below, choosing more reasonable initial parameter "
            "values, setting more conservative parameter bounds, or using a linear costs specification."
        )


class _CovariancesInversionError(Exception):
    """Problems with inversion of a covariance matrix."""

    def __init__(self, approximation):
        self.approximation = approximation


class LinearParameterCovariancesInversionError(_CovariancesInversionError):
    """Problems with inversion of the covariance matrix of linear IV parameters."""

    def __str__(self):
        return (
            f"An estimated covariance matrix of linear IV parameters is singular. Its inverse was approximated by "
            f"{self.approximation}. One or more data matrices may be highly collinear."
        )


class GMMParameterCovariancesInversionError(_CovariancesInversionError):
    """Problems with inversion of the covariance matrix of GMM parameters."""

    def __str__(self):
        return (
            f"An estimated covariance matrix of GMM parameters is singular. Its inverse was approximated by "
            f"{self.approximation}. One or more data matrices may be highly collinear."
        )


class GMMMomentCovariancesInversionError(_CovariancesInversionError):
    """Problems with inversion of the covariance matrix of GMM moments."""

    def __str__(self):
        return (
            f"An estimated covariance matrix of GMM moments is singular. Its inverse was approximated by "
            f"{self.approximation}. One or more data matrices may be highly collinear."
        )


class InvalidCovariancesError(Exception):
    """Failure to compute standard errors because of invalid estimated covariances."""

    def __str__(self):
        return "Failed to compute standard errors because of invalid estimated covariances."


class InvalidWeightsError(Exception):
    """Failure to compute a GMM weighting matrix because of invalid weights."""

    def __str__(self):
        return "Failed to compute a weighting matrix because of invalid estimated moments."


class MultipleErrors(Exception):
    """Multiple errors that occurred around the same time."""

    def __new__(cls, errors):
        """If there is only one error, use it instead."""
        if len(errors) == 1:
            error = next(iter(errors))
            return error()
        return super().__new__(cls)

    def __init__(self, errors):
        """Initialize all exceptions in a deterministic order."""
        self.exceptions = sorted([e() for e in errors], key=str)

    def __str__(self):
        """Join all the exception string representations."""
        return "\n".join(map(str, self.exceptions))
