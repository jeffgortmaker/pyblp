"""Exceptions that are specific to the BLP problem."""


class DeltaFloatingPointError(Exception):
    """Floating point issues when computing delta."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing delta. This problem is often due to overflow and can "
            "sometimes be mitigated by reducing the magnitude of initial parameter values, by setting more "
            "conservative bounds, or by rescaling variables."
        )


class CostsFloatingPointError(Exception):
    """Floating point issues when computing marginal costs."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing marginal costs. This problem is often due to overflow "
            "and can sometimes be mitigated by reducing the magnitude of initial parameter values, by setting more "
            "conservative bounds, or by rescaling other variables."
        )


class SyntheticPricesFloatingPointError(Exception):
    """Floating point issues when computing synthetic prices."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing synthetic prices. This problem is often due to overflow "
            "and can sometimes be mitigated by making sure that the specified parameters are reasonable. For example, "
            "the parameters on prices should imply a generally downward sloping demand curve."
        )


class ChangedPricesFloatingPointError(Exception):
    """Floating point issues when computing changed prices."""

    def __str__(self):
        return (
            "Encountered floating point issues when computing changed prices. This problem is often due to overflow "
            "and can sometimes be mitigated by rescaling variables."
        )


class ThetaConvergenceError(Exception):
    """Convergence issues when optimizing over theta."""

    def __str__(self):
        return (
            "The optimization routine failed to converge. This problem can sometimes be mitigated by choosing more "
            "reasonable initial parameter values, by setting more conservative bounds, by configuring other "
            "optimization settings, by or choosing a different optimization routine."
        )


class DeltaConvergenceError(Exception):
    """Convergence issues with the fixed point computation of delta."""

    def __str__(self):
        return (
            "The fixed point computation of delta failed to converge. This problem can sometimes be mitigated by "
            "increasing the maximum number of fixed point iterations or by reducing the fixed point tolerance."
        )


class SyntheticPricesConvergenceError(Exception):
    """Convergence issues with the fixed point computation of synthetic prices."""

    def __str__(self):
        return (
            "The fixed point computation of synthetic prices failed to converge. This problem can sometimes be "
            "mitigated by increasing the maximum number of fixed point iterations, by reducing the fixed point "
            "tolerance, or by making sure that the specified parameters are reasonable. For example, the linear "
            "parameter on price should be negative and should overwhelm the effects of nonlinear parameters on prices."
        )


class ChangedPricesConvergenceError(Exception):
    """Convergence issues with the fixed point computation of changed prices."""

    def __str__(self):
        return (
            "The fixed point computation of changed prices failed to converge. This problem can sometimes be mitigated "
            "by increasing the maximum number of fixed point iterations or by reducing the fixed point tolerance."
        )


class CostsSingularityError(Exception):
    """Singular matrix encountered when computing marginal costs."""

    def __str__(self):
        return (
            "Encountered a singular intra-firm Jacobian of shares with respect to prices when computing marginal "
            "costs. This problem can sometimes by mitigated by changing initial parameter values or by setting more "
            "conservative bounds."
        )


class NonpositiveSharesError(Exception):
    """Nonpositive shares encountered when computing delta."""

    def __str__(self):
        return (
            "Encountered nonpositive shares when computing delta. This problem can sometimes be mitigated by changing "
            "initial parameter values, by setting more restrictive bounds, by using draws without negative weights, or "
            "by using a nonlinear fixed point formulation."
        )


class NonpositiveCostsError(Exception):
    """Nonpositive marginal costs encountered when recovering gamma in a log-linear specification."""

    def __str__(self):
        return (
            "Encountered nonpositive marginal costs when recovering gamma in a log-linear specification. This problem "
            "can sometimes be mitigated by reducing the magnitude of initial parameter values, by setting more "
            "conservative bounds, by changing how errors are handled, or by using a linear costs specification."
        )


class MultipleErrors(Exception):
    """Multiple errors that occurred around the same time."""

    def __new__(cls, errors):
        """If there is only one error, use it instead."""
        if len(errors) == 1:
            error = next(iter(errors))
            return error()
        return super().__new__(cls)

    def __init__(self, errors):
        """Initialize all exceptions."""
        self.exceptions = [e() for e in errors]

    def __str__(self):
        """Join all the exception string representations."""
        return "\n".join(map(str, self.exceptions))
