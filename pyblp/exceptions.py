"""Exceptions that are specific to the BLP problem."""

import re
import inspect
import collections

import numpy as np

from .utilities import format_number


class _Error(Exception):
    """Common error functionality."""

    def __eq__(self, other):
        """Defer to hashes."""
        return hash(self) == hash(other)

    def __hash__(self):
        """Hash this instance such that in collections it is indistinguishable from others with the same message."""
        return hash((type(self).__name__, str(self)))

    def __repr__(self):
        """Defer to the string representation."""
        return str(self)

    def __str__(self):
        """Replace docstring markdown with simple text."""
        doc = inspect.getdoc(self)

        # normalize LaTeX
        while True:
            match = re.search(r':math:`([^`]+)`', doc)
            if not match:
                break
            start, end = match.span()
            doc = doc[:start] + re.sub(r'\s+', ' ', re.sub(r'[\\{}]', ' ', match.group(1))).lower() + doc[end:]

        # normalize references
        while True:
            match = re.search(r':ref:`([^`]+)`', doc)
            if not match:
                break
            start, end = match.span()
            doc = doc[:start] + re.sub(r'<[^>]+>', '', match.group(1)) + doc[end:]

        # remove all remaining domains and compress whitespace
        return re.sub(r'[\s\n]+', ' ', re.sub(r':[a-z\-]+:|`', '', doc))


class _MultipleReversionError(_Error):
    """Reversion of problematic elements."""

    def __init__(self, bad_indices):
        """Store element counts."""
        self.bad = bad_indices.sum()
        self.total = bad_indices.size

    def __str__(self):
        """Supplement the error with the counts."""
        return f"{super().__str__()} Number of reverted elements: {self.bad} out of {self.total}."


class _InversionError(_Error):
    """Problems with inverting a matrix."""

    def __init__(self, matrix):
        """Compute condition number of the matrix."""
        self.condition = np.nan if not np.isfinite(matrix).all() else np.linalg.cond(matrix)

    def __str__(self):
        """Supplement the error with the condition number."""
        return f"{super().__str__()} Condition number: {format_number(self.condition)}."


class _InversionReplacementError(_InversionError):
    """Problems with inverting a matrix led to the use of a replacement such as an approximation."""

    def __init__(self, matrix, replacement):
        """Store the replacement description."""
        super().__init__(matrix)
        self.replacement = replacement

    def __str__(self):
        """Supplement the error with the description."""
        return f"{super().__str__()} The inverse was replaced with {self.replacement}."


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
    """Specified initial nonlinear parameters are likely to give rise to overflow during choice probability computation.

    Consider choosing smaller initial values, rescaling data, removing outliers, or changing the floating point
    precision.

    """


class NonpositiveCostsError(_Error):
    """Encountered nonpositive marginal costs in a log-linear specification.

    This problem can sometimes be mitigated by bounding costs from below, choosing more reasonable initial parameter
    values, setting more conservative parameter bounds, or using a linear costs specification.

    """


class InvalidParameterCovariancesError(_Error):
    """Failed to compute standard errors because of invalid estimated covariances of GMM parameters."""


class InvalidMomentCovariancesError(_Error):
    """Failed to compute a weighting matrix because of invalid estimated covariances of GMM moments."""


class DeltaFloatingPointError(_Error):
    r"""Encountered floating point issues when computing :math:`\delta` or its Jacobian with respect to :math:`\theta`.

    This problem is often due to prior problems, overflow, or nonpositive shares, and can sometimes be mitigated by
    choosing smaller initial parameter values, setting more conservative bounds, rescaling data, removing outliers,
    changing the floating point precision, using different iteration and integration configurations, or using a
    different fixed point formulation.

    """


class CostsFloatingPointError(_Error):
    r"""Encountered floating point issues when computing marginal costs or their Jacobian with respect to
    :math:`\theta`.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by choosing smaller initial
    parameter values, setting more conservative bounds, rescaling data, removing outliers, or changing the floating
    point precision.

    """


class SyntheticPricesFloatingPointError(_Error):
    """Encountered floating point issues when computing synthetic prices.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by making sure that the
    specified parameters are reasonable. For example, the parameters on prices should generally imply a downward sloping
    demand curve.

    """


class BertrandNashPricesFloatingPointError(_Error):
    """Encountered floating point issues when computing Bertrand-Nash prices.

    This problem is often due to prior problems or overflow and can sometimes be mitigated by rescaling data, removing
    outliers, or changing the floating point precision.

    """


class AbsorptionConvergenceError(_Error):
    """An iterative de-meaning procedure failed to converge when absorbing fixed effects.

    This problem can sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the
    fixed point tolerance, configuring other iteration settings, or choosing less complicated sets of fixed effects.

    """


class ThetaConvergenceError(_Error):
    """The optimization routine failed to converge.

    This problem can sometimes be mitigated by choosing more reasonable initial parameter values, setting more
    conservative bounds, or configuring other optimization settings.

    """


class DeltaConvergenceError(_Error):
    r"""The fixed point computation of :math:`\delta` failed to converge.

    This problem can sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the
    fixed point tolerance, or configuring other iteration settings.

    """


class SyntheticPricesConvergenceError(_Error):
    """The fixed point computation of synthetic prices failed to converge.

    This problem can sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the
    fixed point tolerance, configuring other iteration settings, or making sure the specified parameters are reasonable.
    For example, the parameters on prices should generally imply a downward sloping demand curve.

    """


class BertrandNashPricesConvergenceError(_Error):
    """The fixed point computation of Bertrand-Nash prices failed to converge.

    This problem can sometimes be mitigated by increasing the maximum number of fixed point iterations, increasing the
    fixed point tolerance, or configuring other iteration settings.

    """


class ObjectiveReversionError(_Error):
    """Reverted a problematic GMM objective value."""


class GradientReversionError(_MultipleReversionError):
    """Reverted problematic elements in the GMM objective gradient."""


class DeltaReversionError(_MultipleReversionError):
    r"""Reverted problematic elements in :math:`\delta`."""


class CostsReversionError(_MultipleReversionError):
    """Reverted problematic marginal costs."""


class XiJacobianReversionError(_MultipleReversionError):
    r"""Reverted problematic elements in the Jacobian of :math:`\xi` (equivalently, of :math:`\delta`) with respect to
    :math:`\theta`.

    """


class OmegaJacobianReversionError(_MultipleReversionError):
    r"""Reverted problematic elements in the Jacobian of :math:`\omega` (equivalently, of transformed marginal costs)
    with respect to :math:`\theta`.

    """


class AbsorptionInversionError(_InversionError):
    """Failed to invert the A matrix from :ref:`Somaini and Wolak (2016) <sw16>` when absorbing two-way fixed effects.

    The formulated fixed effects may be highly collinear.

    """


class SharesByXiJacobianInversionError(_InversionReplacementError):
    r"""Failed to invert a Jacobian of shares with respect to :math:`\xi` when computing the Jacobian of :math:`\xi`
    (equivalently, of :math:`\delta`) with respect to :math:`\theta`.

    """


class IntraFirmJacobianInversionError(_InversionReplacementError):
    r"""Failed to invert an intra-firm Jacobian of shares with respect to prices when computing :math:`\eta`."""


class LinearParameterCovariancesInversionError(_InversionReplacementError):
    """Failed to invert an estimated covariance matrix of linear IV parameters.

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
