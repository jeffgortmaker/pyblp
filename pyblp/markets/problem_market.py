"""Market-level BLP problem functionality."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .market import Market
from .. import exceptions, options
from ..configurations.iteration import Iteration
from ..micro import Moments
from ..utilities.basics import Array, Bounds, Error, SolverStats, NumericalErrorHandler


class ProblemMarket(Market):
    """A market underlying the BLP problem."""

    @NumericalErrorHandler(exceptions.GenericNumericalError)
    def solve(
            self, delta: Array, last_delta: Array, last_tilde_costs: Array, moments: Moments, iteration: Iteration,
            fp_type: str, shares_bounds: Bounds, costs_bounds: Bounds, compute_jacobians: bool,
            compute_micro_covariances: bool) -> (
            Tuple[
                Array, Array, Array, Array, Array, Array, Array, Array, SolverStats, Array, Array, Array, List[Error]
            ]):
        """Compute the mean utility for this market that equates market shares to observed values by solving a fixed
        point problem. Then, if compute_jacobians is True, compute the Jacobian (holding beta fixed) of xi
        (equivalently, of delta) with respect to theta. Compute any contributions to micro moment values, their Jacobian
        with respect to theta (if compute_jacobians is True), and, if compute_micro_covariances is True, their
        covariance contributions. If there is a supply side, compute transformed marginal costs for this market. Then,
        if compute_jacobian is True, compute the Jacobian (holding gamma fixed) of omega (equivalently, of transformed
        marginal costs) with respect to theta.
        """
        errors: List[Error] = []

        # solve the contraction
        delta, clipped_shares, stats, delta_errors = self.safely_compute_delta(delta, iteration, fp_type, shares_bounds)
        errors.extend(delta_errors)

        # replace invalid values in delta with their last computed values
        valid_delta = delta.copy()
        bad_delta_index = ~np.isfinite(delta)
        valid_delta[bad_delta_index] = last_delta[bad_delta_index]

        # if needed, pre-compute probabilities
        probabilities = conditionals = None
        if compute_jacobians or moments.MM > 0 or self.K3 > 0:
            probabilities, conditionals = self.compute_probabilities(valid_delta)

        # if needed, pre-compute derivatives of probabilities with respect to parameters
        probabilities_tangent_mapping: Dict[int, Array] = {}
        conditionals_tangent_mapping: Dict[int, Array] = {}
        if compute_jacobians:
            assert probabilities is not None
            for p, parameter in enumerate(self.parameters.unfixed):
                probabilities_tangent, conditionals_tangent = self.compute_probabilities_by_parameter_tangent(
                    parameter, probabilities, conditionals, valid_delta
                )
                probabilities_tangent_mapping[p] = probabilities_tangent
                if self.K3 > 0:
                    conditionals_tangent_mapping[p] = conditionals_tangent

        # compute the Jacobian of xi (equivalently, of delta) with respect to theta
        xi_jacobian = np.full((self.J, self.parameters.P), np.nan, options.dtype)
        if compute_jacobians:
            assert probabilities is not None
            xi_jacobian, xi_jacobian_errors = self.safely_compute_xi_by_theta_jacobian(
                probabilities, conditionals, probabilities_tangent_mapping
            )
            errors.extend(xi_jacobian_errors)

        # if needed, pre-compute derivatives of probabilities with respect to xi
        probabilities_tensor = conditionals_tensor = None
        if compute_jacobians and (moments.MM > 0 or self.K3 > 0):
            assert probabilities is not None
            probabilities_tensor, conditionals_tensor = self.compute_probabilities_by_xi_tensor(
                probabilities, conditionals, compute_conditionals_tensor=self.K3 > 0
            )

        # compute contributions to micro moments, their Jacobian, and their covariances
        if moments.MM == 0:
            micro_numerator = np.zeros((moments.MM, 0), options.dtype)
            micro_denominator = np.zeros((moments.MM, 0), options.dtype)
            micro_numerator_jacobian = np.full((moments.MM, self.parameters.P), np.nan, options.dtype)
            micro_denominator_jacobian = np.full((moments.MM, self.parameters.P), np.nan, options.dtype)
            micro_covariances_numerator = np.full((moments.MM, moments.MM), np.nan, options.dtype)
        else:
            assert probabilities is not None
            (
                micro_numerator, micro_denominator, micro_numerator_jacobian, micro_denominator_jacobian,
                micro_covariances_numerator, micro_errors
            ) = (
                self.safely_compute_micro_contributions(
                    moments, valid_delta, probabilities, probabilities_tangent_mapping, probabilities_tensor,
                    xi_jacobian, compute_jacobians, compute_micro_covariances
                )
            )
            errors.extend(micro_errors)

        # only compute supply side contributions if there is a supply side
        if self.K3 == 0:
            tilde_costs = np.full((self.J, 0), np.nan, options.dtype)
            omega_jacobian = np.full((self.J, self.parameters.P), np.nan, options.dtype)
            clipped_costs = np.zeros((self.J, 1), np.bool_)
        else:
            assert probabilities is not None

            # compute transformed marginal costs
            tilde_costs, clipped_costs, tilde_costs_errors = self.safely_compute_tilde_costs(
                probabilities, conditionals, costs_bounds
            )
            errors.extend(tilde_costs_errors)

            # replace invalid transformed marginal costs with their last computed values
            valid_tilde_costs = tilde_costs.copy()
            bad_tilde_costs_index = ~np.isfinite(tilde_costs)
            valid_tilde_costs[bad_tilde_costs_index] = last_tilde_costs[bad_tilde_costs_index]

            # compute the Jacobian, which is zero for clipped marginal costs
            omega_jacobian = np.full((self.J, self.parameters.P), np.nan, options.dtype)
            if compute_jacobians:
                assert probabilities_tensor is not None
                omega_jacobian, omega_jacobian_errors = self.safely_compute_omega_by_theta_jacobian(
                    valid_tilde_costs, probabilities, conditionals, probabilities_tangent_mapping,
                    conditionals_tangent_mapping, probabilities_tensor, conditionals_tensor, xi_jacobian
                )
                errors.extend(omega_jacobian_errors)
                omega_jacobian[clipped_costs.flat] = 0

        return (
            delta, xi_jacobian, micro_numerator, micro_denominator, micro_numerator_jacobian,
            micro_denominator_jacobian, micro_covariances_numerator, clipped_shares, stats, tilde_costs, omega_jacobian,
            clipped_costs, errors
        )

    @NumericalErrorHandler(exceptions.DeltaNumericalError)
    def safely_compute_delta(
            self, initial_delta: Array, iteration: Iteration, fp_type: str, shares_bounds: Bounds) -> (
            Tuple[Array, Array, SolverStats, List[Error]]):
        """Compute the mean utility for this market that equates market shares to observed values by solving a fixed
        point problem, handling any numerical errors.
        """
        delta, clipped_shares, stats, errors = self.compute_delta(initial_delta, iteration, fp_type, shares_bounds)
        if not stats.converged:
            errors.append(exceptions.DeltaConvergenceError())
        return delta, clipped_shares, stats, errors

    @NumericalErrorHandler(exceptions.XiByThetaJacobianNumericalError)
    def safely_compute_xi_by_theta_jacobian(
            self, probabilities: Array, conditionals: Optional[Array],
            probabilities_tangent_mapping: Dict[int, Array]) -> Tuple[Array, List[Error]]:
        """Compute the Jacobian (holding beta fixed) of xi (equivalently, of delta) with respect to theta, handling any
        numerical errors.
        """
        return self.compute_xi_by_theta_jacobian(probabilities, conditionals, probabilities_tangent_mapping)

    @NumericalErrorHandler(exceptions.MicroMomentsNumericalError)
    def safely_compute_micro_contributions(
            self, moments: Moments, delta: Array, probabilities: Optional[Array],
            probabilities_tangent_mapping: Dict[int, Array], probabilities_tensor: Optional[Array], xi_jacobian: Array,
            compute_jacobians: bool, compute_covariances: bool) -> (
            Tuple[Array, Array, Array, Array, Array, List[Error]]):
        """Compute micro moment value contributions, handling any numerical errors."""
        errors: List[Error] = []
        (
            micro_numerator, micro_denominator, micro_numerator_jacobian, micro_denominator_jacobian,
            micro_covariances_numerator
        ) = (
            self.compute_micro_contributions(
                moments, delta, probabilities, probabilities_tangent_mapping, probabilities_tensor, xi_jacobian,
                compute_jacobians, compute_covariances
            )
        )
        return (
            micro_numerator, micro_denominator, micro_numerator_jacobian, micro_denominator_jacobian,
            micro_covariances_numerator, errors
        )

    @NumericalErrorHandler(exceptions.CostsNumericalError)
    def safely_compute_tilde_costs(
            self, probabilities: Array, conditionals: Optional[Array], costs_bounds: Bounds) -> (
            Tuple[Array, Array, List[Error]]):
        """Compute transformed marginal costs, handling any numerical errors."""
        errors: List[Error] = []

        # compute marginal costs
        eta, eta_errors = self.compute_eta(probabilities=probabilities, conditionals=conditionals)
        errors.extend(eta_errors)
        costs = self.products.prices - eta

        # clip marginal costs that are outside acceptable bounds
        clipped_costs = (costs < costs_bounds[0]) | (costs > costs_bounds[1])
        if clipped_costs.any():
            costs = np.clip(costs, *costs_bounds)

        # take the log of marginal costs under a log-linear specification
        if self.costs_type == 'linear':
            tilde_costs = costs
        else:
            assert self.costs_type == 'log'
            if np.any(costs <= 0):
                errors.append(exceptions.NonpositiveCostsError())
            with np.errstate(all='ignore'):
                tilde_costs = np.log(costs)

        return tilde_costs, clipped_costs, errors

    @NumericalErrorHandler(exceptions.OmegaByThetaJacobianNumericalError)
    def safely_compute_omega_by_theta_jacobian(
            self, tilde_costs: Array, probabilities: Array, conditionals: Optional[Array],
            probabilities_tangent_mapping: Dict[int, Array], conditionals_tangent_mapping: Dict[int, Optional[Array]],
            probabilities_tensor: Array, conditionals_tensor: Optional[Array], xi_jacobian: Array) -> (
            Tuple[Array, List[Error]]):
        """Compute the Jacobian (holding gamma fixed) of omega (equivalently, of transformed marginal costs) with
        respect to theta, handling any numerical errors.
        """
        return self.compute_omega_by_theta_jacobian(
            tilde_costs, probabilities, conditionals, probabilities_tangent_mapping, conditionals_tangent_mapping,
            probabilities_tensor, conditionals_tensor, xi_jacobian
        )
