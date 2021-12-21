"""Market-level BLP problem functionality."""

from typing import List, Tuple

import numpy as np

from .market import Market
from .. import exceptions, options
from ..configurations.iteration import Iteration
from ..micro import Moments
from ..utilities.basics import Array, Bounds, Error, SolverStats, NumericalErrorHandler


class ProblemMarket(Market):
    """A market underlying the BLP problem."""

    def solve_demand(
            self, delta: Array, last_delta: Array, moments: Moments, iteration: Iteration, fp_type: str,
            shares_bounds: Bounds, compute_jacobians: bool, compute_micro_covariances: bool) -> (
            Tuple[Array, Array, Array, Array, Array, Array, Array, Array, SolverStats, List[Error]]):
        """Compute the mean utility for this market that equates market shares to observed values by solving a fixed
        point problem. Then, if compute_jacobians is True, compute the Jacobian (holding beta fixed) of xi
        (equivalently, of delta) with respect to theta. Finally, compute any contributions to micro moment values, their
        Jacobian with respect to theta (if compute_jacobians is True), and, if compute_micro_covariances is True, their
        covariances. Replace null elements in delta with their last values before computing micro moment contributions
        and Jacobians.
        """
        errors: List[Error] = []

        # solve the contraction
        delta, clipped_shares, stats, delta_errors = self.safely_compute_delta(delta, iteration, fp_type, shares_bounds)
        errors.extend(delta_errors)

        # replace invalid values in delta with their last computed values
        valid_delta = delta.copy()
        bad_delta_index = ~np.isfinite(delta)
        valid_delta[bad_delta_index] = last_delta[bad_delta_index]

        # compute the Jacobian
        xi_jacobian = np.full((self.J, self.parameters.P), np.nan, options.dtype)
        if compute_jacobians:
            xi_jacobian, xi_jacobian_errors = self.safely_compute_xi_by_theta_jacobian(valid_delta)
            errors.extend(xi_jacobian_errors)

        # compute contributions to micro moments, their Jacobian, and their covariances
        if moments.MM == 0:
            micro_numerator = np.zeros((moments.MM, 0), options.dtype)
            micro_denominator = np.zeros((moments.MM, 0), options.dtype)
            micro_numerator_jacobian = np.full((moments.MM, self.parameters.P), np.nan, options.dtype)
            micro_denominator_jacobian = np.full((moments.MM, self.parameters.P), np.nan, options.dtype)
            micro_covariances_numerator = np.full((moments.MM, moments.MM), np.nan, options.dtype)
        else:
            (
                micro_numerator, micro_denominator, micro_numerator_jacobian, micro_denominator_jacobian,
                micro_covariances_numerator, micro_errors
            ) = (
                self.safely_compute_micro_contributions(
                    moments, valid_delta, xi_jacobian, compute_jacobians, compute_micro_covariances
                )
            )
            errors.extend(micro_errors)

        return (
            delta, xi_jacobian, micro_numerator, micro_denominator, micro_numerator_jacobian,
            micro_denominator_jacobian, micro_covariances_numerator, clipped_shares, stats, errors
        )

    def solve_supply(
            self, last_tilde_costs: Array, xi_jacobian: Array, costs_bounds: Bounds, compute_jacobian: bool) -> (
            Tuple[Array, Array, Array, List[Error]]):
        """Compute transformed marginal costs for this market. Then, if compute_jacobian is True, compute the Jacobian
        (holding gamma fixed) of omega (equivalently, of transformed marginal costs) with respect to theta. Replace null
        elements in transformed marginal costs with their last values before computing their Jacobian.
        """
        errors: List[Error] = []

        # compute transformed marginal costs
        tilde_costs, clipped_costs, tilde_costs_errors = self.safely_compute_tilde_costs(costs_bounds)
        errors.extend(tilde_costs_errors)

        # replace invalid transformed marginal costs with their last computed values
        valid_tilde_costs = tilde_costs.copy()
        bad_tilde_costs_index = ~np.isfinite(tilde_costs)
        valid_tilde_costs[bad_tilde_costs_index] = last_tilde_costs[bad_tilde_costs_index]

        # compute the Jacobian, which is zero for clipped marginal costs
        omega_jacobian = np.full((self.J, self.parameters.P), np.nan, options.dtype)
        if compute_jacobian:
            omega_jacobian, omega_jacobian_errors = self.safely_compute_omega_by_theta_jacobian(
                valid_tilde_costs, xi_jacobian
            )
            errors.extend(omega_jacobian_errors)
            omega_jacobian[clipped_costs.flat] = 0

        return tilde_costs, omega_jacobian, clipped_costs, errors

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
    def safely_compute_xi_by_theta_jacobian(self, delta: Array) -> Tuple[Array, List[Error]]:
        """Compute the Jacobian (holding beta fixed) of xi (equivalently, of delta) with respect to theta, handling any
        numerical errors.
        """
        return self.compute_xi_by_theta_jacobian(delta)

    @NumericalErrorHandler(exceptions.MicroMomentsNumericalError)
    def safely_compute_micro_contributions(
            self, moments: Moments, delta: Array, xi_jacobian: Array, compute_jacobians: bool,
            compute_covariances: bool) -> Tuple[Array, Array, Array, Array, Array, List[Error]]:
        """Compute micro moment value contributions, handling any numerical errors."""
        errors: List[Error] = []
        (
            micro_numerator, micro_denominator, micro_numerator_jacobian, micro_denominator_jacobian,
            micro_covariances_numerator
        ) = (
            self.compute_micro_contributions(moments, delta, xi_jacobian, compute_jacobians, compute_covariances)
        )
        return (
            micro_numerator, micro_denominator, micro_numerator_jacobian, micro_denominator_jacobian,
            micro_covariances_numerator, errors
        )

    @NumericalErrorHandler(exceptions.CostsNumericalError)
    def safely_compute_tilde_costs(self, costs_bounds: Bounds) -> Tuple[Array, Array, List[Error]]:
        """Compute transformed marginal costs, handling any numerical errors."""
        errors: List[Error] = []

        # compute marginal costs
        eta, eta_errors = self.compute_eta()
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
            self, tilde_costs: Array, xi_jacobian: Array) -> Tuple[Array, List[Error]]:
        """Compute the Jacobian (holding gamma fixed) of omega (equivalently, of transformed marginal costs) with
        respect to theta, handling any numerical errors.
        """
        return self.compute_omega_by_theta_jacobian(tilde_costs, xi_jacobian)
