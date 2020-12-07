"""Market-level BLP problem functionality."""

from typing import Dict, List, Tuple

import numpy as np

from .market import Market
from .. import exceptions, options
from ..configurations.iteration import Iteration
from ..moments import (
    Moment, DemographicExpectationMoment, DemographicCovarianceMoment, DiversionProbabilityMoment,
    DiversionCovarianceMoment
)
from ..parameters import LinearCoefficient
from ..utilities.basics import Array, Bounds, Error, Optional, SolverStats, NumericalErrorHandler


class ProblemMarket(Market):
    """A market underlying the BLP problem."""

    def solve_demand(
            self, delta: Array, last_delta: Array, iteration: Iteration, fp_type: str, shares_bounds: Bounds,
            compute_jacobians: bool, compute_micro_covariances: bool) -> (
            Tuple[Array, Array, Array, Array, Array, Array, SolverStats, List[Error]]):
        """Compute the mean utility for this market that equates market shares to observed values by solving a fixed
        point problem. Then, if compute_jacobians is True, compute the Jacobian (holding beta fixed) of xi
        (equivalently, of delta) with respect to theta. Finally, compute any micro moment values, their Jacobian with
        respect to theta (if compute_jacobians is True), and, if compute_micro_covariances is True, their covariances.
        Replace null elements in delta with their last values before computing micro moments and Jacobians.
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

        # compute micro moments, their Jacobian, and their covariances
        assert self.moments is not None
        micro_values = np.zeros((self.moments.MM, 0), options.dtype)
        micro_jacobian = np.full((self.moments.MM, self.parameters.P), np.nan, options.dtype)
        micro_covariances = np.full((self.moments.MM, self.moments.MM), np.nan, options.dtype)
        if self.moments.MM > 0:
            (
                micro_values, probabilities, conditionals, inside_probabilities, inside_conditionals,
                eliminated_probabilities, eliminated_conditionals, inside_eliminated_sum,
                inside_eliminated_probabilities, inside_eliminated_conditionals, micro_errors
            ) = (
                self.safely_compute_micro_values(valid_delta)
            )
            errors.extend(micro_errors)
            if compute_jacobians:
                micro_jacobian, micro_jacobian_errors = self.safely_compute_micro_by_theta_jacobian(
                    valid_delta, probabilities, conditionals, inside_probabilities, inside_conditionals,
                    eliminated_probabilities, eliminated_conditionals, inside_eliminated_sum,
                    inside_eliminated_probabilities, inside_eliminated_conditionals, xi_jacobian
                )
                errors.extend(micro_jacobian_errors)
            if compute_micro_covariances:
                micro_covariances, micro_covariances_errors = self.safely_compute_micro_covariances(
                    valid_delta, probabilities, conditionals, inside_probabilities, eliminated_probabilities,
                    inside_eliminated_sum
                )
                errors.extend(micro_covariances_errors)

        return delta, micro_values, xi_jacobian, micro_jacobian, micro_covariances, clipped_shares, stats, errors

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
    def safely_compute_micro_values(
            self, delta: Array) -> (
            Tuple[
                Array, Optional[Array], Optional[Array], Optional[Array], Optional[Array], Dict[int, Array],
                Dict[int, Optional[Array]], Optional[Array], Dict[int, Array], Dict[int, Optional[Array]], List[Error]
            ]):
        """Compute micro moment values, handling any numerical errors."""
        errors: List[Error] = []
        (
            micro_values, probabilities, conditionals, inside_probabilities, inside_conditionals,
            eliminated_probabilities, eliminated_conditionals, inside_eliminated_sum, inside_eliminated_probabilities,
            inside_eliminated_conditionals
        ) = (
            self.compute_micro_values(delta)
        )
        return (
            micro_values, probabilities, conditionals, inside_probabilities, inside_conditionals,
            eliminated_probabilities, eliminated_conditionals, inside_eliminated_sum, inside_eliminated_probabilities,
            inside_eliminated_conditionals, errors
        )

    @NumericalErrorHandler(exceptions.MicroMomentsByThetaJacobianNumericalError)
    def safely_compute_micro_by_theta_jacobian(
            self, delta: Array, probabilities: Optional[Array], conditionals: Optional[Array],
            inside_probabilities: Optional[Array], inside_conditionals: Optional[Array],
            eliminated_probabilities: Dict[int, Array], eliminated_conditionals: Dict[int, Optional[Array]],
            inside_eliminated_sum: Optional[Array], inside_eliminated_probabilities: Dict[int, Array],
            inside_eliminated_conditionals: Dict[int, Optional[Array]], xi_jacobian: Array) -> (
            Tuple[Array, List[Error]]):
        """Compute the Jacobian of micro moments with respect to theta, handling any numerical errors."""
        errors: List[Error] = []
        assert self.moments is not None

        # pre-compute tensor derivatives of probabilities with respect to xi
        probabilities_tensor = None
        if probabilities is not None:
            probabilities_tensor, _ = self.compute_probabilities_by_xi_tensor(probabilities, conditionals)

        # pre-compute the same but for probabilities conditional on purchasing an inside good
        inside_tensor = None
        if inside_probabilities is not None:
            inside_tensor, _ = self.compute_probabilities_by_xi_tensor(
                inside_probabilities, inside_conditionals
            )

        # pre-compute the same but for second choice probabilities
        eliminated_tensors = {}
        for j in eliminated_probabilities:
            eliminated_tensors[j], _ = self.compute_probabilities_by_xi_tensor(
                eliminated_probabilities[j], eliminated_conditionals[j]
            )

        # pre-compute the same but for the sum of inside probability products over all first choices
        inside_eliminated_sum_tensor = None
        if inside_eliminated_probabilities:
            assert inside_probabilities is not None and inside_tensor is not None
            inside_eliminated_sum_tensor = np.zeros((self.J, self.J, self.I), options.dtype)
            for j in range(self.J):
                inside_eliminated_tensor_j, _ = self.compute_probabilities_by_xi_tensor(
                    inside_eliminated_probabilities[j], inside_eliminated_conditionals[j]
                )
                inside_eliminated_sum_tensor += inside_probabilities[[j]][None] * inside_eliminated_tensor_j

        # compute the Jacobian
        micro_jacobian = np.zeros((self.moments.MM, self.parameters.P))
        for p, parameter in enumerate(self.parameters.unfixed):
            if isinstance(parameter, LinearCoefficient):
                continue

            # pre-compute tangents of probabilities respect to the parameter
            probabilities_tangent = None
            if probabilities is not None:
                probabilities_tangent, _ = self.compute_probabilities_by_parameter_tangent(
                    parameter, probabilities, conditionals, delta
                )
                probabilities_tangent += (probabilities_tensor * xi_jacobian[:, [p], None]).sum(axis=0)

            # pre-compute the same but for probabilities conditional on purchasing an inside good
            inside_tangent = None
            if inside_probabilities is not None:
                inside_tangent, _ = self.compute_probabilities_by_parameter_tangent(
                    parameter, inside_probabilities, inside_conditionals, delta
                )
                inside_tangent += (inside_tensor * xi_jacobian[:, [p], None]).sum(axis=0)

            # compute the same but for second choice probabilities
            eliminated_tangents = {}
            for j in eliminated_probabilities:
                eliminated_tangents[j], _ = self.compute_probabilities_by_parameter_tangent(
                    parameter, eliminated_probabilities[j], eliminated_conditionals[j], delta
                )
                eliminated_tangents[j] += (eliminated_tensors[j] * xi_jacobian[:, [p], None]).sum(axis=0)

            # compute the same but for the sum of inside probability products over all first choices
            inside_eliminated_sum_tangent = None
            if inside_eliminated_probabilities:
                assert inside_probabilities is not None and inside_tangent is not None
                inside_eliminated_sum_tangent = (inside_eliminated_sum_tensor * xi_jacobian[:, [p], None]).sum(axis=0)
                for j in range(self.J):
                    inside_eliminated_tangent_j, _ = self.compute_probabilities_by_parameter_tangent(
                        parameter, inside_eliminated_probabilities[j], inside_eliminated_conditionals[j], delta
                    )
                    inside_eliminated_sum_tangent += (
                        inside_tangent[[j]] * inside_eliminated_probabilities[j] +
                        inside_probabilities[[j]] * inside_eliminated_tangent_j
                    )

            # fill the gradient of micro moments with respect to the parameter
            for m, moment in enumerate(self.moments.micro_moments):
                micro_jacobian[m, p] = self.agents.weights.T @ self.compute_agent_micro_tangent(
                    moment, probabilities_tangent, inside_probabilities, inside_tangent, eliminated_tangents,
                    inside_eliminated_sum, inside_eliminated_sum_tangent
                )

        return micro_jacobian, errors

    def compute_agent_micro_tangent(
            self, moment: Moment, probabilities_tangent: Optional[Array], inside_probabilities: Optional[Array],
            inside_tangent: Optional[Array], eliminated_tangents: Dict[int, Array],
            inside_eliminated_sum: Optional[Array], inside_eliminated_sum_tangent: Optional[Array]) -> (
            Tuple[Array, Array]):
        """Compute the tangent of agent-specific micro moments with respect to a parameter."""

        # handle a demographic expectation for agents who choose the outside good
        if isinstance(moment, DemographicExpectationMoment) and moment.product_id is None:
            assert probabilities_tangent is not None
            d = self.agents.demographics[:, [moment.demographics_index]]
            outside_probabilities_tangent = -probabilities_tangent.sum(axis=0, keepdims=True).T
            outside_share = 1 - self.products.shares.sum()
            return d * outside_probabilities_tangent / outside_share

        # handle a demographic expectation for agents who choose a certain inside good
        if isinstance(moment, DemographicExpectationMoment):
            assert probabilities_tangent is not None
            j = self.get_product(moment.product_id)
            d = self.agents.demographics[:, [moment.demographics_index]]
            return d * probabilities_tangent[[j]].T / self.products.shares[j]

        # handle a covariance between a product characteristic and a demographic
        if isinstance(moment, DemographicCovarianceMoment):
            assert inside_tangent is not None
            x = self.products.X2[:, [moment.X2_index]]
            d = self.agents.demographics[:, [moment.demographics_index]]
            z_tangent = inside_tangent.T @ x
            demeaned_z_tangent = z_tangent - self.agents.weights.T @ z_tangent
            demeaned_d = d - self.agents.weights.T @ d
            return demeaned_z_tangent * demeaned_d

        # handle the second choice probability of a certain inside good for agents who choose the outside good
        if isinstance(moment, DiversionProbabilityMoment) and moment.product_id1 is None:
            assert inside_tangent is not None
            k = self.get_product(moment.product_id2)
            outside_share = 1 - self.products.shares.sum()
            return inside_tangent[[k]].T / outside_share

        # handle the second choice probability of the outside good for agents who choose a certain inside good
        if isinstance(moment, DiversionProbabilityMoment) and moment.product_id2 is None:
            j = self.get_product(moment.product_id1)
            eliminated_outside_tangent = -eliminated_tangents[j].sum(axis=0, keepdims=True)
            return eliminated_outside_tangent.T / self.products.shares[j]

        # handle the second choice probability of a certain inside good for agents who choose a certain inside good
        if isinstance(moment, DiversionProbabilityMoment):
            j = self.get_product(moment.product_id1)
            k = self.get_product(moment.product_id2)
            return eliminated_tangents[j][[k]].T / self.products.shares[j]

        # handle a covariance between product characteristics of first and second choices
        assert isinstance(moment, DiversionCovarianceMoment)
        assert inside_probabilities is not None and inside_eliminated_sum is not None
        assert inside_tangent is not None and inside_eliminated_sum_tangent is not None
        x1 = self.products.X2[:, [moment.X2_index1]]
        x2 = self.products.X2[:, [moment.X2_index2]]
        z1 = inside_probabilities.T @ x1
        z1_tangent = inside_tangent.T @ x1
        z2 = inside_eliminated_sum.T @ x2
        z2_tangent = inside_eliminated_sum_tangent.T @ x2
        demeaned_z1 = z1 - self.agents.weights.T @ z1
        demeaned_z1_tangent = z1_tangent - self.agents.weights.T @ z1_tangent
        demeaned_z2 = z2 - self.agents.weights.T @ z2
        demeaned_z2_tangent = z2_tangent - self.agents.weights.T @ z2_tangent
        return demeaned_z1_tangent * demeaned_z2 + demeaned_z1 * demeaned_z2_tangent

    @NumericalErrorHandler(exceptions.MicroMomentCovariancesNumericalError)
    def safely_compute_micro_covariances(
            self, delta: Array, probabilities: Optional[Array], conditionals: Optional[Array],
            inside_probabilities: Optional[Array], eliminated_probabilities: Dict[int, Array],
            inside_eliminated_sum: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Compute micro moment covariances, handling any numerical errors."""
        errors: List[Error] = []
        assert self.moments is not None

        # fill a matrix of demeaned micro moments for each agent moment-by-moment
        demeaned_agent_micro = np.zeros((self.I, self.moments.MM), options.dtype)
        for m, moment in enumerate(self.moments.micro_moments):
            agent_micro_m = self.compute_agent_micro_values(
                moment, delta, probabilities, conditionals, inside_probabilities, eliminated_probabilities,
                inside_eliminated_sum
            )
            demeaned_agent_micro[:, [m]] = agent_micro_m - self.agents.weights.T @ agent_micro_m

        # compute the moment covariances, enforcing shape and symmetry
        micro_covariances = demeaned_agent_micro.T @ (self.agents.weights * demeaned_agent_micro)
        return np.c_[micro_covariances + micro_covariances.T] / 2, errors

    @NumericalErrorHandler(exceptions.CostsNumericalError)
    def safely_compute_tilde_costs(self, costs_bounds: Bounds) -> Tuple[Array, Array, List[Error]]:
        """Compute transformed marginal costs, handling any numerical errors."""
        errors: List[Error] = []

        # compute marginal costs
        eta, eta_errors = self.compute_eta()
        errors.extend(eta_errors)
        costs = self.products.prices - eta

        # clip marginal costs that are outside of acceptable bounds
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
