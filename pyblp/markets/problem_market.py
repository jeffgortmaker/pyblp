"""Market-level BLP problem functionality."""

from typing import Dict, List, Tuple

import numpy as np

from .market import Market
from .. import exceptions, options
from ..configurations.iteration import Iteration
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
                micro_values, probabilities, conditionals, inside_probabilities, eliminated_probabilities,
                inside_to_inside_ratios, inside_to_eliminated_probabilities, inside_eliminated_probabilities,
                micro_errors
            ) = (
                self.safely_compute_micro_values(valid_delta)
            )
            errors.extend(micro_errors)
            if compute_jacobians:
                micro_jacobian, micro_jacobian_errors = self.safely_compute_micro_by_theta_jacobian(
                    valid_delta, probabilities, conditionals, inside_probabilities, eliminated_probabilities,
                    inside_to_inside_ratios, inside_to_eliminated_probabilities, inside_eliminated_probabilities,
                    xi_jacobian
                )
                errors.extend(micro_jacobian_errors)
            if compute_micro_covariances:
                micro_covariances, micro_covariances_errors = self.safely_compute_micro_covariances(
                    valid_delta, probabilities, inside_probabilities, eliminated_probabilities, inside_to_inside_ratios,
                    inside_to_eliminated_probabilities
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
                Array, Array, Optional[Array], Optional[Array], Dict[int, Array], Optional[Array], Optional[Array],
                Dict[int, Array], List[Error]
            ]):
        """Compute micro moment values, handling any numerical errors."""
        errors: List[Error] = []
        (
            micro_values, probabilities, conditionals, inside_probabilities, eliminated_probabilities,
            inside_to_inside_ratios, inside_to_eliminated_probabilities, inside_eliminated_probabilities
        ) = (
            self.compute_micro_values(delta)
        )
        return (
            micro_values, probabilities, conditionals, inside_probabilities, eliminated_probabilities,
            inside_to_inside_ratios, inside_to_eliminated_probabilities, inside_eliminated_probabilities, errors
        )

    @NumericalErrorHandler(exceptions.MicroMomentsByThetaJacobianNumericalError)
    def safely_compute_micro_by_theta_jacobian(
            self, delta: Array, probabilities: Array, conditionals: Optional[Array],
            inside_probabilities: Optional[Array], eliminated_probabilities: Dict[int, Array],
            inside_to_inside_ratios: Optional[Array], inside_to_eliminated_probabilities: Optional[Array],
            inside_eliminated_probabilities: Dict[int, Array], xi_jacobian: Array) -> Tuple[Array, List[Error]]:
        """Compute the Jacobian of micro moments with respect to theta, handling any numerical errors."""
        errors: List[Error] = []
        assert self.moments is not None

        # pre-compute tensor derivatives of probabilities with respect to xi
        probabilities_tensor, _ = self.compute_probabilities_by_xi_tensor(probabilities, conditionals)

        # pre-compute ratios of eliminated to standard probabilities, replacing problematic elements with zeros so that
        #   the associated derivatives will be zero
        with np.errstate(all='ignore'):
            # do this for probabilities conditional on purchasing an inside good
            inside_ratio = None
            if inside_probabilities is not None:
                inside_ratio = inside_probabilities / probabilities
                inside_ratio[~np.isfinite(inside_ratio)] = 0

            # do this for second choice probabilities
            eliminated_ratios = {}
            for j in eliminated_probabilities:
                eliminated_ratios[j] = eliminated_probabilities[j] / probabilities
                eliminated_ratios[j][~np.isfinite(eliminated_ratios[j])] = 0

            # do the same for probabilities of purchasing any inside good first and a specific inside good second
            inside_eliminated_ratios = {}
            if inside_eliminated_probabilities:
                for j in range(self.J):
                    inside_eliminated_ratios[j] = inside_eliminated_probabilities[j] / probabilities
                    inside_eliminated_ratios[j][~np.isfinite(inside_eliminated_ratios[j])] = 0

        # compute the Jacobian
        micro_jacobian = np.zeros((self.moments.MM, self.parameters.P))
        for p, parameter in enumerate(self.parameters.unfixed):
            if isinstance(parameter, LinearCoefficient):
                continue

            # pre-compute tangents of probabilities respect to the parameter
            probabilities_tangent, _ = self.compute_probabilities_by_parameter_tangent(
                parameter, probabilities, conditionals, delta
            )
            probabilities_tangent += (probabilities_tensor * xi_jacobian[:, [p], None]).sum(axis=0)

            # pre-compute the same but for probabilities conditional on purchasing an inside good
            inside_tangent = None
            if inside_probabilities is not None:
                inside_tangent = inside_ratio * (
                    probabilities_tangent -
                    inside_probabilities * probabilities_tangent.sum(axis=0, keepdims=True)
                )

            # pre-compute the same but for second choice probabilities
            eliminated_tangents = {}
            for j in eliminated_probabilities:
                eliminated_tangents[j] = eliminated_ratios[j] * (
                    probabilities_tangent +
                    eliminated_probabilities[j] * probabilities_tangent[[j]]
                )

            # pre-compute the same but for the ratio of the probability each agent's first and second choices are both
            #   inside goods to the corresponding aggregate share
            inside_to_inside_tangent = None
            if inside_to_inside_ratios is not None:
                inside_to_inside_probabilities = np.zeros((self.I, 1), options.dtype)
                inside_to_inside_probabilities_tangent = np.zeros((self.I, 1), options.dtype)
                for j in range(self.J):
                    j_to_inside_probabilities = eliminated_probabilities[j].sum(axis=0, keepdims=True).T
                    j_to_inside_tangent = eliminated_tangents[j].sum(axis=0, keepdims=True).T
                    inside_to_inside_probabilities += probabilities[[j]].T * j_to_inside_probabilities
                    inside_to_inside_probabilities_tangent += (
                        probabilities_tangent[[j]].T * j_to_inside_probabilities +
                        probabilities[[j]].T * j_to_inside_tangent
                    )

                inside_to_inside_share = self.agents.weights.T @ inside_to_inside_probabilities
                inside_to_inside_share_tangent = self.agents.weights.T @ inside_to_inside_probabilities_tangent
                inside_to_inside_tangent = inside_to_inside_ratios * (
                    inside_to_inside_probabilities_tangent / inside_to_inside_probabilities -
                    inside_to_inside_share_tangent / inside_to_inside_share
                )

            # pre-compute the same but for probabilities of purchasing any inside good first and a specific inside good
            #   second
            inside_to_eliminated_tangent = None
            if inside_eliminated_probabilities:
                assert inside_probabilities is not None and inside_tangent is not None
                inside_to_eliminated_tangent = np.zeros((self.J, self.I), options.dtype)
                probabilities_tangent_sum = probabilities_tangent.sum(axis=0, keepdims=True)
                for j in range(self.J):
                    inside_eliminated_tangent = inside_eliminated_ratios[j] * (
                        probabilities_tangent -
                        inside_eliminated_probabilities[j] * (probabilities_tangent_sum - probabilities_tangent[j])
                    )
                    inside_to_eliminated_tangent += (
                        inside_tangent[[j]] * inside_eliminated_probabilities[j] +
                        inside_probabilities[[j]] * inside_eliminated_tangent
                    )

            # fill the gradient of micro moments with respect to the parameter
            for m, moment in enumerate(self.moments.micro_moments):
                micro_jacobian[m, p] = self.agents.weights.T @ moment._compute_agent_values_tangent(
                    self, p, delta, probabilities, probabilities_tangent, inside_probabilities, inside_tangent,
                    eliminated_probabilities, eliminated_tangents, inside_to_inside_ratios, inside_to_inside_tangent,
                    inside_to_eliminated_probabilities, inside_to_eliminated_tangent
                )

        return micro_jacobian, errors

    @NumericalErrorHandler(exceptions.MicroMomentCovariancesNumericalError)
    def safely_compute_micro_covariances(
            self, delta: Array, probabilities: Array, inside_probabilities: Optional[Array],
            eliminated_probabilities: Dict[int, Array], inside_to_inside_ratios: Optional[Array],
            inside_to_eliminated_probabilities: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Compute micro moment covariances, handling any numerical errors."""
        errors: List[Error] = []
        assert self.moments is not None

        # fill a matrix of demeaned micro moments for each agent moment-by-moment
        demeaned_agent_micro = np.zeros((self.I, self.moments.MM), options.dtype)
        for m, moment in enumerate(self.moments.micro_moments):
            agent_micro_m = moment._compute_agent_values(
                self, delta, probabilities, inside_probabilities, eliminated_probabilities, inside_to_inside_ratios,
                inside_to_eliminated_probabilities
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
