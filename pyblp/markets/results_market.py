"""Market-level structuring of BLP results."""

from typing import Any, List, Optional, Tuple

import numpy as np

from .market import Market
from .. import exceptions, options
from ..configurations.iteration import Iteration
from ..utilities.algebra import approximately_invert
from ..utilities.basics import Array, Bounds, Error, SolverStats, NumericalErrorHandler


class ResultsMarket(Market):
    """A market in structured BLP results."""

    @NumericalErrorHandler(exceptions.EquilibriumRealizationNumericalError)
    def safely_solve_equilibrium_realization(
            self, costs: Array, prices: Optional[Array], iteration: Optional[Iteration], constant_costs: bool) -> (
            Tuple[Array, Array, Array, SolverStats, List[Error]]):
        """If not already estimated, compute equilibrium prices along with the associated mean utility and shares for a
        realization of marginal costs (for bootstrapping and optimal instruments), along with parameters (for
        bootstrapping only), handling any numerical errors.
        """
        errors: List[Error] = []

        # solve the fixed point problem if prices haven't already been estimated
        if iteration is None:
            assert prices is not None
            stats = SolverStats()
        else:
            prices, stats = self.compute_equilibrium_prices(costs, iteration, constant_costs)
            if not stats.converged:
                errors.append(exceptions.EquilibriumPricesConvergenceError())

        # compute the associated mean utility and shares
        delta = self.update_delta_with_variable('prices', prices)
        mu = self.update_mu_with_variable('prices', prices)
        shares = self.compute_probabilities(delta, mu)[0] @ self.agents.weights
        return prices, shares, delta, stats, errors

    @NumericalErrorHandler(exceptions.JacobianRealizationNumericalError)
    def safely_compute_jacobian_realizations(self, tilde_costs: Array) -> Tuple[Array, Array, List[Error]]:
        """Compute the Jacobian (holding beta fixed) of xi (equivalently, of delta) with respect to theta for a
        realization of the market, handling any numerical errors. If there is a supply side, do the same for omega.
        """
        errors: List[Error] = []

        # pre-compute probabilities and their derivatives with respect to parameters
        probabilities, conditionals = self.compute_probabilities()
        probabilities_tangent_mapping, conditionals_tangent_mapping = (
            self.compute_probabilities_by_parameter_tangent_mapping(
                probabilities, conditionals, keep_conditionals=self.K3 > 0
            )
        )

        # compute the demand-side Jacobian
        xi_jacobian, xi_jacobian_errors = self.compute_xi_by_theta_jacobian(
            probabilities, conditionals, probabilities_tangent_mapping
        )
        errors.extend(xi_jacobian_errors)

        # compute the supply-side Jacobian if there is a supply side
        if self.K3 == 0:
            omega_jacobian = np.full((self.J, self.parameters.P), np.nan, options.dtype)
        else:
            # adjust for the contribution of xi's dependence on theta
            self.update_probabilities_by_parameter_tangent_mapping(
                probabilities_tangent_mapping, conditionals_tangent_mapping, probabilities, conditionals, xi_jacobian
            )

            # compute the supply-side Jacobian
            eta, capital_delta_inverse, eta_errors = self.compute_eta(
                probabilities=probabilities, conditionals=conditionals, keep_capital_delta_inverse=True
            )
            errors.extend(eta_errors)
            omega_jacobian = self.compute_omega_by_theta_jacobian(
                tilde_costs, eta, capital_delta_inverse, probabilities, conditionals, probabilities_tangent_mapping,
                conditionals_tangent_mapping,
            )

        return xi_jacobian, omega_jacobian, errors

    @NumericalErrorHandler(exceptions.DeltaNumericalError)
    def safely_compute_delta(
            self, iteration: Iteration, fp_type: str, shares_bounds: Bounds) -> Tuple[Array, List[Error]]:
        """Compute the mean utility for this market that equates market shares to observed values by solving a fixed
        point problem (starting at the delta with which this market was initialized), handling any numerical errors.
        """
        delta, clipped_shares, stats, errors = self.compute_delta(self.delta, iteration, fp_type, shares_bounds)
        if clipped_shares.any():
            errors.append(exceptions.ClippedSharesError())
        if not stats.converged:
            errors.append(exceptions.DeltaConvergenceError())
        return delta, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_aggregate_elasticity(self, factor: float, name: Optional[str]) -> Tuple[Array, List[Error]]:
        """Estimate the aggregate elasticity of demand with respect to a variable, handling any numerical errors."""
        errors: List[Error] = []
        if name is None:
            assert self.delta is not None
            delta = (1 + factor) * self.delta
            mu = self.mu
        else:
            scaled_variable = (1 + factor) * self.products[name]
            delta = self.update_delta_with_variable(name, scaled_variable)
            mu = self.update_mu_with_variable(name, scaled_variable)
        shares = self.compute_probabilities(delta, mu)[0] @ self.agents.weights
        aggregate_elasticities = (shares - self.products.shares).sum() / factor
        return aggregate_elasticities, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_elasticities(self, name: Optional[str]) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of elasticities of demand with respect to a variable, handling any numerical errors."""
        errors: List[Error] = []
        if name is None:
            assert self.delta is not None
            derivatives = np.ones((self.J, self.I), options.dtype)
            variable = self.delta
        else:
            derivatives = self.compute_utility_derivatives(name)
            variable = self.products[name]
        jacobian = self.compute_shares_by_variable_jacobian(derivatives)
        elasticities = jacobian * variable.T / self.products.shares
        return elasticities, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_demand_jacobian(self, name: str) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of derivatives of demand with respect to a variable, handling any numerical errors."""
        errors: List[Error] = []
        derivatives = self.compute_utility_derivatives(name)
        jacobian = self.compute_shares_by_variable_jacobian(derivatives)
        return jacobian, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_demand_hessian(self, name: str) -> Tuple[Array, List[Error]]:
        """Estimate second derivatives of demand with respect to a variable, handling any numerical errors."""
        errors: List[Error] = []
        derivatives = self.compute_utility_derivatives(name)
        second_derivatives = self.compute_utility_derivatives(name, order=2)
        hessian = self.compute_shares_by_variable_hessian(derivatives, second_derivatives)
        return hessian, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_diversion_ratios(self, name: Optional[str]) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of diversion ratios with respect to a variable, handling any numerical errors."""
        errors: List[Error] = []
        if name is None:
            derivatives = np.ones((self.J, self.I), options.dtype)
        else:
            derivatives = self.compute_utility_derivatives(name)
        jacobian = self.compute_shares_by_variable_jacobian(derivatives)
        jacobian_diagonal = np.c_[jacobian.diagonal()]
        np.fill_diagonal(jacobian, -jacobian.sum(axis=1))
        ratios = -jacobian / np.tile(jacobian_diagonal, self.J)
        return ratios, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_long_run_diversion_ratios(self) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of long-run diversion ratios, handling any numerical errors."""
        errors: List[Error] = []
        ratios = np.zeros((self.J, self.J), options.dtype)
        for j in range(self.J):
            shares_without_j = self.compute_probabilities(eliminate_product=j)[0] @ self.agents.weights
            ratios[j] = (shares_without_j - self.products.shares).flat / self.products.shares[j]
            ratios[j, j] = -ratios[j].sum()
        return ratios, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_probabilities(
            self, prices: Optional[Array] = None, delta: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of choice probabilities at specified prices. By default, use unchanged prices and mean
        utilities, handling any numerical errors.
        """
        errors: List[Error] = []
        if prices is None:
            mu = None
        else:
            mu = self.update_mu_with_variable('prices', prices)
            if delta is None:
                delta = self.update_delta_with_variable('prices', prices)

        probabilities = self.compute_probabilities(delta, mu)[0]
        return probabilities, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_extract_diagonal(self, matrix: Array) -> Tuple[Array, List[Error]]:
        """Extract the diagonal from a matrix, handling any numerical errors."""
        errors: List[Error] = []
        diagonal = matrix[:, :self.J].diagonal()
        return diagonal, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_extract_diagonal_mean(self, matrix: Array) -> Tuple[Array, List[Error]]:
        """Extract the mean of the diagonal from a matrix, handling any numerical errors."""
        errors: List[Error] = []
        diagonal_mean = matrix[:, :self.J].diagonal().mean()
        return diagonal_mean, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_costs(
            self, firm_ids: Optional[Array] = None, ownership: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Estimate marginal costs, handling any numerical errors."""
        errors: List[Error] = []
        ownership_matrix = self.get_ownership_matrix(firm_ids, ownership)
        eta, _, eta_errors = self.compute_eta(ownership_matrix)
        errors.extend(eta_errors)
        costs = self.products.prices - eta
        return costs, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_profit_hessian(
            self, prices: Optional[Array], costs: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Estimate second derivatives of profits with respect to prices. By default, use unchanged firm IDs and compute
        marginal costs, handling any numerical errors.
        """
        errors: List[Error] = []
        if costs is None:
            costs, costs_errors = self.safely_compute_costs()
            errors.extend(costs_errors)
        hessian = self.compute_profit_hessian(costs, prices)
        return hessian, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_passthrough(
            self, firm_ids: Optional[Array], ownership: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Estimate the passthrough matrix, handling any numerical errors."""
        errors: List[Error] = []

        # compute derivatives of shares with respect to prices
        probabilities, conditionals = self.compute_probabilities()
        utility_derivatives = self.compute_utility_derivatives('prices')
        utility_second_derivatives = self.compute_utility_derivatives('prices', order=2)
        jacobian = self.compute_shares_by_variable_jacobian(utility_derivatives, probabilities, conditionals)
        hessian = self.compute_shares_by_variable_hessian(
            utility_derivatives, utility_second_derivatives, probabilities, conditionals
        )

        # compute the capital delta matrix and its derivatives
        ownership_matrix = self.get_ownership_matrix(firm_ids, ownership)
        capital_delta = -ownership_matrix * jacobian
        capital_delta_derivatives = -ownership_matrix[..., None] * hessian

        # compute the inverse of capital delta
        capital_delta_inverse, replacement = approximately_invert(capital_delta)
        if replacement:
            errors.append(exceptions.IntraFirmJacobianInversionError(capital_delta, replacement))

        # compute the inverse of the passthrough matrix
        passthrough_inverse = np.zeros((self.J, self.J), options.dtype)
        for j in range(self.J):
            passthrough_inverse[:, [j]] = (
                capital_delta_inverse @ capital_delta_derivatives[..., j] @ capital_delta_inverse @ self.products.shares
            )

        passthrough_inverse += np.eye(self.J) - capital_delta_inverse @ jacobian

        # compute the passthrough matrix
        passthrough, replacement = approximately_invert(passthrough_inverse)
        if replacement:
            errors.append(exceptions.PassthroughInversionError(passthrough_inverse, replacement))

        return passthrough, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_approximate_equilibrium_prices(
            self, firm_ids: Optional[Array], ownership: Optional[Array], costs: Optional[Array]) -> (
            Tuple[Array, List[Error]]):
        """Estimate approximate equilibrium prices under the assumption that shares and their price derivatives are
        unaffected by firm ID changes. By default, use unchanged firm IDs and compute marginal costs, handling any
        numerical errors.
        """
        errors: List[Error] = []
        ownership_matrix = self.get_ownership_matrix(firm_ids, ownership)
        if costs is None:
            costs, costs_errors = self.safely_compute_costs()
            errors.extend(costs_errors)
        eta, _, eta_errors = self.compute_eta(ownership_matrix)
        errors.extend(eta_errors)
        prices = costs + eta
        return prices, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_prices(
            self, iteration: Iteration, constant_costs: bool, firm_ids: Optional[Array], ownership: Optional[Array],
            costs: Optional[Array], prices: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Estimate equilibrium prices. By default, use unchanged firm IDs, use unchanged prices as starting values,
        and compute marginal costs, handling any numerical errors.
        """
        errors: List[Error] = []
        ownership_matrix = self.get_ownership_matrix(firm_ids, ownership)
        if costs is None:
            costs, costs_errors = self.safely_compute_costs()
            errors.extend(costs_errors)
        prices, converged = self.compute_equilibrium_prices(costs, iteration, constant_costs, prices, ownership_matrix)
        if not converged:
            errors.append(exceptions.EquilibriumPricesConvergenceError())
        return prices, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_shares(self, prices: Optional[Array], delta: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Estimate shares evaluated at specified prices. By default, use unchanged prices and mean utilities, handling
        any numerical errors.
        """
        probabilities, errors = self.safely_compute_probabilities(prices, delta)
        shares = probabilities @ self.agents.weights
        return shares, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_hhi(self, firm_ids: Optional[Array], shares: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Estimate HHI. By default, use unchanged firm IDs and shares, handling any numerical errors."""
        errors: List[Error] = []
        if firm_ids is None:
            firm_ids = self.products.firm_ids
        if shares is None:
            shares = self.products.shares
        hhi = 1e4 * sum((shares[firm_ids == f].sum() / shares.sum())**2 for f in np.unique(firm_ids))
        return hhi, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_markups(self, prices: Optional[Array], costs: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Estimate markups. By default, use unchanged prices and compute marginal costs, handling any numerical
        errors.
        """
        errors: List[Error] = []
        if prices is None:
            prices = self.products.prices
        if costs is None:
            costs, costs_errors = self.safely_compute_costs()
            errors.extend(costs_errors)
        markups = (prices - costs) / prices
        return markups, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_profits(
            self, prices: Optional[Array], shares: Optional[Array], costs: Optional[Array]) -> (
            Tuple[Array, List[Error]]):
        """Estimate population-normalized gross expected profits. By default, use unchanged prices, use unchanged
        shares, and compute marginal costs, handling any numerical errors.
        """
        errors: List[Error] = []
        if prices is None:
            prices = self.products.prices
        if shares is None:
            shares = self.products.shares
        if costs is None:
            costs, costs_errors = self.safely_compute_costs()
            errors.extend(costs_errors)
        profits = (prices - costs) * shares
        return profits, errors

    @NumericalErrorHandler(exceptions.PostEstimationNumericalError)
    def safely_compute_consumer_surplus(
            self, keep_all: bool, eliminate_product_ids: Optional[Any], prices: Optional[Array]) -> (
            Tuple[Array, List[Error]]):
        """Estimate population-normalized consumer surplus or keep all individual-level surpluses. By default, use
        unchanged prices, handling any numerical errors.
        """
        errors: List[Error] = []
        if prices is None:
            delta = self.delta
            mu = self.mu
        else:
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
        if self.K2 == 0:
            mu = 0

        # compute the exponentiated utilities that will be summed in the expression for consumer surplus (re-scale for
        #   robustness to overflow)
        utilities = delta + mu
        if self.H > 0:
            utilities /= 1 - self.rho
        utility_reduction = np.clip(utilities.max(axis=0, keepdims=True), 0, None)
        exp_utilities = np.exp(utilities - utility_reduction)
        scale_weights = 1

        # eliminate any products from the choice set
        if eliminate_product_ids is not None:
            for j, product_id in enumerate(self.products.product_ids):
                if product_id in eliminate_product_ids:
                    exp_utilities[j] = 0

        # handle nesting
        if self.H == 0:
            log_scale = -utility_reduction
        else:
            exp_utilities = np.exp(np.log(self.groups.sum(exp_utilities)) * (1 - self.group_rho))
            min_rho = np.min(self.group_rho)
            log_scale = -utility_reduction * (1 - min_rho)
            if self.rho_size > 1:
                scale_weights = np.exp(-utility_reduction * (self.group_rho - min_rho))

        # compute the derivatives of utility with respect to prices, which are assumed to be constant across products
        derivatives = -self.compute_utility_derivatives('prices')[0]

        # compute individual-level consumer surpluses
        numerator = np.log(np.exp(log_scale) + (scale_weights * exp_utilities).sum(axis=0, keepdims=True)) - log_scale
        surpluses = numerator / derivatives
        if keep_all:
            return surpluses, errors

        # integrate over agents
        surplus = surpluses @ self.agents.weights
        return surplus, errors
