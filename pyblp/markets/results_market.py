"""Market-level structuring of BLP results."""

from typing import List, Optional, Tuple

import numpy as np

from .market import Market
from .. import exceptions, options
from ..configurations.iteration import Iteration
from ..utilities.basics import Array, Error, SolverStats, numerical_error_handler


class ResultsMarket(Market):
    """A market in structured BLP results."""

    @numerical_error_handler(exceptions.EquilibriumRealizationFloatingPointError)
    def safely_solve_equilibrium_realization(
            self, costs: Array, prices: Optional[Array], iteration: Optional[Iteration]) -> (
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
            prices, stats = self.compute_equilibrium_prices(costs, iteration)
            if not stats.converged:
                errors.append(exceptions.EquilibriumPricesConvergenceError())

        # compute the associated mean utility and shares
        delta = self.update_delta_with_variable('prices', prices)
        mu = self.update_mu_with_variable('prices', prices)
        shares = self.compute_probabilities(delta, mu)[0] @ self.agents.weights
        return prices, shares, delta, stats, errors

    @numerical_error_handler(exceptions.XiByThetaJacobianRealizationFloatingPointError)
    def safely_compute_xi_by_theta_jacobian_realization(self) -> Tuple[Array, List[Error]]:
        """Compute the Jacobian of xi (equivalently, of delta) with respect to theta for a realization of the market,
        handling any numerical errors.
        """
        return self.compute_xi_by_theta_jacobian()

    @numerical_error_handler(exceptions.OmegaByThetaJacobianRealizationFloatingPointError)
    def safely_compute_omega_by_theta_jacobian_realization(
            self, tilde_costs: Array, xi_jacobian: Array, costs_type: str) -> Tuple[Array, List[Error]]:
        """Compute the Jacobian of omega (equivalently, of transformed marginal costs) with respect to theta for a
        realization of the market, handling any numerical errors.
        """
        return self.compute_omega_by_theta_jacobian(tilde_costs, xi_jacobian, costs_type)

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_compute_aggregate_elasticity(self, factor: float, name: str) -> Tuple[Array, List[Error]]:
        """Estimate the aggregate elasticity of demand with respect to a variable, handling any numerical errors."""
        errors: List[Error] = []
        scaled_variable = (1 + factor) * self.products[name]
        delta = self.update_delta_with_variable(name, scaled_variable)
        mu = self.update_mu_with_variable(name, scaled_variable)
        shares = self.compute_probabilities(delta, mu)[0] @ self.agents.weights
        aggregate_elasticities = (shares - self.products.shares).sum() / factor
        return aggregate_elasticities, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_compute_elasticities(self, name: str) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of elasticities of demand with respect to a variable, handling any numerical errors."""
        errors: List[Error] = []
        derivatives = self.compute_utility_derivatives(name)
        jacobian = self.compute_shares_by_variable_jacobian(derivatives)
        elasticities = jacobian * self.products[name].T / self.products.shares
        return elasticities, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_compute_diversion_ratios(self, name: str) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of diversion ratios with respect to a variable, handling any numerical errors."""
        errors: List[Error] = []
        derivatives = self.compute_utility_derivatives(name)
        jacobian = self.compute_shares_by_variable_jacobian(derivatives)

        # replace the diagonal with derivatives with respect to the outside option
        jacobian_diagonal = np.c_[jacobian.diagonal()]
        jacobian[np.diag_indices_from(jacobian)] = -jacobian.sum(axis=1)

        # compute the ratios
        ratios = -jacobian / np.tile(jacobian_diagonal, self.J)
        return ratios, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_compute_long_run_diversion_ratios(self) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of long-run diversion ratios, handling any numerical errors."""
        errors: List[Error] = []

        # compute share differences when products are excluded and store outside share differences on the diagonal
        changes = np.zeros((self.J, self.J), options.dtype)
        for j in range(self.J):
            shares_without_j = self.compute_probabilities(eliminate_product=j)[0] @ self.agents.weights
            changes[j] = (shares_without_j - self.products.shares).flat
            changes[j, j] = -changes[j].sum()

        # compute the ratios
        ratios = changes / np.tile(self.products.shares, self.J)
        return ratios, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_extract_diagonal(self, matrix: Array) -> Tuple[Array, List[Error]]:
        """Extract the diagonal from a matrix, handling any numerical errors."""
        errors: List[Error] = []
        diagonal = matrix[:, :self.J].diagonal()
        return diagonal, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_extract_diagonal_mean(self, matrix: Array) -> Tuple[Array, List[Error]]:
        """Extract the mean of the diagonal from a matrix, handling any numerical errors."""
        errors: List[Error] = []
        diagonal_mean = matrix[:, :self.J].diagonal().mean()
        return diagonal_mean, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_compute_costs(self) -> Tuple[Array, List[Error]]:
        """Estimate marginal costs, handling any numerical errors."""
        errors: List[Error] = []
        eta, eta_errors = self.compute_eta()
        errors.extend(eta_errors)
        costs = self.products.prices - eta
        return costs, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
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
        eta, eta_errors = self.compute_eta(ownership_matrix)
        errors.extend(eta_errors)
        prices = costs + eta
        return prices, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_compute_prices(
            self, iteration: Iteration, firm_ids: Optional[Array], ownership: Optional[Array], costs: Optional[Array],
            prices: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Estimate equilibrium prices. By default, use unchanged firm IDs, use unchanged prices as starting values,
        and compute marginal costs, handling any numerical errors.
        """
        errors: List[Error] = []
        ownership_matrix = self.get_ownership_matrix(firm_ids, ownership)
        if costs is None:
            costs, costs_errors = self.safely_compute_costs()
            errors.extend(costs_errors)
        prices, converged = self.compute_equilibrium_prices(costs, iteration, ownership_matrix, prices)[:2]
        if not converged:
            errors.append(exceptions.EquilibriumPricesConvergenceError())
        return prices, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_compute_shares(self, prices: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Estimate shares evaluated at specified prices. By default, use unchanged prices, handling any numerical
        errors.
        """
        errors: List[Error] = []
        if prices is None:
            prices = self.products.prices
        delta = self.update_delta_with_variable('prices', prices)
        mu = self.update_mu_with_variable('prices', prices)
        shares = self.compute_probabilities(delta, mu)[0] @ self.agents.weights
        return shares, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_compute_hhi(self, firm_ids: Optional[Array], shares: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Estimate HHI. By default, use unchanged firm IDs and shares, handling any numerical errors."""
        errors: List[Error] = []
        if firm_ids is None:
            firm_ids = self.products.firm_ids
        if shares is None:
            shares = self.products.shares
        hhi = 1e4 * sum((shares[firm_ids == f].sum() / shares.sum())**2 for f in np.unique(firm_ids))
        return hhi, errors

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
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

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
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

    @numerical_error_handler(exceptions.PostEstimationFloatingPointError)
    def safely_compute_consumer_surplus(self, prices: Optional[Array]) -> Tuple[Array, List[Error]]:
        """Estimate population-normalized consumer surplus. By default, use unchanged prices, handling any numerical
        errors.
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
        max_utilities = np.max(utilities, axis=0, keepdims=True)
        exp_utilities = np.exp(utilities - max_utilities)
        scale_weights = 1
        if self.H == 0:
            log_scale = -max_utilities
        else:
            exp_utilities = np.exp(np.log(self.groups.sum(exp_utilities)) * (1 - self.group_rho))
            min_rho = np.min(self.group_rho)
            log_scale = -max_utilities * (1 - min_rho)
            if self.rho_size > 1:
                scale_weights = np.exp(-max_utilities * (self.group_rho - min_rho))

        # compute the derivatives of utility with respect to prices, which are assumed to be constant across products
        derivatives = -self.compute_utility_derivatives('prices')[0]

        # compute consumer surplus
        numerator = np.log(np.exp(log_scale) + (scale_weights * exp_utilities).sum(axis=0, keepdims=True)) - log_scale
        consumer_surplus = (numerator / derivatives) @ self.agents.weights
        return consumer_surplus, errors
