"""Market-level simulation of synthetic BLP data."""

from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from .market import Market
from .. import exceptions, options
from ..configurations.iteration import Iteration
from ..utilities.basics import Array, Bounds, Error, SolverStats, NumericalErrorHandler


class SimulationMarket(Market):
    """A market in a simulation of synthetic BLP data."""

    def compute_endogenous(
            self, costs: Array, prices: Array, iteration: Iteration, constant_costs: bool, compute_gradients: bool,
            compute_hessians: bool) -> (
            Tuple[
                Array, Array, Array, Array, SolverStats, Optional[Dict[Hashable, Array]],
                Optional[Dict[Hashable, Array]], List[Error]
            ]):
        """Compute endogenous prices and shares, along with the associated delta and costs. Optionally compute firms'
        profit gradients and Hessians.
        """
        errors: List[Error] = []
        prices, stats, price_errors = self.safely_compute_equilibrium_prices(costs, iteration, constant_costs, prices)
        shares, share_errors = self.safely_compute_shares(prices)
        errors.extend(price_errors + share_errors)

        # update mean utilities and marginal costs
        with np.errstate(all='ignore'):
            delta = self.update_delta_with_variable('prices', prices)
            if not constant_costs:
                costs = self.update_costs_with_variable(costs, 'shares', shares)

            # optionally compute profit gradients
            profit_gradients = None
            if compute_gradients:
                profit_gradients = {}
                ownership = self.get_ownership_matrix()
                jacobian = self.compute_profit_jacobian(costs, prices)
                firm_profit_gradient = (ownership * jacobian).sum(axis=0)
                for firm_id in np.unique(self.products.firm_ids.flatten()):
                    firm_index = self.products.firm_ids.flat == firm_id
                    profit_gradients[firm_id] = firm_profit_gradient[firm_index]

            # optionally compute profit Hessians
            profit_hessians = None
            if compute_hessians:
                profit_hessians = {}
                ownership = self.get_ownership_matrix()
                hessian = self.compute_profit_hessian(costs, prices)
                firm_profit_hessian = (ownership[..., None] * hessian).sum(axis=0)
                for firm_id in np.unique(self.products.firm_ids.flatten()):
                    firm_index = self.products.firm_ids.flat == firm_id
                    profit_hessians[firm_id] = firm_profit_hessian[firm_index][:, firm_index]

        return prices, shares, delta, costs, stats, profit_gradients, profit_hessians, errors

    def compute_exogenous(
            self, initial_delta: Array, iteration: Iteration, fp_type: str, shares_bounds: Bounds) -> (
            Tuple[Array, Array, SolverStats, List[Error]]):
        """Compute delta and transformed marginal costs, which map to the exogenous product characteristics."""
        errors: List[Error] = []
        delta, stats, delta_errors = self.safely_compute_delta(initial_delta, iteration, fp_type, shares_bounds)
        errors.extend(delta_errors)
        tilde_costs = np.zeros((self.J, 0), options.dtype)
        if self.K3 > 0:
            tilde_costs, tilde_costs_errors = self.safely_compute_tilde_costs(delta)
            errors.extend(tilde_costs_errors)
        return delta, tilde_costs, stats, errors

    @NumericalErrorHandler(exceptions.SyntheticPricesNumericalError)
    def safely_compute_equilibrium_prices(
            self, costs: Array, iteration: Iteration, constant_costs: bool, prices: Array) -> (
            Tuple[Array, SolverStats, List[Error]]):
        """Compute equilibrium prices by iterating over the zeta-markup equation, handling any numerical errors."""
        errors: List[Error] = []
        prices, stats = self.compute_equilibrium_prices(costs, iteration, constant_costs, prices)
        if not stats.converged:
            errors.append(exceptions.SyntheticPricesConvergenceError())
        return prices, stats, errors

    @NumericalErrorHandler(exceptions.SyntheticSharesNumericalError)
    def safely_compute_shares(self, prices: Array) -> Tuple[Array, List[Error]]:
        """Compute equilibrium shares associated with prices, handling any numerical errors."""
        errors: List[Error] = []
        shares = self.compute_shares(prices)
        return shares, errors

    @NumericalErrorHandler(exceptions.SyntheticDeltaNumericalError)
    def safely_compute_delta(
            self, initial_delta: Array, iteration: Iteration, fp_type: str, shares_bounds: Bounds) -> (
            Tuple[Array, SolverStats, List[Error]]):
        """Compute delta, handling any numerical errors."""
        delta, clipped_shares, stats, errors = self.compute_delta(initial_delta, iteration, fp_type, shares_bounds)
        if clipped_shares.any():
            errors.append(exceptions.ClippedSharesError())
        if not stats.converged:
            errors.append(exceptions.SyntheticDeltaConvergenceError())
        return delta, stats, errors

    @NumericalErrorHandler(exceptions.SyntheticCostsNumericalError)
    def safely_compute_tilde_costs(self, delta: Array) -> Tuple[Array, List[Error]]:
        """Compute transformed marginal costs, handling any numerical errors."""
        errors: List[Error] = []

        # compute marginal costs
        probabilities, conditionals = self.compute_probabilities(delta)
        eta, _, eta_errors = self.compute_eta(probabilities=probabilities, conditionals=conditionals)
        errors.extend(eta_errors)
        costs = self.products.prices - eta

        # take the log of marginal costs under a log-linear specification
        if self.costs_type == 'linear':
            tilde_costs = costs
        else:
            assert self.costs_type == 'log'
            if np.any(costs <= 0):
                errors.append(exceptions.NonpositiveSyntheticCostsError())
            with np.errstate(all='ignore'):
                tilde_costs = np.log(costs)

        return tilde_costs, errors
