"""Market-level simulation of synthetic BLP data."""

from typing import List, Tuple

import numpy as np

from .market import Market
from .. import exceptions, options
from ..configurations.iteration import Iteration
from ..utilities.basics import Array, Error, SolverStats, NumericalErrorHandler


class SimulationMarket(Market):
    """A market in a simulation of synthetic BLP data."""

    def compute_endogenous(
            self, costs: Array, prices: Array, iteration: Iteration) -> Tuple[Array, Array, SolverStats, List[Error]]:
        """Compute endogenous prices and shares"""
        errors: List[Error] = []
        prices, stats, price_errors = self.safely_compute_equilibrium_prices(costs, iteration, prices)
        shares, share_errors = self.safely_compute_shares(prices)
        errors.extend(price_errors + share_errors)
        return prices, shares, stats, errors

    def compute_exogenous(
            self, initial_delta: Array, iteration: Iteration, fp_type: str) -> (
            Tuple[Array, Array, SolverStats, List[Error]]):
        """Compute delta and transformed marginal costs, which map to the exogenous product characteristics."""
        errors: List[Error] = []
        delta, stats, delta_errors = self.safely_compute_delta(initial_delta, iteration, fp_type)
        errors.extend(delta_errors)
        tilde_costs = np.zeros((self.J, 0), options.dtype)
        if self.K3 > 0:
            tilde_costs, tilde_costs_errors = self.safely_compute_tilde_costs(delta)
            errors.extend(tilde_costs_errors)
        return delta, tilde_costs, stats, errors

    @NumericalErrorHandler(exceptions.SyntheticPricesNumericalError)
    def safely_compute_equilibrium_prices(
            self, costs: Array, iteration: Iteration, prices: Array) -> Tuple[Array, SolverStats, List[Error]]:
        """Compute equilibrium prices by iterating over the zeta-markup equation, handling any numerical errors."""
        errors: List[Error] = []
        prices, stats = self.compute_equilibrium_prices(costs, iteration, prices)
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
            self, initial_delta: Array, iteration: Iteration, fp_type: str) -> Tuple[Array, SolverStats, List[Error]]:
        """Compute delta, handling any numerical errors."""
        delta, stats, errors = self.compute_delta(initial_delta, iteration, fp_type)
        if not stats.converged:
            errors.append(exceptions.SyntheticDeltaConvergenceError())
        return delta, stats, errors

    @NumericalErrorHandler(exceptions.SyntheticCostsNumericalError)
    def safely_compute_tilde_costs(self, delta: Array) -> Tuple[Array, List[Error]]:
        """Compute transformed marginal costs, handling any numerical errors."""
        errors: List[Error] = []

        # compute marginal costs
        eta, eta_errors = self.compute_eta(delta=delta)
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
