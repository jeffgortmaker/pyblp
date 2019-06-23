"""Market-level simulation of synthetic BLP data."""

from typing import List, Tuple

from .market import Market
from .. import exceptions
from ..configurations.iteration import Iteration
from ..utilities.basics import Array, Error, SolverStats, NumericalErrorHandler


class SimulationMarket(Market):
    """A market in a simulation of synthetic BLP data."""

    def compute_prices_and_shares(
            self, costs: Array, prices: Array, iteration: Iteration) -> Tuple[Array, Array, SolverStats, List[Error]]:
        """Compute endogenous prices and shares"""
        errors: List[Error] = []
        prices, stats, price_errors = self.safely_compute_equilibrium_prices(costs, iteration, prices)
        shares, share_errors = self.safely_compute_shares(prices)
        errors.extend(price_errors + share_errors)
        return prices, shares, stats, errors

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
