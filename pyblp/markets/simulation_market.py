"""Market-level simulation of synthetic BLP data."""

from typing import List, Optional, Tuple

from .market import Market
from .. import exceptions
from ..configurations.iteration import Iteration
from ..utilities.basics import Array, Error, SolverStats, numerical_error_handler


class SimulationMarket(Market):
    """A market in a simulation of synthetic BLP data."""

    def solve(
            self, firm_ids: Optional[Array], ownership: Optional[Array], costs: Array, prices: Array,
            iteration: Iteration) -> Tuple[Array, Array, SolverStats, List[Error]]:
        """Solve for synthetic prices and shares. By default, use unchanged firm IDs."""
        errors: List[Error] = []
        ownership_matrix = self.get_ownership_matrix(firm_ids, ownership)
        prices, stats, price_errors = self.safely_compute_equilibrium_prices(costs, iteration, ownership_matrix, prices)
        shares, share_errors = self.safely_compute_shares(prices)
        errors.extend(price_errors + share_errors)
        return prices, shares, stats, errors

    @numerical_error_handler(exceptions.SyntheticPricesFloatingPointError)
    def safely_compute_equilibrium_prices(
            self, costs: Array, iteration: Iteration, ownership_matrix: Array, prices: Array) -> (
            Tuple[Array, SolverStats, List[Error]]):
        """Compute equilibrium prices by iterating over the zeta-markup equation, handling any numerical errors."""
        errors: List[Error] = []
        prices, stats = self.compute_equilibrium_prices(costs, iteration, ownership_matrix, prices)
        if not stats.converged:
            errors.append(exceptions.SyntheticPricesConvergenceError())
        return prices, stats, errors

    @numerical_error_handler(exceptions.SyntheticSharesFloatingPointError)
    def safely_compute_shares(self, prices: Array) -> Tuple[Array, List[Error]]:
        """Compute equilibrium shares associated with prices, handling any numerical errors."""
        errors: List[Error] = []
        shares = self.compute_shares(prices)
        return shares, errors
