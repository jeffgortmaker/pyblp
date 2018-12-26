"""Market-level simulation of synthetic BLP data."""

from typing import List, Optional, Tuple

import numpy as np

from .market import Market
from .. import exceptions
from ..configurations.iteration import Iteration
from ..utilities.basics import Array, Error


class SimulationMarket(Market):
    """A market in a simulation of synthetic BLP data."""

    def solve(
            self, firm_ids: Optional[Array], ownership: Optional[Array], costs: Array, prices: Array,
            iteration: Iteration) -> Tuple[Array, Array, List[Error], bool, int, int]:
        """Solve for synthetic prices and shares. By default, use unchanged firm IDs."""
        errors: List[Error] = []
        ownership_matrix = self.get_ownership_matrix(firm_ids, ownership)

        # configure NumPy to identify floating point errors
        with np.errstate(divide='call', over='call', under='ignore', invalid='call'):
            np.seterrcall(lambda *_: errors.append(exceptions.SyntheticPricesFloatingPointError()))

            # solve the fixed point problem
            prices, converged, iterations, evaluations = self.compute_equilibrium_prices(
                costs, iteration, ownership_matrix, prices
            )
            if not converged:
                errors.append(exceptions.SyntheticPricesConvergenceError())

            # switch to identifying floating point errors with synthetic share computation
            np.seterrcall(lambda *_: errors.append(exceptions.SyntheticSharesFloatingPointError()))

            # compute the associated shares
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
            shares = self.compute_probabilities(delta, mu)[0] @ self.agents.weights
            return prices, shares, errors, converged, iterations, evaluations
