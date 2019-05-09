"""Market level structuring of simulated synthetic BLP data."""

from typing import List, Tuple

from .market import Market
from .. import exceptions
from ..utilities.basics import Array, Error, numerical_error_handler


class SimulationResultsMarket(Market):
    """A market in a solved simulation of synthetic BLP data."""

    @numerical_error_handler(exceptions.SyntheticMicroMomentsFloatingPointError)
    def safely_compute_micro(self) -> Tuple[Array, List[Error]]:
        """Compute micro moments, handling any numerical errors."""
        errors: List[Error] = []
        micro = self.compute_micro()
        return micro, errors
