"""Market level structuring of simulated synthetic BLP data."""

from typing import List, Tuple

from .market import Market
from .. import exceptions
from ..utilities.basics import Array, Error, NumericalErrorHandler


class SimulationResultsMarket(Market):
    """A market in a solved simulation of synthetic BLP data."""

    @NumericalErrorHandler(exceptions.SyntheticMicroMomentsNumericalError)
    def safely_compute_micro(self) -> Tuple[Array, List[Error]]:
        """Compute micro moments, handling any numerical errors."""
        errors: List[Error] = []
        micro = self.compute_micro()
        return micro, errors
