"""Market level structuring of simulated synthetic BLP data."""

from typing import List, Tuple

from .market import Market
from .. import exceptions
from ..micro import MicroDataset, Moments
from ..utilities.basics import Array, Error, NumericalErrorHandler


class SimulationResultsMarket(Market):
    """A market in a solved simulation of synthetic BLP data."""

    @NumericalErrorHandler(exceptions.SyntheticMicroDataNumericalError)
    def safely_compute_micro_weights(self, dataset: MicroDataset) -> Tuple[Array, List[Error]]:
        """Compute probabilities needed for simulating micro data, handling any numerical errors."""
        errors: List[Error] = []
        weights_mapping, _, _, _ = self.compute_micro_dataset_contributions([dataset])
        return weights_mapping[dataset], errors

    @NumericalErrorHandler(exceptions.SyntheticMicroMomentsNumericalError)
    def safely_compute_micro_contributions(self, moments: Moments) -> Tuple[Array, Array, List[Error]]:
        """Compute micro moment value contributions, handling any numerical errors."""
        errors: List[Error] = []
        micro_numerator, micro_denominator, _, _, _, _, _ = self.compute_micro_contributions(moments)
        return micro_numerator, micro_denominator, errors
