"""Market level structuring of simulated synthetic BLP data."""

from typing import Any, List, Optional, Tuple

import numpy as np

from .market import Market
from .. import exceptions, options
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

    @NumericalErrorHandler(exceptions.MicroScoresNumericalError)
    def safely_compute_score_denominator_contributions(
            self, dataset: MicroDataset) -> Tuple[Array, Array, Array, List[Error]]:
        """Compute denominator contributions to micro scores, handling any numerical errors."""

        # compute probabilities and their derivatives
        probabilities, conditionals = self.compute_probabilities()
        probabilities_tangent_mapping, conditionals_tangent_mapping = (
            self.compute_probabilities_by_parameter_tangent_mapping(probabilities, conditionals)
        )
        xi_jacobian, errors = self.compute_xi_by_theta_jacobian(
            probabilities, conditionals, probabilities_tangent_mapping
        )
        self.update_probabilities_by_parameter_tangent_mapping(
            probabilities_tangent_mapping, conditionals_tangent_mapping, probabilities, conditionals, xi_jacobian
        )

        # compute contributions
        _, denominator_mapping, _, tangent_mapping = self.compute_micro_dataset_contributions(
            [dataset], self.delta, probabilities, probabilities_tangent_mapping, compute_jacobians=True
        )
        if dataset in denominator_mapping:
            denominator = denominator_mapping[dataset]
            jacobian = np.array([tangent_mapping[(dataset, p)] for p in range(self.parameters.P)])
        else:
            denominator = 0
            jacobian = np.zeros(self.parameters.P, options.dtype)

        return xi_jacobian, denominator, jacobian, errors

    @NumericalErrorHandler(exceptions.MicroScoresNumericalError)
    def safely_compute_score_numerator_contributions(
            self, dataset: MicroDataset, j: Optional[Any], k: Optional[Any], xi_jacobian: Array) -> (
            Tuple[Array, Array, List[Error]]):
        """Compute numerator contributions to micro scores, handling any numerical errors."""
        errors: List[Error] = []

        # compute probabilities and their derivatives
        probabilities, conditionals = self.compute_probabilities()
        probabilities_tangent_mapping, conditionals_tangent_mapping = (
            self.compute_probabilities_by_parameter_tangent_mapping(probabilities, conditionals)
        )
        self.update_probabilities_by_parameter_tangent_mapping(
            probabilities_tangent_mapping, conditionals_tangent_mapping, probabilities, conditionals, xi_jacobian
        )

        # obtain weights and their derivatives
        weights_mapping, _, tangent_mapping, _ = self.compute_micro_dataset_contributions(
            [dataset], self.delta, probabilities, probabilities_tangent_mapping, compute_jacobians=True
        )
        if dataset in weights_mapping:
            weights = weights_mapping[dataset]
            tangent = np.stack([tangent_mapping[(dataset, p)] for p in range(self.parameters.P)], axis=-1)
        else:
            weights = np.zeros_like(self.compute_micro_weights(dataset))
            tangent = np.zeros(list(weights.shape) + [self.parameters.P], options.dtype)

        # validate choices and select corresponding weights if specified
        if j is not None:
            try:
                weights = weights[:, j]
                tangent = tangent[:, j]
            except IndexError as exception:
                message = f"In market '{self.t}', choice index '{j}' is not between 0 and {weights.shape[1] - 1}."
                raise ValueError(message) from exception

        # validate second choices and select corresponding weights if specified and there are second choices
        if k is not None and len(weights.shape) == 1 + int(j is None) + 1:
            try:
                weights = weights[:, k] if j is not None else weights[:, :, k]
                tangent = tangent[:, k] if j is not None else tangent[:, :, k]
            except IndexError as exception:
                message = f"In market '{self.t}', choice index '{k}' is not between 0 and {weights.shape[-1] - 1}."
                raise ValueError(message) from exception

        # integrate over agents to get the numerator contributions
        numerator = weights.sum(axis=0)
        jacobian = tangent.sum(axis=0)

        return numerator, jacobian, errors
