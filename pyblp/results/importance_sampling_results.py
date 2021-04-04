"""Economy-level structuring of importance sampling results."""

from pathlib import Path
import pickle
from typing import Sequence, TYPE_CHECKING, Union

import numpy as np

from .problem_results import ProblemResults
from ..utilities.basics import Array, Groups, RecArray, StringRepresentation, format_seconds, format_table


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import ImportanceSamplingProblem  # noqa


class ImportanceSamplingResults(StringRepresentation):
    r"""Results of importance sampling.

    Along with the sampled agents, these results also contain a number of useful importance sampling diagnostics from
    :ref:`references:Owen (2013)`.

    The :meth:`ImportanceSamplingResults.to_problem` method can be used to update the original :class:`Problem` with
    the importance sampling agent data.

    Attributes
    ----------
    problem_results : `ProblemResults`
        :class:`ProblemResults` that was used to compute these importance sampling results.
    sampled_agents : `Agents`
        Importance sampling agent data structured as :class:`Agents`. The :func:`data_to_dict` function can be used to
        convert this into a more usable data type.
    computation_time : `float`
        Number of seconds it took to do importance sampling.
    draws : `int`
        Number of importance sampling draws in each market.
    diagnostic_market_ids : `ndarray`
        Market IDs the correspond to the ordering of the following arrays of weight diagnostics.
    weight_sums : `ndarray`
        Sum of weights in each market: :math:`\sum_i w_{it}`. If importance sampling was successful, weights should not
        sum to numbers too far from one.
    effective_draws : `ndarray`
        Effective sample sizes in each market: :math:`\frac{(\sum_i w_{it})^2}{\sum_i w_{it}^2}`.
    effective_draws_for_variance : `ndarray`
        Effective sample sizes for variance estimates in each market:
        :math:`\frac{(\sum_i w_{it}^2)^2}{\sum_i w_{it}^4}`.
    effective_draws_for_skewness : `ndarray`
        Effective sample sizes for gauging skewness in each market:
        :math:`\frac{(\sum_i w_{it}^2)^3}{(\sum_i w_{it}^3)^2}`.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    problem_results: ProblemResults
    sampled_agents: RecArray
    computation_time: float
    draws: int
    diagnostic_market_ids: Array
    weight_sums: Array
    effective_draws: Array
    effective_draws_for_variance: Array
    effective_draws_for_skewness: Array

    def __init__(
            self, problem_results: ProblemResults, sampled_agents: RecArray, start_time: float, end_time: float,
            draws: int) -> None:
        """Structure importance sampling results and compute diagnostics."""
        self.problem_results = problem_results
        self.sampled_agents = sampled_agents
        self.computation_time = end_time - start_time
        self.draws = draws

        # compute weight sums
        groups = Groups(self.sampled_agents.market_ids)
        self.diagnostic_market_ids = groups.unique
        self.weight_sums = groups.sum(self.sampled_agents.weights)

        # compute effective draws
        squared_weight_sums = groups.sum(self.sampled_agents.weights**2)
        self.effective_draws = self.weight_sums**2 / squared_weight_sums
        self.effective_draws_for_variance = squared_weight_sums**2 / groups.sum(self.sampled_agents.weights**4)
        self.effective_draws_for_skewness = squared_weight_sums**3 / groups.sum(self.sampled_agents.weights**3)**2

    def __str__(self) -> str:
        """Format importance sampling results as a string."""
        header = [
            ("Computation", "Time"), ("Total", "Sampling Draws"), ("Sampling Draws", "per Market"),
            ("Min", "Effective Draws"), ("Min", "Effective Draws", "for Variance"),
            ("Min", "Effective Draws", "for Skewness"), ("Min", "Weight Sum"), ("Max", "Weight Sum")
        ]
        values = [
            format_seconds(self.computation_time), self.sampled_agents.shape[0], self.draws,
            np.min(self.effective_draws).astype(int), np.min(self.effective_draws_for_variance).astype(int),
            np.min(self.effective_draws_for_skewness).astype(int), np.min(self.weight_sums), np.max(self.weight_sums)
        ]
        return format_table(header, values, title="Importance Sampling Results Summary")

    def to_pickle(self, path: Union[str, Path]) -> None:
        """Save these results as a pickle file.

        Parameters
        ----------
        path: `str or Path`
            File path to which these results will be saved.

        """
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    def to_dict(
            self, attributes: Sequence[str] = (
                'sampled_agents', 'computation_time', 'draws', 'diagnostic_market_ids', 'weight_sums',
                'effective_draws', 'effective_draws_for_variance', 'effective_draws_for_skewness'
            )) -> dict:
        """Convert these results into a dictionary that maps attribute names to values.

        Parameters
        ----------
        attributes : `sequence of str, optional`
            Names of attributes that will be added to the dictionary. By default, all :class:`ImportanceSamplingResults`
            attributes are added except for :attr:`ImportanceSamplingResults.problem_results`.

        Returns
        -------
        `dict`
            Mapping from attribute names to values.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        return {k: getattr(self, k) for k in attributes}

    def to_problem(self) -> 'ImportanceSamplingProblem':
        """Re-create the problem with the agent data constructed from importance sampling.

        The re-created problem will be exactly the same, except :attr:`Problem.agents` will be replaced with
        :attr:`ImportanceSamplingResults.sampled_agents`.

        Returns
        -------
        `ImportanceSamplingProblem`
            :class:`ImportanceSamplingProblem`, which is a :class:`Problem` updated to use agent data constructed from
            importance sampling.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        from ..economies.problem import ImportanceSamplingProblem  # noqa
        return ImportanceSamplingProblem(self.problem_results.problem, self.sampled_agents)
