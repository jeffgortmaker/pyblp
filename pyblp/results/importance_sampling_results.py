"""Economy-level structuring of importance sampling results."""

from typing import Dict, Hashable, Sequence, TYPE_CHECKING

import numpy as np

from .problem_results import ProblemResults
from ..utilities.basics import Array, Groups, RecArray, SolverStats, StringRepresentation, format_seconds, format_table


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
    precise_delta : `ndarray`
        Estimated :math:`\hat{\delta}(\hat{\theta})` used to do importance sampling. By default, this is
        :attr:`ProblemResults.delta`.
    computation_time : `float`
        Number of seconds it took to do importance sampling.
    draws : `int`
        Number of importance sampling draws in each market.
    weight_sums : `ndarray`
        Sum of weights in each market: :math:`\sum_i w_{it}`. If importance sampling was successful, weights should not
        sum to numbers too far from one.
    effective_draws : `ndarray`
        Effective sample sizes in each market: :math:`\frac{(\sum_i w_{it})^2}{\sum_i w_i^2}`.
    effective_draws_for_variance : `ndarray`
        Effective sample sizes for variance estimates in each market: :math:`\frac{(\sum_i w_i^2)^2}{\sum_i w_i^4}`.
    effective_draws_for_skewness : `ndarray`
        Effective sample sizes for gauging skewness in each market: :math:`\frac{(\sum_i w_i^2)^3}{(\sum_i w_i^3)^2}`.
    fp_converged : `ndarray`
        Flags for convergence of the iteration routine used to compute :attr:`ImportanceSamplingResults.precise_delta`
        in each market. Flags are in the same order as :attr:`Problem.unique_market_ids`.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute
        :attr:`ImportanceSamplingResults.precise_delta` in each market. Numbers are in the same order as
        :attr:`Problem.unique_market_ids`.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute :attr:`ImportanceSamplingResults.precise_delta` was evaluated in
        each market. Numbers are in the same order as :attr:`Problem.unique_market_ids`.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    problem_results: ProblemResults
    sampled_agents: RecArray
    precise_delta: Array
    computation_time: float
    draws: int
    weight_sums: Array
    effective_draws: Array
    effective_draws_for_variance: Array
    effective_draws_for_skewness: Array
    fp_converged: Array
    fp_iterations: Array
    contraction_evaluations: Array

    def __init__(
            self, problem_results: ProblemResults, sampled_agents: RecArray, precise_delta: Array, start_time: float,
            end_time: float, draws: int, iteration_stats: Dict[Hashable, SolverStats]) -> None:
        """Structure importance sampling results and compute diagnostics."""
        self.problem_results = problem_results
        self.sampled_agents = sampled_agents
        self.precise_delta = precise_delta
        self.computation_time = end_time - start_time
        self.draws = draws
        unique_market_ids = problem_results.problem.unique_market_ids
        self.fp_converged = np.array(
            [iteration_stats[t].converged if iteration_stats else True for t in unique_market_ids], dtype=np.bool
        )
        self.fp_iterations = np.array(
            [iteration_stats[t].iterations if iteration_stats else 0 for t in unique_market_ids], dtype=np.int
        )
        self.contraction_evaluations = np.array(
            [iteration_stats[t].evaluations if iteration_stats else 0 for t in unique_market_ids], dtype=np.int
        )

        # compute weight sums
        groups = Groups(self.sampled_agents.market_ids)
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
        if self.fp_iterations.sum() > 0 or self.contraction_evaluations.sum() > 0:
            header.extend([("Fixed Point", "Iterations"), ("Contraction", "Evaluations")])
            values.extend([self.fp_iterations.sum(), self.contraction_evaluations.sum()])
        return format_table(header, values, title="Importance Sampling Results Summary")

    def to_dict(
            self, attributes: Sequence[str] = (
                'sampled_agents', 'precise_delta', 'computation_time', 'draws', 'fp_converged', 'fp_iterations',
                'contraction_evaluations'
            )) -> dict:
        """Convert these results into a dictionary that maps attribute names to values.

        Once converted to a dictionary, these results can be saved to a file with :func:`pickle.dump`.

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
