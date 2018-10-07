"""Economy-level structuring of optimal instrument results."""

from typing import Hashable, Sequence, TYPE_CHECKING

import numpy as np

from .problem_results import ProblemResults
from ...utilities.basics import Array, Mapping, StringRepresentation, TableFormatter, format_seconds


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..problem import OptimalInstrumentProblem  # noqa


class OptimalInstrumentResults(StringRepresentation):
    r"""Results of optimal instrument computation.

    The :meth:`OptimalInstrumentResults.to_problem` method can be used to update the original :class:`Problem` with
    the computed optimal excluded instruments.

    Attributes
    ----------
    problem_results : `ProblemResults`
        :class:`ProblemResults` that was used to compute these optimal instrument results.
    demand_instruments: `ndarray`
        Estimated optimal excluded demand-side instruments, :math:`\mathscr{Z}_D`.
    supply_instruments: `ndarray`
        Estimated optimal excluded supply-side instruments, :math:`\mathscr{Z}_S`.
    inverse_covariance_matrix: `ndarray`
        Inverse of the sample covariance matrix of the estimated :math:`\xi` and :math:`\omega`, which is used to
        normalize the expected Jacobians. If a supply side was not estimated, this is simply the sample estimate of
        :math:`1 / \text{Var}(\xi)`.
    expected_xi_by_theta_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\xi / \partial\theta \mid Z]`.
    expected_xi_by_alpha_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\xi / \partial\alpha \mid Z]`.
    expected_omega_by_theta_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\omega / \partial\theta \mid Z]`.
    expected_omega_by_alpha_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\omega / \partial\alpha \mid Z]`.
    computation_time : `float`
        Number of seconds it took to compute optimal excluded instruments.
    draws : `int`
        Number of draws used to approximate the integral over the error term density.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute equilibrium prices in each market
        for each error term draw. Rows are in the same order as :attr:`Problem.unique_market_ids` and column indices
        correspond to draws.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute equilibrium prices was evaluated in each market for each error
        term draw. Rows are in the same order as :attr:`Problem.unique_market_ids` and column indices correspond to
        draws.

    Examples
    --------
    .. only:: html

       - :doc:`Tutorial </tutorial>`

    .. only:: latex

       Refer to the online documentation.

    """

    problem_results: ProblemResults
    demand_instruments: Array
    supply_instruments: Array
    inverse_covariance_matrix: Array
    expected_xi_by_theta_jacobian: Array
    expected_xi_by_alpha_jacobian: Array
    expected_omega_by_theta_jacobian: Array
    expected_omega_by_alpha_jacobian: Array
    computation_time: float
    draws: int
    fp_iterations: Array
    contraction_evaluations: Array

    def __init__(
            self, problem_results: ProblemResults, demand_instruments: Array, supply_instruments: Array,
            inverse_covariance_matrix: Array, expected_xi_by_theta_jacobian: Array,
            expected_xi_by_alpha_jacobian: Array, expected_omega_by_theta_jacobian: Array,
            expected_omega_by_alpha_jacobian: Array, start_time: float, end_time: float, draws: int,
            iteration_mappings: Sequence[Mapping[Hashable, int]],
            evaluation_mappings: Sequence[Mapping[Hashable, int]]) -> None:
        """Structure optimal excluded instrument computation results."""
        self.problem_results = problem_results
        self.demand_instruments = demand_instruments
        self.supply_instruments = supply_instruments
        self.inverse_covariance_matrix = inverse_covariance_matrix
        self.expected_xi_by_theta_jacobian = expected_xi_by_theta_jacobian
        self.expected_xi_by_alpha_jacobian = expected_xi_by_alpha_jacobian
        self.expected_omega_by_theta_jacobian = expected_omega_by_theta_jacobian
        self.expected_omega_by_alpha_jacobian = expected_omega_by_alpha_jacobian
        self.computation_time = end_time - start_time
        self.draws = draws
        self.fp_iterations = self.cumulative_fp_iterations = np.array(
            [[m[t] if m else 0 for m in iteration_mappings] for t in problem_results.problem.unique_market_ids]
        )
        self.contraction_evaluations = self.cumulative_contraction_evaluations = np.array(
            [[m[t] if m else 0 for m in evaluation_mappings] for t in problem_results.problem.unique_market_ids]
        )

    def __str__(self) -> str:
        """Format optimal instrument computation results as a string."""
        header = [("Computation", "Time"), ("Error Term", "Draws")]
        values = [format_seconds(self.computation_time), self.draws]
        if self.fp_iterations.sum() > 0 or self.contraction_evaluations.sum() > 0:
            header.extend([("Total Fixed Point", "Iterations"), ("Total Contraction", "Evaluations")])
            values.extend([self.fp_iterations.sum(), self.contraction_evaluations.sum()])
        widths = [max(len(k1), len(k2)) for k1, k2 in header]
        formatter = TableFormatter(widths)
        return "\n".join([
            "Optimal Instrument Results Summary:",
            formatter.line(),
            formatter([k[0] for k in header]),
            formatter([k[1] for k in header], underline=True),
            formatter(values),
            formatter.line()
        ])

    def to_problem(self) -> 'OptimalInstrumentProblem':
        """Re-create the problem with estimated optimal excluded instruments.

        The re-created problem will be exactly the same, except that excluded instruments will be replaced with the
        estimated optimal excluded instruments.

        Returns
        -------
        `OptimalInstrumentProblem`
            :class:`OptimalInstrumentProblem`, which is a :class:`Problem` updated to use the estimated optimal
            excluded instruments.

        Examples
        --------
        .. only:: html

           - :doc:`Tutorial </tutorial>`

        .. only:: latex

           Refer to the online documentation.

        """
        from ..problem import OptimalInstrumentProblem  # noqa
        return OptimalInstrumentProblem(self.problem_results.problem, self.demand_instruments, self.supply_instruments)
