"""Economy-level structuring of optimal instrument results."""

import collections
import time
from typing import Dict, Hashable, List, Sequence, TYPE_CHECKING

import numpy as np

from .problem_results import ProblemResults
from ... import options
from ...configurations.formulation import ColumnFormulation
from ...parameters import PiParameter, RhoParameter, SigmaParameter
from ...utilities.basics import (
    Array, Mapping, StringRepresentation, TableFormatter, format_seconds, output, update_matrices
)


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..problem import StructuredProblem  # noqa


class OptimalInstrumentResults(StringRepresentation):
    r"""Results of optimal instrument computation.

    The :meth:`OptimalInstrumentResults.to_problem` method can be used to update the original :class:`Problem` with
    the computed optimal instruments. If a supply side was estimated, some columns of optimal instruments may need to
    be dropped because of collinearity issues. Refer to :meth:`OptimalInstrumentResults.to_problem` for more information
    about how to drop these collinear instruments.

    Attributes
    ----------
    problem_results : `ProblemResults`
        :class:`ProblemResults` that was used to compute these optimal instrument results.
    demand_instruments: `ndarray`
        Estimated optimal demand-side instruments, :math:`\mathscr{Z}_D`.
    supply_instruments: `ndarray`
        Estimated optimal supply-side instruments, :math:`\mathscr{Z}_S`.
    inverse_covariance_matrix: `ndarray`
        Inverse of the sample covariance matrix of the estimated :math:`\xi` and :math:`\omega`, which is used to
        normalize the expected Jacobians. If a supply side was not estimated, this is simply the sample estimate of
        :math:`1 / \text{Var}(\xi)`.
    expected_xi_by_theta_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\xi / \partial\theta \mid Z]`.
    expected_xi_by_beta_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\xi / \partial\beta \mid Z]`.
    expected_omega_by_theta_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\omega / \partial\theta \mid Z]`.
    expected_omega_by_beta_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\omega / \partial\beta \mid Z]`.
    computation_time : `float`
        Number of seconds it took to compute optimal instruments.
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
    MD : `int`
        Number of computed optimal demand-side instruments, :math:`M_D`.
    MS : `int`
        Number of computed optimal supply-side instruments, :math:`M_S`.

    Examples
    --------
    For an example of how to use this class, refer to the :doc:`Examples </examples>` section.

    """

    problem_results: ProblemResults
    demand_instruments: Array
    supply_instruments: Array
    inverse_covariance_matrix: Array
    expected_xi_by_theta_jacobian: Array
    expected_xi_by_beta_jacobian: Array
    expected_omega_by_theta_jacobian: Array
    expected_omega_by_beta_jacobian: Array
    computation_time: float
    draws: int
    fp_iterations: Array
    contraction_evaluations: Array
    MD: int
    MS: int

    def __init__(
            self, problem_results: ProblemResults, demand_instruments: Array, supply_instruments: Array,
            inverse_covariance_matrix: Array, expected_xi_by_theta_jacobian: Array,
            expected_xi_by_beta_jacobian: Array, expected_omega_by_theta_jacobian: Array,
            expected_omega_by_beta_jacobian: Array, start_time: float, end_time: float, draws: int,
            iteration_mappings: Sequence[Mapping[Hashable, int]],
            evaluation_mappings: Sequence[Mapping[Hashable, int]]) -> None:
        """Structure optimal instrument computation results."""
        self.problem_results = problem_results
        self.demand_instruments = demand_instruments
        self.supply_instruments = supply_instruments
        self.inverse_covariance_matrix = inverse_covariance_matrix
        self.expected_xi_by_theta_jacobian = expected_xi_by_theta_jacobian
        self.expected_xi_by_beta_jacobian = expected_xi_by_beta_jacobian
        self.expected_omega_by_theta_jacobian = expected_omega_by_theta_jacobian
        self.expected_omega_by_beta_jacobian = expected_omega_by_beta_jacobian
        self.computation_time = end_time - start_time
        self.draws = draws
        self.fp_iterations = self.cumulative_fp_iterations = np.array(
            [[m[t] if m else 0 for m in iteration_mappings] for t in problem_results.problem.unique_market_ids]
        )
        self.contraction_evaluations = self.cumulative_contraction_evaluations = np.array(
            [[m[t] if m else 0 for m in evaluation_mappings] for t in problem_results.problem.unique_market_ids]
        )
        self.MD = demand_instruments.shape[1]
        self.MS = supply_instruments.shape[1]

    def __str__(self) -> str:
        """Format optimal instrument computation results as a string."""

        # construct a section containing summary information
        summary_header = [
            ("Computation", "Time"), ("Error Term", "Draws"), ("MD:", "Demand Instruments"),
            ("MS:", "Supply Instruments")
        ]
        summary_values = [format_seconds(self.computation_time), self.draws, self.MD, self.MS]
        if self.fp_iterations.sum() > 0 or self.contraction_evaluations.sum() > 0:
            summary_header.extend([("Total Fixed Point", "Iterations"), ("Total Contraction", "Evaluations")])
            summary_values.extend([self.fp_iterations.sum(), self.contraction_evaluations.sum()])
        summary_widths = [max(len(k1), len(k2)) for k1, k2 in summary_header]
        summary_formatter = TableFormatter(summary_widths)
        summary_section = [
            "Optimal Instrument Computation Results Summary:",
            summary_formatter.line(),
            summary_formatter([k[0] for k in summary_header]),
            summary_formatter([k[1] for k in summary_header], underline=True),
            summary_formatter(summary_values),
            summary_formatter.line()
        ]

        # collect information about formulations associated with instruments
        problem = self.problem_results.problem
        formulation_mappings: List[Dict[str, ColumnFormulation]] = []  # noqa
        for parameter in self.problem_results._nonlinear_parameters.unfixed:
            if isinstance(parameter, SigmaParameter):
                formulation_mappings.append({
                    "Sigma Row": problem._X2_formulations[parameter.location[0]],
                    "Sigma Column": problem._X2_formulations[parameter.location[1]],
                })
            elif isinstance(parameter, PiParameter):
                formulation_mappings.append({
                    "Pi Row": problem._X2_formulations[parameter.location[0]],
                    "Pi Column": problem._X2_formulations[parameter.location[1]],
                })
            else:
                assert isinstance(parameter, RhoParameter)
                group_label = "All Groups" if parameter.single else problem.unique_nesting_ids[parameter.location[0]]
                formulation_mappings.append({"Rho Element": group_label})
        formulation_mappings.extend([{"Beta Element": f} for f in problem._X1_formulations])
        formulation_mappings.extend([{"Gamma Element": f} for f in problem._X3_formulations])

        # construct a section containing formulation information
        formulation_header = ["Column Indices:"] + list(map(str, range(len(formulation_mappings))))
        formulation_widths = [max(len(formulation_header[0]), max(map(len, formulation_mappings)))]
        for mapping in formulation_mappings:
            formulation_widths.append(max(5, max(map(len, map(str, mapping.values())))))
        formulation_formatter = TableFormatter(formulation_widths)
        formulation_section = [
            "Instruments:",
            formulation_formatter.line(),
            formulation_formatter(formulation_header, underline=True)
        ]
        for key in ["Sigma Row", "Sigma Column", "Pi Row", "Pi Column", "Rho Element", "Beta Element", "Gamma Element"]:
            formulations = [str(m.get(key, "")) for m in formulation_mappings]
            if any(formulations):
                formulation_section.append(formulation_formatter([key] + formulations))
        formulation_section.append(formulation_formatter.line())

        # combine the sections into one string
        return "\n\n".join("\n".join(s) for s in [summary_section, formulation_section])

    def to_problem(
            self, delete_demand_instruments: Sequence[int] = (),
            delete_supply_instruments: Sequence[int] = ()) -> 'StructuredProblem':
        """Re-create the problem with estimated optimal instruments.

        The re-created problem will be exactly the same, except that instruments will be replaced with the estimated
        optimal instruments.

        With a supply side, dropping one or more columns of instruments is often necessary because :math:`X_1` and
        :math:`X_3` are often formulated to include identical or similar exogenous product characteristics. Optimal
        instruments for these characteristics (the re-scaled characteristics themselves) will be collinear. The
        `delete_demand_instruments` and `delete_supply_instruments` arguments can be used to delete instruments
        when re-creating the problem. Outputted optimal instrument results indicate which instrument column indices
        correspond to which product characteristics.

        For example, if :math:`X_1` contains some exogenous characteristic ``x`` and :math:`X_3` contains ``log(x)``,
        both the demand- and supply-side optimal instruments will contain scaled versions of these almost collinear
        characteristics. One way to deal with this is to include the column index of the instrument for ``log(x)`` in
        `delete_demand_instruments` and to include the column index of the instrument for ``x`` in
        `delete_supply_instruments`.

        .. note::

           Any fixed effects that were absorbed in the original problem will be be absorbed here too. However, compared
           to a problem updated with optimal instruments when fixed effects are included as indicators, results may be
           slightly different.

        Parameters
        ----------
        delete_demand_instruments : `tuple of int, optional`
            Column indices of :attr:`OptimalInstrumentResults.demand_instruments` to drop when re-creating the problem.
        delete_supply_instruments : `tuple of int, optional`
            Column indices of :attr:`OptimalInstrumentResults.supply_instruments` to drop when re-creating the problem.
            This is only relevant if a supply side was estimated.

        Returns
        -------
        `Problem`
            :class:`Problem` updated to use the estimated optimal instruments.

        Examples
        --------
        For an example of turning these results into a :class:`Problem`, refer to the :doc:`Examples </examples>`
        section.

        """

        # keep track of long it takes to re-create the problem
        output("Re-creating the problem ...")
        start_time = time.time()

        # validate the indices
        if not isinstance(delete_demand_instruments, collections.Sequence):
            raise TypeError("delete_demand_instruments must be a tuple.")
        if not isinstance(delete_supply_instruments, collections.Sequence):
            raise TypeError("delete_supply_instruments must be a tuple.")
        if not all(i in range(self.MD) for i in delete_demand_instruments):
            raise ValueError(f"delete_demand_instruments must contain column indices between 0 and {self.MD}.")
        if not all(i in range(self.MS) for i in delete_supply_instruments):
            raise ValueError(f"delete_supply_instruments must contain column indices between 0 and {self.MS}.")
        if self.MS == 0 and delete_supply_instruments:
            raise ValueError("A supply side was not estimated, so delete_supply_instruments should not be specified.")

        # update the products array
        updated_products = update_matrices(self.problem_results.problem.products, {
            'ZD': (np.delete(self.demand_instruments, delete_demand_instruments, axis=1), options.dtype),
            'ZS': (np.delete(self.supply_instruments, delete_supply_instruments, axis=1), options.dtype)
        })

        # re-create the problem
        from ..problem import StructuredProblem  # noqa
        problem = StructuredProblem(
            self.problem_results.problem.product_formulations, self.problem_results.problem.agent_formulation,
            updated_products, self.problem_results.problem.agents, updating_instruments=True
        )
        output(f"Re-created the problem after {format_seconds(time.time() - start_time)}.")
        output("")
        output(problem)
        return problem
