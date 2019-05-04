"""Economy-level structuring of optimal instrument results."""

from typing import Hashable, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .problem_results import ProblemResults
from ..configurations.formulation import Formulation
from ..parameters import LinearCoefficient
from ..utilities.basics import Array, Mapping, StringRepresentation, TableFormatter, format_seconds


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import OptimalInstrumentProblem  # noqa


class OptimalInstrumentResults(StringRepresentation):
    r"""Results of optimal instrument computation.

    The :meth:`OptimalInstrumentResults.to_problem` method can be used to update the original :class:`Problem` with
    the computed optimal instruments.

    Attributes
    ----------
    problem_results : `ProblemResults`
        :class:`ProblemResults` that was used to compute these optimal instrument results.
    demand_instruments: `ndarray`
        Estimated optimal demand-side instruments for :math:`\theta`, :math:`Z_D^\textit{Opt}`.
    supply_instruments: `ndarray`
        Estimated optimal supply-side instruments for :math:`\theta`, :math:`Z_S^\textit{Opt}`.
    supply_shifter_formulation : `Formulation or None`
        :class:`Formulation` configuration for supply shifters that will by default be included in the full set of
        optimal demand-side instruments. This is only constructed if a supply side was estimated, and it can be changed
        in :meth:`OptimalInstrumentResults.to_problem`. By default, this is the formulation for :math:`X_3` from
        :class:`Problem` excluding any variables in the formulation for :math:`X_1`.
    demand_shifter_formulation : `Formulation or None`
        :class:`Formulation` configuration for demand shifters that will by default be included in the full set of
        optimal supply-side instruments. This is only constructed if a supply side was estimated, and it can be changed
        in :meth:`OptimalInstrumentResults.to_problem`. By default, this is the formulation for :math:`X_1^x` from
        :class:`Problem` excluding any variables in the formulation for :math:`X_3`.
    inverse_covariance_matrix: `ndarray`
        Inverse of the sample covariance matrix of the estimated :math:`\xi` and :math:`\omega`, which is used to
        normalize the expected Jacobians. If a supply side was not estimated, this is simply the sample estimate of
        :math:`1 / \sigma_{\xi}^2`.
    expected_xi_by_theta_jacobian: `ndarray`
        Estimated :math:`E[\frac{\partial\xi}{\partial\theta} \mid Z]`.
    expected_omega_by_theta_jacobian: `ndarray`
        Estimated :math:`E[\frac{\partial\omega}{\partial\theta} \mid Z]`.
    expected_prices : `ndarray`
        Vector of expected prices conditional on all exogenous variables, :math:`E[p \mid Z]`, which may have been
        specified in :meth:`ProblemResults.compute_optimal_instruments`.
    computation_time : `float`
        Number of seconds it took to compute optimal excluded instruments.
    draws : `int`
        Number of draws used to approximate the integral over the error term density.
    fp_converged : `ndarray`
        Flags for convergence of the iteration routine used to compute equilibrium prices in each market. Rows are in
        the same order as :attr:`Problem.unique_market_ids` and column indices correspond to draws.
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
        - :doc:`Tutorial </tutorial>`

    """

    problem_results: ProblemResults
    demand_instruments: Array
    supply_instruments: Array
    demand_shifter_formulation: Optional[Formulation]
    supply_shifter_formulation: Optional[Formulation]
    inverse_covariance_matrix: Array
    expected_xi_by_theta_jacobian: Array
    expected_omega_by_theta_jacobian: Array
    expected_prices: Array
    computation_time: float
    draws: int
    fp_converged: Array
    fp_iterations: Array
    contraction_evaluations: Array

    def __init__(
            self, problem_results: ProblemResults, demand_instruments: Array, supply_instruments: Array,
            inverse_covariance_matrix: Array, expected_xi_jacobian: Array, expected_omega_jacobian: Array,
            expected_prices: Array, start_time: float, end_time: float, draws: int,
            converged_mappings: Sequence[Mapping[Hashable, bool]], iteration_mappings: Sequence[Mapping[Hashable, int]],
            evaluation_mappings: Sequence[Mapping[Hashable, int]]) -> None:
        """Structure optimal excluded instrument computation results. Also identify supply and demand shifters that will
        be added to the optimal instruments when converting them into a problem.
        """
        self.problem_results = problem_results
        self.demand_instruments = demand_instruments
        self.supply_instruments = supply_instruments
        self.inverse_covariance_matrix = inverse_covariance_matrix
        self.expected_xi_by_theta_jacobian = expected_xi_jacobian
        self.expected_omega_by_theta_jacobian = expected_omega_jacobian
        self.expected_prices = expected_prices
        self.computation_time = end_time - start_time
        self.draws = draws
        self.fp_converged = np.array(
            [[m[t] if m else True for m in converged_mappings] for t in problem_results.problem.unique_market_ids],
            dtype=np.int
        )
        self.fp_iterations = np.array(
            [[m[t] if m else 0 for m in iteration_mappings] for t in problem_results.problem.unique_market_ids],
            dtype=np.int
        )
        self.contraction_evaluations = np.array(
            [[m[t] if m else 0 for m in evaluation_mappings] for t in problem_results.problem.unique_market_ids],
            dtype=np.int
        )

        # construct default supply and demand shifter formulations
        self.supply_shifter_formulation = self.demand_shifter_formulation = None
        if self.problem_results.problem.K3 > 0:
            assert self.problem_results.problem.product_formulations[0] is not None
            assert self.problem_results.problem.product_formulations[2] is not None
            X1_expressions = self.problem_results.problem.product_formulations[0]._expressions
            X3_expressions = self.problem_results.problem.product_formulations[2]._expressions
            supply_shifters = {str(e) for e in X3_expressions}
            demand_shifters = {str(e) for e in X1_expressions if all(str(s) != 'prices' for s in e.free_symbols)}
            if supply_shifters - demand_shifters:
                supply_shifter_formula = ' + '.join(sorted(supply_shifters - demand_shifters))
                self.supply_shifter_formulation = Formulation(f'{supply_shifter_formula} - 1')
            if demand_shifters - supply_shifters:
                demand_shifter_formula = ' + '.join(sorted(demand_shifters - supply_shifters))
                self.demand_shifter_formulation = Formulation(f'{demand_shifter_formula} - 1')

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

    def to_problem(
            self, supply_shifter_formulation: Optional[Formulation] = None,
            demand_shifter_formulation: Optional[Formulation] = None) -> 'OptimalInstrumentProblem':
        r"""Re-create the problem with estimated feasible optimal instruments.

        The re-created problem will be exactly the same, except that instruments will be replaced with estimated
        feasible optimal instruments.

        .. note::

           Most of the explanation here is only important if a supply side was estimated.

        The optimal excluded demand-side instruments consist of the following:

            1. Estimated optimal demand-side instruments for :math:`\theta`, :math:`Z_D^\textit{Opt}`, excluding columns
               of instruments for any exogenous linear parameters that were not concentrated out, but rather included in
               :math:`\theta` by :meth:`Problem.solve`.

            2. Optimal instruments for any linear demand-side parameters on endogenous product characteristics,
               :math:`\alpha`, which were concentrated out and hence not included in :math:`\theta`. These optimal
               instruments are simply an integral of the endogenous product characteristics, :math:`X_1^p`, over the
               joint density of :math:`\xi` and :math:`\omega`. It is only possible to concentrate out :math:`\alpha`
               when there isn't a supply side, so the approximation of these optimal instruments is simply :math:`X_1^p`
               evaluated at the constant vector of expected prices, :math:`E[p \mid Z]`, specified in
               :meth:`ProblemResults.compute_optimal_instruments`.

            3. If a supply side was estimated, any supply shifters, which are by default formulated by
               :attr:`OptimalInstrumentResults.supply_shifter_formulation`: all characteristics in :math:`X_3` not in
               :math:`X_1`.

        Similarly, if a supply side was estimated, the optimal excluded supply-side instruments consist of the
        following:

            1. Estimated optimal supply-side instruments for :math:`\theta`, :math:`Z_S^\textit{Opt}`, excluding columns
               of instruments for any exogenous linear parameters that were not concentrated out, but rather included in
               :math:`\theta` by :meth:`Problem.solve`.

            2. If a supply side was estimated, any demand shifters, which are by default formulated by
               :attr:`OptimalInstrumentResults.demand_shifter_formulation`: all characteristics in :math:`X_1^x` not in
               :math:`X_3`.

        As usual, the excluded demand-side instruments will be supplemented with :math:`X_1^x` and the excluded
        supply-side instruments will be supplemented with :math:`X_3`. The same fixed effects configured in
        :class:`Problem` will be absorbed.

        .. warning::

           If a supply side was estimated, the addition of supply- and demand-shifters may create collinearity issues.
           Make sure to check that shifters and other product characteristics are not collinear.

        Parameters
        ----------
        supply_shifter_formulation : `Formulation, optional`
            :class:`Formulation` configuration for supply shifters to be included in the set of optimal demand-side
            instruments. This is only used if a supply side was estimated. Intercepts will be ignored. By default,
            :attr:`OptimalInstrumentResults.supply_shifter_formulation` is used.
        demand_shifter_formulation : `Formulation, optional`
            :class:`Formulation` configuration for demand shifters to be included in the set of optimal supply-side
            instruments. This is only used if a supply side was estimated. Intercepts will be ignored. By default,
            :attr:`OptimalInstrumentResults.demand_shifter_formulation` is used.

        Returns
        -------
        `OptimalInstrumentProblem`
            :class:`OptimalInstrumentProblem`, which is a :class:`Problem` updated to use the estimated optimal
            instruments.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """

        # configure or validate the supply shifter formulation
        if self.problem_results.problem.K3 == 0:
            if supply_shifter_formulation is not None:
                raise TypeError("A supply side was not estimated, so supply_shifter_formulation should be None.")
        elif supply_shifter_formulation is None:
            supply_shifter_formulation = self.supply_shifter_formulation
        elif not isinstance(supply_shifter_formulation, Formulation):
            raise TypeError("supply_shifter_formulation must be None or a Formulation instance.")
        elif supply_shifter_formulation._names:
            supply_shifter_formulation = Formulation(f'{supply_shifter_formulation._formula} - 1')
        else:
            supply_shifter_formulation = None

        # configure or validate the demand shifter formulation
        if self.problem_results.problem.K3 == 0:
            if demand_shifter_formulation is not None:
                raise TypeError("A demand side was not estimated, so demand_shifter_formulation should be None.")
        elif demand_shifter_formulation is None:
            demand_shifter_formulation = self.demand_shifter_formulation
        elif not isinstance(demand_shifter_formulation, Formulation):
            raise TypeError("demand_shifter_formulation must be None or a Formulation instance.")
        elif demand_shifter_formulation._names:
            demand_shifter_formulation = Formulation(f'{demand_shifter_formulation._formula} - 1')
        else:
            demand_shifter_formulation = None

        # identify which parameters in theta are exogenous linear parameters
        dropped_index = np.zeros(self.problem_results._parameters.P, np.bool)
        for p, parameter in enumerate(self.problem_results._parameters.unfixed):
            if not isinstance(parameter, LinearCoefficient):
                continue
            if 'prices' in parameter.get_product_formulation(self.problem_results.problem).names:
                continue
            dropped_index[p] = True

        # build excluded demand-side instruments
        demand_instruments = self.demand_instruments[:, ~dropped_index]
        if self.problem_results._parameters.eliminated_alpha_index.any():
            assert self.expected_prices is not None
            demand_instruments = np.c_[
                demand_instruments,
                self.problem_results.problem._compute_true_X1(
                    {'prices': self.expected_prices},
                    self.problem_results._parameters.eliminated_alpha_index.flatten()
                )
            ]
        if supply_shifter_formulation is not None:
            demand_instruments = np.c_[
                demand_instruments,
                supply_shifter_formulation._build_matrix(self.problem_results.problem.products)[0]
            ]

        # build excluded supply-side instruments
        if self.problem_results.problem.K3 == 0:
            supply_instruments = self.supply_instruments
        else:
            supply_instruments = self.supply_instruments[:, ~dropped_index]
            if demand_shifter_formulation is not None:
                supply_instruments = np.c_[
                    supply_instruments,
                    demand_shifter_formulation._build_matrix(self.problem_results.problem.products)[0]
                ]

        # initialize the problem
        from ..economies.problem import OptimalInstrumentProblem  # noqa
        return OptimalInstrumentProblem(self.problem_results.problem, demand_instruments, supply_instruments)
