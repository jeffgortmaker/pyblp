"""Economy-level structuring of BLP simulation results."""

from pathlib import Path
import pickle
from typing import Dict, Hashable, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import numpy as np

from .economy_results import EconomyResults
from .. import options
from ..configurations.formulation import Formulation
from ..configurations.integration import Integration
from ..construction import build_blp_instruments, build_matrix
from ..utilities.basics import (
    Array, SolverStats, Mapping, update_matrices, RecArray, format_number, format_seconds, format_table
)


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import Problem  # noqa
    from ..economies.simulation import Simulation  # noqa


class SimulationResults(EconomyResults):
    r"""Results of a solved simulation of synthetic BLP data.

    The :meth:`SimulationResults.to_problem` method can be used to convert the full set of simulated data (along with
    some basic default instruments) and configured information into a :class:`Problem`. Additionally, this class has
    duplicates of the following :class:`ProblemResults` methods:

        - :meth:`ProblemResults.compute_aggregate_elasticities`
        - :meth:`ProblemResults.compute_elasticities`
        - :meth:`ProblemResults.compute_demand_jacobians`
        - :meth:`ProblemResults.compute_demand_hessians`
        - :meth:`ProblemResults.compute_profit_hessians`
        - :meth:`ProblemResults.compute_diversion_ratios`
        - :meth:`ProblemResults.compute_long_run_diversion_ratios`
        - :meth:`ProblemResults.compute_probabilities`
        - :meth:`ProblemResults.extract_diagonals`
        - :meth:`ProblemResults.extract_diagonal_means`
        - :meth:`ProblemResults.compute_delta`
        - :meth:`ProblemResults.compute_costs`
        - :meth:`ProblemResults.compute_passthrough`
        - :meth:`ProblemResults.compute_approximate_prices`
        - :meth:`ProblemResults.compute_prices`
        - :meth:`ProblemResults.compute_shares`
        - :meth:`ProblemResults.compute_hhi`
        - :meth:`ProblemResults.compute_markups`
        - :meth:`ProblemResults.compute_profits`
        - :meth:`ProblemResults.compute_consumer_surpluses`
        - :meth:`ProblemResults.compute_micro_values`
        - :meth:`ProblemResults.compute_micro_scores`
        - :meth:`ProblemResults.compute_agent_scores`
        - :meth:`ProblemResults.simulate_micro_data`

    Attributes
    ----------
    simulation : `Simulation`
        :class:`Simulation` that created these results.
    product_data : `recarray`
        Simulated :attr:`Simulation.product_data` with product characteristics replaced so as to be consistent with the
        true parameters. If :meth:`Simulation.replace_endogenous` was used to create these results, prices and
        market shares were replaced. If :meth:`Simulation.replace_exogenous` was used, exogenous characteristics were
        replaced instead. The :func:`data_to_dict` function can be used to convert this into a more usable data type.
    delta : `ndarray`
        Simulated mean utility, :math:`\delta`.
    costs : `ndarray`
        Simulated marginal costs, :math:`c`.
    computation_time : `float`
        Number of seconds it took to compute prices and market shares.
    fp_converged : `ndarray`
        Flags for convergence of the iteration routine used to compute prices or :math:`\delta` (depending on the method
        used to create these results) in each market. Flags are in the same order as
        :attr:`Simulation.unique_market_ids`.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute prices or :math:`\delta` in each
        market. Counts are in the same order as :attr:`Simulation.unique_market_ids`.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute prices or :math:`\delta` was evaluated in each market. Counts
        are in the same order as :attr:`Simulation.unique_market_ids`.
    profit_gradients : `dict`
        Mapping from market IDs :math:`t` to mappings from firm IDs :math:`f` to profit gradients. This is only computed
        if these results were created by :meth:`Simulation.replace_endogenous`. The profit gradient for firm :math:`f`
        in market :math:`t` is a :math:`J_{ft}` vector with element :math:`k \in J_{ft}`

        .. math::

           \frac{\partial \pi_{ft}}{\partial p_{kt}}
           = \sum_{j \in J_{ft}} \frac{\partial \pi_{jt}}{\partial p_{kt}}

        where population-normalized profits are

        .. math:: \pi_{jt} = (p_{jt} - c_{jt}) s_{jt}.

        When there is a nontrivial ownership structure, the sum is over all products :math:`j \in J_t` and the terms are
        weighted by the firm's (possibly partial) ownership of product :math:`j`, given by :math:`\mathscr{H}_{jk}`.

    profit_gradient_norms : `dict`
        Mapping from market IDs :math:`t` to mappings from firm IDs :math:`f` to the infinity norm of profit gradients.
        This is only computed if these results were created by :meth:`Simulation.replace_endogenous`. If a norm is near
        to zero, the firm's choice of profits is near to a local optimum.
    profit_hessians : `dict`
        Mapping from market IDs :math:`t` to mappings from firm IDs :math:`f` to profit Hessians. This is only computed
        if these results were created by :meth:`Simulation.replace_endogenous`. The profit Hessian for firm :math:`f` in
        market :math:`t` is a :math:`J_{ft} \times J_{ft}` matrix with element :math:`(k, \ell) \in J_{ft}^2`

        .. math::

           \frac{\partial^2 \pi_{ft}}{\partial p_{kt} \partial p_{\ell t}}
           = \sum_{j \in J_{ft}} \frac{\partial^2 \pi_{jt}}{\partial p_{kt} \partial p_{\ell t}}

        where population-normalized profits are

        .. math:: \pi_{jt} = (p_{jt} - c_{jt}) s_{jt}.

        When there is a nontrivial ownership structure, the sum is over all products :math:`j \in J_t` and the terms are
        weighted by the firm's (possibly partial) ownership of product :math:`j`, given by :math:`\mathscr{H}_{jk}`.

    profit_hessian_eigenvalues : `dict`
        Mapping from market IDs :math:`t` to mappings from firm IDs :math:`f` to the eigenvalues of profit Hessians.
        This is only computed if these results were created by :meth:`Simulation.replace_endogenous`. If the fixed point
        converged and all eigenvalues are negative, the firm's choice of profits is a local maximum.


    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    simulation: 'Simulation'
    product_data: RecArray
    delta: Array
    costs: Optional[Array]
    computation_time: float
    fp_converged: Array
    fp_iterations: Array
    contraction_evaluations: Array
    profit_gradients: Optional[Dict[Hashable, Dict[Hashable, Array]]]
    profit_gradient_norms: Optional[Dict[Hashable, Dict[Hashable, Array]]]
    profit_hessians: Optional[Dict[Hashable, Dict[Hashable, Array]]]
    profit_hessian_eigenvalues: Optional[Dict[Hashable, Dict[Hashable, Array]]]
    _data_override: Dict[str, Array]

    def __init__(
            self, simulation: 'Simulation', data_override: Dict[str, Array], delta: Array, costs: Optional[Array],
            start_time: float, end_time: float, iteration_stats: Dict[Hashable, SolverStats],
            profit_gradients: Optional[Dict[Hashable, Dict[Hashable, Array]]] = None,
            profit_gradient_norms: Optional[Dict[Hashable, Dict[Hashable, Array]]] = None,
            profit_hessians: Optional[Dict[Hashable, Dict[Hashable, Array]]] = None,
            profit_hessian_eigenvalues: Optional[Dict[Hashable, Dict[Hashable, Array]]] = None) -> None:
        """Structure simulation results."""
        super().__init__(
            simulation, simulation._parameters, simulation.sigma, simulation.pi, simulation.rho, simulation.beta,
            simulation.gamma, delta, data_override
        )
        self.simulation = simulation
        self.product_data = update_matrices(
            simulation.product_data,
            {k: (v, v.dtype) for k, v in data_override.items()}
        )
        self.delta = delta
        self.costs = costs
        self.computation_time = end_time - start_time
        self.fp_converged = np.array(
            [iteration_stats[t].converged for t in simulation.unique_market_ids], dtype=np.bool_
        )
        self.fp_iterations = np.array(
            [iteration_stats[t].iterations for t in simulation.unique_market_ids], dtype=np.int64
        )
        self.contraction_evaluations = np.array(
            [iteration_stats[t].evaluations for t in simulation.unique_market_ids], dtype=np.int64
        )
        self.profit_gradients = profit_gradients
        self.profit_gradient_norms = profit_gradient_norms
        self.profit_hessians = profit_hessians
        self.profit_hessian_eigenvalues = profit_hessian_eigenvalues

    def __str__(self) -> str:
        """Format simulation results as a string."""
        header = [
            ("Computation", "Time"),
            ("Fixed Point", "Failures"),
            ("Fixed Point", "Iterations"),
            ("Contraction", "Evaluations"),
        ]
        values = [
            format_seconds(self.computation_time),
            (~self.fp_converged).sum(),
            self.fp_iterations.sum(),
            self.contraction_evaluations.sum(),
        ]

        if self.profit_gradient_norms is not None:
            max_norm = -np.inf
            for profit_gradient_norms_t in self.profit_gradient_norms.values():
                for profit_gradient_norm_ft in profit_gradient_norms_t.values():
                    if np.isfinite(profit_gradient_norm_ft):
                        max_norm = max(max_norm, profit_gradient_norm_ft)

            header.append(("Profit Gradients", "Max Norm"))
            values.append(format_number(max_norm))

        if self.profit_hessian_eigenvalues is not None:
            min_eigenvalue = +np.inf
            max_eigenvalue = -np.inf
            for profit_hessian_eigenvalues_t in self.profit_hessian_eigenvalues.values():
                for profit_hessian_eigenvalues_ft in profit_hessian_eigenvalues_t.values():
                    if np.isfinite(profit_hessian_eigenvalues_ft).any():
                        min_eigenvalue = min(min_eigenvalue, profit_hessian_eigenvalues_ft.min())
                        max_eigenvalue = max(max_eigenvalue, profit_hessian_eigenvalues_ft.max())

            header.extend([("Profit Hessians", "Min Eigenvalue"), ("Profit Hessians", "Max Eigenvalue")])
            values.extend([format_number(min_eigenvalue), format_number(max_eigenvalue)])

        return format_table(header, values, title="Simulation Results Summary")

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
                'product_data', 'delta', 'costs', 'computation_time', 'fp_converged', 'fp_iterations',
                'contraction_evaluations', 'profit_gradients', 'profit_gradient_norms', 'profit_hessians',
                'profit_hessian_eigenvalues'
            )) -> dict:
        """Convert these results into a dictionary that maps attribute names to values.

        Parameters
        ----------
        attributes : `sequence of str, optional`
            Name of attributes that will be added to the dictionary. By default, all :class:`SimulationResults`
            attributes are added except for :attr:`SimulationResults.simulation`.

        Returns
        -------
        `dict`
            Mapping from attribute names to values.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        return {k: getattr(self, k) for k in attributes}

    def to_problem(
            self, product_formulations: Optional[Union[Formulation, Sequence[Optional[Formulation]]]] = None,
            product_data: Optional[Mapping] = None, agent_formulation: Optional[Formulation] = None,
            agent_data: Optional[Mapping] = None, integration: Optional[Integration] = None,
            rc_types: Optional[Sequence[str]] = None, epsilon_scale: Optional[float] = None,
            costs_type: Optional[str] = None, add_exogenous: bool = True) -> 'Problem':
        """Convert the solved simulation into a problem.

        Arguments are the same as those of :class:`Problem`. By default, the structure of the problem will be the same
        as that of the solved simulation.

        By default, some simple "sums of characteristics" BLP instruments are constructed. Demand-side instruments are
        constructed by :func:`build_blp_instruments` from variables in :math:`X_1^{\text{ex}}`, along with any supply
        shifters (variables in :math:`X_3^{\text{ex}}` but not :math:`X_1^{\text{ex}}`). Supply side instruments are
        constructed from variables in :math:`X_3^{\text{ex}}`, along with any demand shifters (variables in
        :math:`X_1^{\text{ex}}` but not :math:`X_3^{\text{ex}}`). Instruments will also be constructed from columns of
        ones if there is variation in :math:`J_t`, the number of products per market. Any constant columns will be
        dropped. For example, if each firm owns exactly one product in each market, the "rival" columns of instruments
        will be zero and hence dropped.

        .. note::

           These excluded instruments are constructed only for convenience. Especially for more complicated problems,
           they should be replaced with better instruments.

        Parameters
        ----------
        product_formulations : `Formulation or sequence of Formulation, optional`
            By default, :attr:`Simulation.product_formulations`.
        product_data : `structured array-like, optional`
            By default, :attr:`SimulationResults.product_data` with excluded instruments.
        agent_formulation : `Formulation, optional`
            By default, :attr:`Simulation.agent_formulation`.
        agent_data : `structured array-like, optional`
            By default, :attr:`Simulation.agent_data`.
        integration : `Integration, optional`
            By default, this is unspecified.
        rc_types : `sequence of str, optional`
            By default, :attr:`Simulation.rc_types`.
        epsilon_scale : `float, optional`
            By default, :attr:`Simulation.epsilon_scale`.
        costs_type : `str, optional`
            By default, :attr:`Simulation.costs_type`.
        add_exogenous : `bool, optional`
            By default, ``True``.

        Returns
        -------
        `Problem`
            A BLP problem.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        if product_formulations is None:
            product_formulations = self.simulation.product_formulations
        if product_data is None:
            demand_instruments, supply_instruments = self._compute_default_instruments()
            product_data = update_matrices(self.product_data, {
                'demand_instruments': (demand_instruments, options.dtype),
                'supply_instruments': (supply_instruments, options.dtype)
            })
            assert product_data is not None
        if agent_formulation is None:
            agent_formulation = self.simulation.agent_formulation
        if agent_data is None:
            agent_data = self.simulation.agent_data
        if rc_types is None:
            rc_types = self.simulation.rc_types
        if epsilon_scale is None:
            epsilon_scale = self.simulation.epsilon_scale
        if costs_type is None:
            costs_type = self.simulation.costs_type
        from ..economies.problem import Problem  # noqa
        return Problem(
            product_formulations, product_data, agent_formulation, agent_data, integration, rc_types, epsilon_scale,
            costs_type, add_exogenous
        )

    def _compute_default_instruments(self) -> Tuple[Array, Array]:
        """Compute default sums of characteristics excluded BLP instruments."""

        # collect exogenous variables names that will be used in instruments
        assert self.simulation.product_formulations[0] is not None
        X1_names = self.simulation.product_formulations[0]._names - {'prices'}
        X3_names: Set[str] = set()
        if self.simulation.product_formulations[2] is not None:
            X3_names = self.simulation.product_formulations[2]._names - {'shares'}

        # determine whether there's variation in the number of products per markets
        J_variation = any(i.size < self.simulation._max_J for i in self.simulation._product_market_indices.values())

        # construct the BLP instruments, dropping any constant columns
        demand_instruments = np.zeros((self.simulation.N, 0), options.dtype)
        supply_instruments = np.zeros((self.simulation.N, 0), options.dtype)
        demand_formula = ' + '.join(['1' if J_variation else '0'] + sorted(X1_names))
        supply_formula = ' + '.join(['1' if J_variation else '0'] + sorted(X3_names))
        if demand_formula != '0':
            demand_instruments = build_blp_instruments(Formulation(demand_formula), self.simulation.product_data)
            demand_instruments = demand_instruments[:, (demand_instruments != demand_instruments[0]).any(axis=0)]
        if supply_formula != '0' and self.simulation.K3 > 0:
            supply_instruments = build_blp_instruments(Formulation(supply_formula), self.simulation.product_data)
            supply_instruments = supply_instruments[:, (supply_instruments != supply_instruments[0]).any(axis=0)]

        # add supply and demand shifters
        supply_shifter_formula = ' + '.join(['0'] + sorted(X3_names - X1_names))
        demand_shifter_formula = ' + '.join(['0'] + sorted(X1_names - X3_names))
        if supply_shifter_formula != '0':
            supply_shifters = build_matrix(Formulation(supply_shifter_formula), self.simulation.product_data)
            demand_instruments = np.c_[demand_instruments, supply_shifters]
        if demand_shifter_formula != '0' and self.simulation.K3 > 0:
            demand_shifters = build_matrix(Formulation(demand_shifter_formula), self.simulation.product_data)
            supply_instruments = np.c_[supply_instruments, demand_shifters]
        return demand_instruments, supply_instruments
