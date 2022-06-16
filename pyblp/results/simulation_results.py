"""Economy-level structuring of BLP simulation results."""

import collections
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import numpy as np
import scipy.sparse

from .results import Results
from .. import exceptions, options
from ..configurations.formulation import Formulation
from ..configurations.integration import Integration
from ..construction import build_blp_instruments, build_matrix
from ..markets.results_market import ResultsMarket
from ..markets.simulation_results_market import SimulationResultsMarket
from ..micro import MicroDataset, MicroMoment, Moments
from ..parameters import Parameters
from ..primitives import Agents
from ..utilities.basics import (
    Array, Error, SolverStats, generate_items, get_indices, Mapping, output, extract_matrix, output_progress,
    structure_matrices, update_matrices, RecArray, format_number, format_seconds, format_table
)


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import Problem  # noqa
    from ..economies.simulation import Simulation  # noqa


class SimulationResults(Results):
    r"""Results of a solved simulation of synthetic BLP data.

    This class has the same methods as :class:`ProblemResults` that compute post-estimation outputs in one or more
    markets, but not other methods like :meth:`ProblemResults.compute_optimal_instruments` that do not make sense in a
    simulated dataset.

    In addition, the :meth:`SimulationResults.to_problem` method can be used to convert the full set of simulated data
    (along with some basic default instruments) and configured information into a :class:`Problem`.

    The :meth:`SimulationResults.replace_micro_moment_values` method can be used to compute simulated micro moment
    values, :meth:`SimulationResults.build_micro_data` can be used to simulate data underlying a micro dataset, and
    :meth:`SimulationResults.compute_micro_scores` can be used to compute scores for micro data.

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
        super().__init__(simulation, simulation._parameters)
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
        self._data_override = data_override

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

    def _combine_arrays(
            self, compute_market_results: Callable, market_ids: Array, fixed_args: Sequence = (),
            market_args: Sequence = (), agent_data: Optional[Mapping] = None,
            integration: Optional[Integration] = None) -> Array:
        """Compute arrays for one or all markets and stack them into a single array. An array for a single market is
        computed by passing fixed_args (identical for all markets) and market_args (matrices with as many rows as there
        are products that are restricted to the market) to compute_market_results, a ResultsMarket method that returns
        the output for the market any errors encountered during computation. Agent data and an integration configuration
        can be optionally specified to override agent data.
        """
        errors: List[Error] = []

        # keep track of how long it takes to compute the arrays
        start_time = time.time()

        # structure or construct different agent data
        if agent_data is None and integration is None:
            agents = self.simulation.agents
            agents_market_indices = self.simulation._agent_market_indices
        else:
            agents = Agents(self.simulation.products, self.simulation.agent_formulation, agent_data, integration)
            agents_market_indices = get_indices(agents.market_ids)

        def market_factory(s: Hashable) -> tuple:
            """Build a market along with arguments used to compute arrays."""
            indices_s = self.simulation._product_market_indices[s]
            market_s = ResultsMarket(
                self.simulation, s, self._parameters, self.simulation.sigma, self.simulation.pi, self.simulation.rho,
                self.simulation.beta, self.simulation.gamma, self.delta, self._data_override,
                agents_override=agents[agents_market_indices[s]]
            )
            if market_ids.size == 1:
                args_s = market_args
            else:
                args_s = [None if a is None else a[indices_s] for a in market_args]
            return (market_s, *fixed_args, *args_s)

        # construct a mapping from market IDs to market-specific arrays
        array_mapping: Dict[Hashable, Array] = {}
        generator = generate_items(market_ids, market_factory, compute_market_results)
        if market_ids.size > 1:
            generator = output_progress(generator, market_ids.size, start_time)
        for t, (array_t, errors_t) in generator:
            array_mapping[t] = np.c_[array_t]
            errors.extend(errors_t)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # determine the sizes of dimensions
        dimension_sizes = []
        for dimension in range(len(array_mapping[market_ids[0]].shape)):
            if dimension == 0:
                dimension_sizes.append(sum(array_mapping[t].shape[dimension] for t in market_ids))
            else:
                dimension_sizes.append(max(array_mapping[t].shape[dimension] for t in market_ids))

        # preserve the original product order or the sorted market order when stacking the arrays
        combined = np.full(dimension_sizes, np.nan, options.dtype)
        for t, array_t in array_mapping.items():
            slices = (slice(0, s) for s in array_t.shape[1:])
            if dimension_sizes[0] == market_ids.size:
                combined[(market_ids == t, *slices)] = array_t
            elif dimension_sizes[0] == self.simulation.N:
                combined[(self.simulation._product_market_indices[t], *slices)] = array_t
            else:
                assert market_ids.size == 1
                combined = array_t

        # output how long it took to compute the arrays
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return combined

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
                'product_data', 'computation_time', 'fp_converged', 'fp_iterations', 'contraction_evaluations'
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

    def replace_micro_moment_values(self, micro_moments: Sequence[MicroMoment]) -> List[MicroMoment]:
        r"""Compute simulated micro moment values :math:`v_m`.

        Parameters
        ----------
        micro_moments : `sequence of MicroMoment`
            :class:`MicroMoment` instances. The ``value`` argument will be replaced and is hence ignored.

        Returns
        -------
        `list of MicroMoment`
            The same :class:`MicroMoment` instances but with their values replaced by simulated values.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        errors: List[Error] = []

        # keep track of long it takes to compute micro moment values
        output("Replacing micro moment values ...")
        start_time = time.time()

        # validate and structure micro moments
        moments = Moments(micro_moments, self.simulation)
        if moments.MM == 0:
            return []

        def market_factory(s: Hashable) -> Tuple[SimulationResultsMarket, Moments]:
            """Build a market along with arguments used to compute micro moment values."""
            market_s = SimulationResultsMarket(
                self.simulation, s, self._parameters, self.simulation.sigma, self.simulation.pi, self.simulation.rho,
                self.simulation.beta, self.simulation.gamma, self.delta, self._data_override
            )
            return market_s, moments

        # compute micro moments values market-by-market
        micro_numerator_mapping: Dict[Hashable, Array] = {}
        micro_denominator_mapping: Dict[Hashable, Array] = {}
        generator = generate_items(
            self.simulation.unique_market_ids, market_factory,
            SimulationResultsMarket.safely_compute_micro_contributions
        )
        for t, (micro_numerator_t, micro_denominator_t, errors_t) in generator:
            micro_numerator_mapping[t] = scipy.sparse.csr_matrix(micro_numerator_t)
            micro_denominator_mapping[t] = scipy.sparse.csr_matrix(micro_denominator_t)
            errors.extend(errors_t)

        # aggregate micro moments across all markets (this is done after market-by-market computation to preserve
        #   numerical stability with different market orderings)
        with np.errstate(all='ignore'):
            micro_numerator = scipy.sparse.csr_matrix((moments.MM, 1), dtype=options.dtype)
            micro_denominator = scipy.sparse.csr_matrix((moments.MM, 1), dtype=options.dtype)
            for t in self.simulation.unique_market_ids:
                micro_numerator += micro_numerator_mapping[t]
                micro_denominator += micro_denominator_mapping[t]

            micro_numerator = micro_numerator.toarray()
            micro_denominator = micro_denominator.toarray()
            micro_values = micro_numerator / micro_denominator

        # construct new micro moments
        updated_micro_moments: List[MicroMoment] = []
        for micro_moment, value in zip(moments.micro_moments, micro_values.flatten()):
            updated_micro_moments.append(MicroMoment(
                micro_moment.name, micro_moment.dataset, value, micro_moment.compute_values
            ))

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # output how long it took to compute the micro values
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return updated_micro_moments

    def build_micro_data(
            self, dataset: Optional[MicroDataset] = None, integration: Optional[Integration] = None,
            seed: Optional[int] = None) -> RecArray:
        r"""Construct micro data for score computation.

        By default, this method builds micro data with each observation corresponding to an agent in
        :attr:`Simulation.agent_data`. These data can be passed to :meth:`SimulationResults.compute_micro_scores` to
        compute scores for each agent and choice.

        If a ``dataset`` is specified, micro observations underlying the dataset are simulated according to survey
        weights :math:`w_{dijt}`, agent weights :math:`w_{it}`, choice probabilities :math:`s_{ijt}`, and if the
        dataset contains second choice data, second choice probabilities as well. These data can also be passed to
        :meth:`SimulationResults.compute_micro_scores` to compute scores for each observation.

        If ``integration`` is specified, each observation will be duplicated by as many rows as there are nodes created
        by the integration configuration, with nodes and weights replaced by those constructed by the configuration.
        These replicated rows correspond to an integral over unobserved heterogeneity for each observation.

        Parameters
        ----------
        dataset : `MicroDataset, optional`
            The :class:`MicroDataset` for which micro data will be simulated. By default, micro data are built without
            reference to a micro dataset, but rather with each observation corresponding to an agent in
            :attr:`Simulation.agent_data`. If specified, micro observations will be simulated that underlie the dataset.
        integration : `Integration, optional`
            :class:`Integration` configuration for how to build nodes and weights for integration over unobserved
            heterogeneity for each observation. By default, there is a single node for each observation with
            integration weight equal to ``1``. Intuitively, this means that nodes are treated as observed
            heterogeneity, just as demographics. To treat nodes as unobserved heterogeneity, either specify this
            configuration, or manually duplicate each row of the returned micro data with different integration nodes.
        seed : `int, optional`
            Passed to :class:`numpy.random.RandomState` to seed the random number generator before data are simulated.
            By default, a seed is not passed to the random number generator. A seed is only used if the ``dataset`` is
            specified.

        Returns
        -------
        `recarray`
            Micro data with as many rows as :attr:`Simulation.agent_data` if ``dataset`` is not specified, and otherwise
            with as many rows as the dataset's ``observations``. Fields:

            - **micro_ids** : (`object`) - IDs corresponding to observations :math:`n`. If ``dataset`` is not specified,
              the IDs correspond to indices of :attr:`Simulation.agent_data`, from zero to one minus the number of rows.
              Otherwise, they go from zero to one minus the number of ``observations`` in the dataset.

            - **market_ids** : (`object`) - Market IDs :math:`t_n` for each observation :math:`n`.

            - **weights** : (`numeric`) - Weights for integration over unobserved heterogeneity for each observation.
              If ``integration`` is not specified, these will all be equal to ``1``. Otherwise, they will be constructed
              by the :class:`Integration` configuration.

            - **agent_indices** : (`int`) - Within-market indices of agents :math:`i_n` that take on values from
              :math:`0` to :math:`I_t - 1`. The ordering is the same as agents within :attr:`Simulation.agent_data`.

            If a ``dataset`` is specified, choices will also be simulated:

            - **choice_indices** : (`int`) - Within-market indices of simulated choices :math:`j_n`. If
              ``compute_weights`` in the ``dataset`` returns an array with :math:`J_t` elements in its second axis, then
              choice indices take on values from :math:`0` to :math:`J_t - 1` where :math:`0` corresponds to the first
              inside good. If it returns an array with :math:`1 + J_t` elements in its second axis, then choice indices
              take on values from :math:`0` to :math:`J_t` where :math:`0` corresponds to the outside good. The ordering
              of inside goods is the same as products within ``product_data`` passed to :class:`Simulation`.

            - **second_choice_indices** : (`int`) - Within-market indices of simulated second choices :math:`k_n`, if
              the dataset contains second choice data. If ``compute_weights`` in the ``dataset`` returns an array with
              :math:`J_t` elements in its third axis, then second choice indices take on values from :math:`0` to
              :math:`J_t - 1` where :math:`0` corresponds to the first inside good. If it returns an array with
              :math:`1 + J_t` elements in its third axis, then second choice indices take on values from :math:`0` to
              :math:`J_t` where :math:`0` corresponds to the outside good. The ordering of inside goods is the same as
              products within ``product_data`` passed to :class:`Simulation`.

            Other fields, such as ``nodes`` and any demographics, are extracted from :attr:`Simulation.agent_data`
            according to the ``market_ids`` and ``agent_indices`` fields.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        errors: List[Error] = []

        # keep track of long it takes to build micro data
        output("Building micro data ...")
        start_time = time.time()

        # validate any integration configuration
        if integration is not None and not isinstance(integration, Integration):
            raise ValueError(f"integration must be None or an Integration instance.")

        # determine the datatypes to use to conserve on memory
        agent_dtype = choice_dtype = np.uint64
        for dtype in [np.uint32, np.uint8]:
            if self.simulation._max_I <= np.iinfo(dtype).max:
                agent_dtype = dtype
            if self.simulation._max_J <= np.iinfo(dtype).max:
                choice_dtype = dtype

        # if there isn't a micro dataset, simply build micro data from agents
        agent_data = self.simulation.agent_data
        agent_market_indices = self.simulation._agent_market_indices
        micro_data_mapping: Dict[str, Tuple[Array, Any]] = collections.OrderedDict()
        if dataset is None:
            # verify that there are agent data
            if agent_data is None:
                raise RuntimeError("The simulation does not have any agent data from which to build micro data.")

            # construct the micro data
            micro_data_mapping['micro_ids'] = (np.arange(agent_data.size), np.object_)
            micro_data_mapping['market_ids'] = (agent_data.market_ids, np.object_)
            micro_data_mapping['weights'] = (np.ones(agent_data.size), options.dtype)

            # add nodes and demographics from agent data
            for key in agent_data.dtype.names:
                if key in micro_data_mapping or agent_data[key].size == 0:
                    continue
                micro_data_mapping[key] = (agent_data[key], agent_data[key].dtype)

            # add agent indices
            micro_data_mapping['agent_indices'] = (np.zeros(agent_data.size), agent_dtype)
            for indices in agent_market_indices.values():
                micro_data_mapping['agent_indices'][0][indices] = np.arange(indices.size)
        else:
            # validate the micro dataset
            if not isinstance(dataset, MicroDataset):
                raise TypeError("dataset must be a MicroDataset.")
            dataset._validate(self.simulation)

            # collect the relevant market ids
            if dataset.market_ids is None:
                market_ids = self.simulation.unique_market_ids
            else:
                market_ids = np.asarray(list(dataset.market_ids))

            def market_factory(s: Hashable) -> Tuple[SimulationResultsMarket, MicroDataset]:
                """Build a market along with arguments used to compute weights needed for simulation."""
                assert dataset is not None
                market_s = SimulationResultsMarket(
                    self.simulation, s, self._parameters, self.simulation.sigma, self.simulation.pi,
                    self.simulation.rho, self.simulation.beta, self.simulation.gamma, self.delta, self._data_override
                )
                return market_s, dataset

            # construct mappings from market IDs to probabilities, IDs, and indices needed for simulation
            weights_mapping: Dict[Hashable, Array] = {}
            agent_indices_mapping: Dict[Hashable, Array] = {}
            choice_indices_mapping: Dict[Hashable, Array] = {}
            second_choice_indices_mapping: Dict[Hashable, Array] = {}
            generator = generate_items(market_ids, market_factory, SimulationResultsMarket.safely_compute_micro_weights)
            if market_ids.size > 1:
                generator = output_progress(generator, market_ids.size, start_time)
            for t, (weights_t, errors_t) in generator:
                errors.extend(errors_t)
                indices_t = np.nonzero(weights_t)
                weights_mapping[t] = weights_t[indices_t]
                agent_indices_mapping[t] = indices_t[0].astype(agent_dtype)
                choice_indices_mapping[t] = indices_t[1].astype(choice_dtype)
                if len(indices_t) == 3:
                    second_choice_indices_mapping[t] = indices_t[2].astype(choice_dtype)

            # output a warning about any errors
            if errors:
                output("")
                output(exceptions.MultipleErrors(errors))
                output("")

            # simulate choices
            state = np.random.RandomState(seed)
            weights_data = np.concatenate([weights_mapping[t] for t in market_ids])
            choices = state.choice(weights_data.size, p=weights_data / weights_data.sum(), size=dataset.observations)

            # construct the micro data
            micro_data_mapping['micro_ids'] = (np.arange(dataset.observations), np.object_)
            micro_data_mapping['market_ids'] = (
                np.concatenate([np.full(agent_indices_mapping[t].size, t) for t in market_ids])[choices], np.object_
            )
            micro_data_mapping['weights'] = (np.ones(dataset.observations), options.dtype)

            # add nodes and demographics from agent data
            if agent_data is not None:
                for key in agent_data.dtype.names:
                    if key in micro_data_mapping or agent_data[key].size == 0:
                        continue
                    values = {t: agent_data[key][agent_market_indices[t]][agent_indices_mapping[t]] for t in market_ids}
                    micro_data_mapping[key] = (
                        np.concatenate([values[t] for t in market_ids])[choices], agent_data[key].dtype
                    )

            # add agent and choice indices
            micro_data_mapping['agent_indices'] = (
                np.concatenate([agent_indices_mapping[t] for t in market_ids])[choices], agent_dtype
            )
            micro_data_mapping['choice_indices'] = (
                np.concatenate([choice_indices_mapping[t] for t in market_ids])[choices], choice_dtype
            )
            if second_choice_indices_mapping:
                micro_data_mapping['second_choice_indices'] = (
                    np.concatenate([second_choice_indices_mapping[t] for t in market_ids])[choices], choice_dtype
                )

        # optionally duplicate each row, replacing unobserved heterogeneity
        if integration is not None and self.simulation.K2 > 0:
            micro_ids, nodes, weights = integration._build_many(self.simulation.K2, micro_data_mapping['micro_ids'][0])
            repeats = np.bincount(micro_ids)
            micro_data_mapping['micro_ids'] = (micro_ids, np.object_)
            micro_data_mapping['nodes'] = (nodes, nodes.dtype)
            micro_data_mapping['weights'] = (weights, weights.dtype)
            for key, (values, dtype) in micro_data_mapping.items():
                if key not in {'micro_ids', 'nodes', 'weights'}:
                    micro_data_mapping[key] = (np.repeat(values, repeats, axis=0), dtype)

        # build the micro data and output how long this took
        micro_data = structure_matrices(micro_data_mapping)
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return micro_data

    def compute_micro_scores(self, dataset: MicroDataset, micro_data: Mapping) -> Array:
        r"""Compute scores for each observation in micro data.

        In micro dataset :math:`d`, the score for observation :math:`n \in N_d` is

        .. math::
           :label: score

           \mathscr{S}_n = \frac{\partial\log\mathscr{P}_n}{\partial\theta'},

        in which :math:`\theta` are nonlinear parameters and the conditional probability of observation :math:`n` is

        .. math::

           \mathscr{P}_n = \frac{
               \sum_{i \in I_n} w_{it_n} s_{ij_nt_n} w_{dij_nt_n}
           }{
               \sum_{t \in T} \sum_{i \in I_t} \sum_{j \in J_t \cup \{0\}} w_{it} s_{ijt} w_{dijt}
           }

        where :math:`i \in I_n` integrates over unobserved heterogeneity for observation :math:`n`.

        Parameters
        ----------
        dataset : `MicroDataset`
            The :class:`MicroDataset` for which scores will be computed. The ``compute_weights`` function will be called
            separately for each observation :math:`n`. There can be different number of observations in ``micro_data``
            than suggested by the ``dataset``.
        micro_data : `structured array-like, optional`
            Micro data underlying the ``dataset``, which may either be real micro data or built by
            :meth:`SimulationResults.build_micro_data`.

            The ``micro_ids`` field corresponds to observations :math:`n`. Different rows for the same :math:`n`
            correspond to :math:`i \in I_n` that integrate over unobserved heterogeneity. In addition to ``nodes`` and
            any demographic variables that are used by the :class:`Simulation`, the following fields are required:

                - **micro_ids** : (`object`) - IDs corresponding to observations :math:`n`.

                - **market_ids** : (`object`) - Market IDs :math:`t_n` for each observation :math:`n`.

                - **weights** : (`numeric`) - Weights :math:`w_{it_n}` for integration over unobserved heterogeneity
                  :math:`i \in I_n` for each observation.

            If choice indices are not specified, scores will be computed for each possible choice :math:`j_n`.
            Otherwise, scores will be computed for just the specified choice for each observation:

                - **choice_indices** : (`int`) - Within-market indices of choices :math:`j_n`. If ``compute_weights`` in
                  the ``dataset`` returns an array with :math:`J_t` elements in its second axis, then choice indices
                  take on values from :math:`0` to :math:`J_t - 1` where :math:`0` corresponds to the first inside good.
                  If it returns an array with :math:`1 + J_t` elements in its second axis, then choice indices take on
                  values from :math:`0` to :math:`J_t` where :math:`0` corresponds to the outside good. The ordering of
                  inside goods is the same as products within ``product_data`` passed to :class:`Simulation`.

            If the ``dataset`` configuration uses second choice data and second choice indices are not specified, scores
            will ge compute for each possible second choice :math:`k_n`. Otherwise, scores will be computed for just the
            specified second choice for each observation:

                - **second_choice_indices** : (`int`) - Within-market indices of second choices :math:`k_n`. If
                  ``compute_weights`` in the ``dataset`` returns an array with :math:`J_t` elements in its third axis,
                  then second choice indices take on values from :math:`0` to :math:`J_t - 1` where :math:`0`
                  corresponds to the first inside good. If it returns an array with :math:`1 + J_t` elements in its
                  third axis, then second choice indices take on values from :math:`0` to :math:`J_t` where :math:`0`
                  corresponds to the outside good. The ordering of inside goods is the same as products within
                  ``product_data`` passed to :class:`Simulation`.

        Returns
        -------
        `ndarray`
            Scores :math:`\mathscr{S}_n`. Rows correspond to elements of nonlinear parameters in :math:`\theta` in the
            following order (the same as for :attr:`ProblemResults.theta`): :math:`\hat{\Sigma}`, :math:`\hat{\Pi}`,
            :math:`\hat{\rho}`. Columns are in the same order as the sorted unique value sof ``micro_ids`` in
            ``micro_data``.

            If ``choice_indices`` are not specified, there will be a third dimension with indices that correspond to
            ``choice_indices``. Similarly, if the ``dataset`` configuration uses second choice data and
            ``second_choice_indices`` are not specified, there will be an additional dimension with indices that
            correspond to ``second_choice_indices``. For both of these extra dimensions, if a market has fewer products
            indexed by the ``dataset`` than others, extra indices will contain ``numpy.nan``.

            Ill-defined scores :math:`\mathscr{S}_n` with :math:`\mathscr{P}_n = 0` will be also be ``numpy.nan``. If
            the ``micro_data`` were generated according to the weights specified by the ``dataset``, there should be no
            such ill-defined scores. On the other hand, if the ``micro_data`` were built by
            :attr:`SimulationResults.build_micro_data` without reference to a `dataset, there may be ill-defined scores
            that should be set to some arbitrary value (which will be multiplied by zero weights in the same
            ``dataset``) if being used to create micro moments.

        """
        errors: List[Error] = []

        # keep track of long it takes to compute scores
        output("Computing micro scores ...")
        start_time = time.time()

        # validate the micro dataset
        if not isinstance(dataset, MicroDataset):
            raise TypeError("dataset must be a MicroDataset.")
        dataset._validate(self.simulation)

        # only compute scores for nonlinear parameters
        nonlinear_parameters = Parameters(
            self.simulation, self.simulation.sigma, self.simulation.pi, self.simulation.rho, allow_linear_nans=True,
            check_alpha=False
        )
        if nonlinear_parameters.P == 0:
            raise RuntimeError("There are no nonlinear parameters for which to compute scores.")

        # validate the micro data
        micro_ids = extract_matrix(micro_data, 'micro_ids')
        market_ids = extract_matrix(micro_data, 'market_ids')
        choice_indices = extract_matrix(micro_data, 'choice_indices')
        second_choice_indices = extract_matrix(micro_data, 'second_choice_indices')
        if micro_ids is None:
            raise KeyError("micro_data must have a micro_ids field.")
        if micro_ids.shape[1] > 1:
            raise ValueError("The micro_ids field of micro_data must be one-dimensional.")
        if market_ids is None:
            raise KeyError("micro_data must have a market_ids field.")
        if market_ids.shape[1] > 1:
            raise ValueError("The market_ids field of micro_data must be one-dimensional.")
        if choice_indices is not None and choice_indices.shape[1] > 1:
            raise ValueError("The choice_indices field of micro_data must be one-dimensional.")
        if second_choice_indices is not None and second_choice_indices.shape[1] > 1:
            raise ValueError("The second_choice_indices field of micro_data must be one-dimensional.")

        # pre-compute unique micro observation IDs and indices
        unique_micro_ids = np.unique(micro_ids)
        unique_market_ids = np.unique(market_ids)
        micro_indices = get_indices(micro_ids)

        # validate that market IDs and choices are the same within market
        for n, indices_n in micro_indices.items():
            unique_market_ids_n = np.unique(market_ids[indices_n])
            if unique_market_ids_n.size > 1:
                raise ValueError(f"Micro ID '{n}' has more than one market ID: {list(unique_market_ids_n)}.")
            if choice_indices is not None:
                unique_choice_indices_n = np.unique(choice_indices[indices_n])
                if unique_choice_indices_n.size > 1:
                    raise ValueError(f"Micro ID '{n}' has more than one choice index: {list(unique_choice_indices_n)}.")
            if second_choice_indices is not None:
                unique_second_choice_indices = np.unique(second_choice_indices[indices_n])
                if unique_second_choice_indices.size > 1:
                    raise ValueError(
                        f"Micro ID '{n}' has more than one second choice index {list(unique_second_choice_indices)}."
                    )

        def denominator_market_factory(s: Hashable) -> Tuple[SimulationResultsMarket, MicroDataset]:
            """Build a market along with arguments used to compute denominator contributions."""
            market_s = SimulationResultsMarket(
                self.simulation, s, nonlinear_parameters, self.simulation.sigma, self.simulation.pi,
                self.simulation.rho, self.simulation.beta, self.simulation.gamma, self.delta, self._data_override
            )
            return market_s, dataset

        # construct mappings from market IDs to xi Jacobians and denominator contributions
        output("Computing contributions from the denominator of observation probabilities ...")
        xi_jacobian_mapping: Dict[Hashable, Array] = {}
        denominator_mapping: Dict[Hashable, Array] = {}
        denominator_jacobian_mapping: Dict[Hashable, Array] = {}
        generator = generate_items(
            unique_market_ids, denominator_market_factory,
            SimulationResultsMarket.safely_compute_score_denominator_contributions
        )
        for t, (xi_jacobian_t, denominator_t, denominator_jacobian_t, errors_t) in generator:
            xi_jacobian_mapping[t] = xi_jacobian_t
            denominator_mapping[t] = denominator_t
            denominator_jacobian_mapping[t] = denominator_jacobian_t
            errors.extend(errors_t)

        def numerator_market_factory(i: Hashable) -> tuple:
            """Build a market for a micro observation along with arguments used to numerator contributions."""
            assert market_ids is not None
            t_i = market_ids[micro_indices[i]].take(0)
            j_i = None if choice_indices is None else choice_indices[micro_indices[i]].take(0)
            k_i = None if second_choice_indices is None else second_choice_indices[micro_indices[i]].take(0)
            products_i = self.simulation.products[self.simulation._product_market_indices[t_i]]
            agents_i = Agents(products_i, self.simulation.agent_formulation, micro_data[micro_indices[i]])
            market_i = SimulationResultsMarket(
                self.simulation, t_i, nonlinear_parameters, self.simulation.sigma, self.simulation.pi,
                self.simulation.rho, self.simulation.beta, self.simulation.gamma, self.delta, self._data_override,
                products_i, agents_i,
            )
            return market_i, dataset, j_i, k_i, xi_jacobian_mapping[t_i]

        # construct mappings from observations to numerator contributions
        output("Computing contributions from the numerator of observation probabilities ...")
        numerator_mapping: Dict[Hashable, Array] = {}
        numerator_jacobian_mapping: Dict[Hashable, Array] = {}
        generator = generate_items(
            unique_micro_ids, numerator_market_factory,
            SimulationResultsMarket.safely_compute_score_numerator_contributions
        )
        if unique_micro_ids.size > 1:
            generator = output_progress(generator, unique_micro_ids.size, start_time)
        for n, (numerator_n, numerator_jacobian_n, errors_n) in generator:
            numerator_mapping[n] = numerator_n
            numerator_jacobian_mapping[n] = numerator_jacobian_n
            errors.extend(errors_n)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # determine the sizes of numerator contribution dimensions
        dimension_sizes = []
        for dimension in range(len(numerator_mapping[unique_micro_ids[0]].shape)):
            dimension_sizes.append(max(numerator_mapping[n].shape[dimension] for n in unique_micro_ids))

        # pad numerator contributions
        if dimension_sizes:
            for n in unique_micro_ids:
                pad = np.full(dimension_sizes, np.nan, options.dtype)
                pad_jacobian = np.full(dimension_sizes + [nonlinear_parameters.P], np.nan, options.dtype)
                pad[[slice(0, s) for s in numerator_mapping[n].shape]] = numerator_mapping[n]
                pad_jacobian[[slice(0, s) for s in numerator_jacobian_mapping[n].shape]] = numerator_jacobian_mapping[n]
                numerator_mapping[n] = pad
                numerator_jacobian_mapping[n] = pad_jacobian

        # stack the contributions
        numerator = np.stack([numerator_mapping[n] for n in unique_micro_ids])
        numerator_jacobian = np.stack([numerator_jacobian_mapping[n] for n in unique_micro_ids])
        denominator = np.stack([denominator_mapping[t] for t in unique_market_ids]).sum()
        denominator_jacobian = np.stack([denominator_jacobian_mapping[t] for t in unique_market_ids]).sum(axis=0)

        # guarantee that places where there will be division by zero are nan (these scores end up not mattering
        #   since when used they will be weighted by zero by the same dataset)
        numerator_zeros = numerator == 0
        numerator[numerator_zeros] = 1
        numerator_jacobian[numerator_zeros] = np.nan
        if denominator == 0:
            denominator = 1
            denominator_jacobian[:] = np.nan

        # construct the scores, moving the dimension indexing parameters to the beginning
        scores = np.moveaxis(numerator_jacobian / numerator[..., None] - denominator_jacobian / denominator, -1, 0)

        # output how long it took to compute scores
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return scores
