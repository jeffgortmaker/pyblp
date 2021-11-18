"""Economy-level structuring of BLP simulation results."""

from pathlib import Path
import pickle
import time
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import numpy as np

from .results import Results
from .. import exceptions, options
from ..configurations.formulation import Formulation
from ..configurations.integration import Integration
from ..construction import build_blp_instruments, build_matrix
from ..markets.results_market import ResultsMarket
from ..markets.simulation_results_market import SimulationResultsMarket
from ..moments import Moment, EconomyMoments
from ..primitives import Agents
from ..utilities.basics import (
    Array, Error, SolverStats, generate_items, get_indices, Mapping, output, output_progress, update_matrices, RecArray,
    format_seconds, format_table
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
    (along with some basic default instruments) and configured information into a :class:`Problem`. The
    :meth:`SimulationResults.compute_micro_values` method can be used to compute simulated micro moment values.

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

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    simulation: 'Simulation'
    product_data: RecArray
    delta: Array
    costs: Array
    computation_time: float
    fp_converged: Array
    fp_iterations: Array
    contraction_evaluations: Array
    _data_override: Dict[str, Array]

    def __init__(
            self, simulation: 'Simulation', data_override: Dict[str, Array], delta: Array, costs: Array,
            start_time: float, end_time: float, iteration_stats: Dict[Hashable, SolverStats]) -> None:
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
        self._data_override = data_override

    def __str__(self) -> str:
        """Format simulation results as a string."""
        header = [("Computation", "Time"), ("Fixed Point", "Iterations"), ("Contraction", "Evaluations")]
        values = [
            format_seconds(self.computation_time),
            self.fp_iterations.sum(),
            self.contraction_evaluations.sum()
        ]
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
                self.simulation.beta, self.simulation.gamma, self.delta,
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
            distributions: Optional[Sequence[str]] = None, epsilon_scale: Optional[float] = None,
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
        distributions : `sequence of str, optional`
            By default, :attr:`Simulation.distributions`.
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
        if distributions is None:
            distributions = self.simulation.distributions
        if epsilon_scale is None:
            epsilon_scale = self.simulation.epsilon_scale
        if costs_type is None:
            costs_type = self.simulation.costs_type
        from ..economies.problem import Problem  # noqa
        return Problem(
            product_formulations, product_data, agent_formulation, agent_data, integration, distributions,
            epsilon_scale, costs_type, add_exogenous
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

    def compute_micro_values(self, micro_moments: Sequence[Moment]) -> Array:
        r"""Compute simulated micro moment values :math:`v_m`.

        Parameters
        ----------
        micro_moments : `sequence of Moment`
            Configurations for the micro moments. For a list of supported moments, refer to
            :ref:`api:Micro Moment Classes`. Since only simulated values :math:`v_m` are computed, the ``value`` and
            ``observations`` specified when initializing the moments will be ignored.

        Returns
        -------
        `ndarray`
            Micro moment values :math:`v_m`, which are in the same order as ``micro_moments``. If there are multiple
            markets :math:`t \in T_m` in which a micro moment is relevant, this is an average of market-specific
            :math:`v_{mt}` values.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        errors: List[Error] = []

        # keep track of long it takes to compute micro moment values
        output("Computing micro moment values ...")
        start_time = time.time()

        # validate and structure micro moments before outputting related information
        moments = EconomyMoments(self.simulation, micro_moments)
        if moments.MM == 0:
            raise ValueError("At least one micro moment should be specified.")
        output("")
        output(moments.format("Micro Moments"))

        # compute the mean utility
        delta = self.simulation._compute_true_X1(self._data_override) @ self.simulation.beta + self.simulation.xi

        def market_factory(s: Hashable) -> Tuple[SimulationResultsMarket]:
            """Build a market along with arguments used to compute micro moment values."""
            market_s = SimulationResultsMarket(
                self.simulation, s, self.simulation._parameters, self.simulation.sigma, self.simulation.pi,
                self.simulation.rho, delta=delta, moments=moments, data_override=self._data_override
            )
            return market_s,

        # compute micro moments values market-by-market
        market_micro_values = np.full((self.simulation.T, moments.MM), np.nan, options.dtype)
        generator = generate_items(
            self.simulation.unique_market_ids, market_factory, SimulationResultsMarket.safely_compute_micro_values
        )
        for t, (micro_values_t, errors_t) in generator:
            market_micro_values[self.simulation._market_indices[t], moments.market_indices[t]] = micro_values_t.flat
            errors.extend(errors_t)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # compute averages
        micro_values = np.zeros(moments.MM, options.dtype)
        with np.errstate(all='ignore'):
            for t in self.simulation.unique_market_ids:
                indices = moments.market_indices[t]
                weights = moments.market_weights[t]
                if indices.size > 0:
                    micro_values[indices] += weights * market_micro_values[self.simulation._market_indices[t], indices]

        # output how long it took to compute the micro moments
        end_time = time.time()
        output("")
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return micro_values
