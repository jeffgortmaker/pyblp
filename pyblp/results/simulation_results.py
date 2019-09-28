"""Economy-level structuring of BLP simulation results."""

import time
from typing import Dict, Hashable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import numpy as np

from .. import exceptions, options
from ..configurations.formulation import Formulation
from ..configurations.integration import Integration
from ..construction import build_blp_instruments, build_matrix
from ..markets.simulation_results_market import SimulationResultsMarket
from ..moments import Moment, EconomyMoments
from ..utilities.basics import (
    Array, Error, SolverStats, generate_items, Mapping, output, update_matrices, RecArray, StringRepresentation,
    format_seconds, format_table
)


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import Problem  # noqa
    from ..economies.simulation import Simulation  # noqa


class SimulationResults(StringRepresentation):
    r"""Results of a solved simulation of synthetic BLP data.

    The :meth:`SimulationResults.to_problem` method can be used to convert the full set of simulated data (along with
    some basic default instruments) and configured information into a :class:`Problem`.

    Attributes
    ----------
    simulation : `Simulation`
        :class:`Simulation` that created these results.
    product_data : `recarray`
        Simulated :attr:`Simulation.product_data` with product characteristics replaced so as to be consistent with the
        true parameters. If :meth:`Simulation.replace_endogenous` was used to create these results, prices and
        marketshares were replaced. If :meth:`Simulation.replace_exogenous` was used, exogenous characteristics were
        replaced instead.
    computation_time : `float`
        Number of seconds it took to compute prices and marketshares.
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
    computation_time: float
    fp_converged: Array
    fp_iterations: Array
    contraction_evaluations: Array
    _data_override: Dict[str, Array]

    def __init__(
            self, simulation: 'Simulation', data_override: Dict[str, Array], start_time: float, end_time: float,
            iteration_stats: Dict[Hashable, SolverStats]) -> None:
        """Structure simulation results."""
        self.simulation = simulation
        self.product_data = update_matrices(
            simulation.product_data,
            {k: (v, v.dtype) for k, v in data_override.items()}
        )
        self.computation_time = end_time - start_time
        self.fp_converged = np.array(
            [iteration_stats[t].converged for t in simulation.unique_market_ids], dtype=np.bool
        )
        self.fp_iterations = np.array(
            [iteration_stats[t].iterations for t in simulation.unique_market_ids], dtype=np.int
        )
        self.contraction_evaluations = np.array(
            [iteration_stats[t].evaluations for t in simulation.unique_market_ids], dtype=np.int
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

    def to_dict(
            self, attributes: Sequence[str] = (
                'product_data', 'computation_time', 'fp_converged', 'fp_iterations', 'contraction_evaluations'
            )) -> dict:
        """Convert these results into a dictionary that maps attribute names to values.

        Once converted to a dictionary, these results can be saved to a file with :func:`pickle.dump`.

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
            costs_type: Optional[str] = None) -> 'Problem':
        """Convert the solved simulation into a problem.

        Arguments are the same as those of :class:`Problem`. By default, the structure of the problem will be the same
        as that of the solved simulation.

        By default, some simple "sums of characteristics" BLP instruments are constructed. Demand-side instruments are
        constructed by :func:`build_blp_instruments` from variables in :math:`X_1^x`, along with any supply shifters
        (variables in :math:`X_3` but not :math:`X_1`). Supply side instruments are constructed from variables in
        :math:`X_3`, along with any demand shifters (variables in :math:`X_1` but not :math:`X_3`). Instruments will
        also be constructed from columns of ones if there is variation in :math:`J_t`, the number of products per
        market. Any constant columns will be dropped. For example, if each firm owns exactly one product in each market,
        the "rival" columns of instruments will be zero and hence dropped.

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
        costs_type : `str, optional`
            By default, :attr:`Simulation.costs_type`.

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
        if costs_type is None:
            costs_type = self.simulation.costs_type
        from ..economies.problem import Problem  # noqa
        return Problem(product_formulations, product_data, agent_formulation, agent_data, integration, costs_type)

    def _compute_default_instruments(self) -> Tuple[Array, Array]:
        """Compute default sums of characteristics excluded BLP instruments."""

        # collect exogenous variables names that will be used in instruments
        assert self.simulation.product_formulations[0] is not None
        X1_names = self.simulation.product_formulations[0]._names - {'prices'}
        X3_names: Set[str] = set()
        if self.simulation.product_formulations[2] is not None:
            X3_names = self.simulation.product_formulations[2]._names

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

    def compute_micro(self, micro_moments: Sequence[Moment]) -> Array:
        r"""Compute averaged micro moment values, :math:`\bar{g}_M`.

        Typically, this method is used to compute the values that micro moments aim to match. This can be done by
        setting ``value=0`` in each of the configured ``micro_moments``.

        Parameters
        ----------
        micro_moments : `sequence of FirstChoiceCovarianceMoment`
            Configurations for the averaged micro moments that will be computed. The only type of micro moment currently
            supported is the :class:`FirstChoiceCovarianceMoment`.

        Returns
        -------
        `ndarray`
            Averaged micro moments, :math:`\bar{g}_M`, in :eq:`averaged_micro_moments`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        errors: List[Error] = []

        # keep track of long it takes to compute micro moments
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

        # define a factory for computing market-level micro moments
        def market_factory(s: Hashable) -> Tuple[SimulationResultsMarket]:
            """Build a market along with arguments used to compute micro moments."""
            market_s = SimulationResultsMarket(
                self.simulation, s, self.simulation._parameters, self.simulation.sigma, self.simulation.pi,
                self.simulation.rho, self.simulation.beta, delta, moments, self._data_override
            )
            return market_s,

        # compute micro moments (averaged across markets) market-by-market
        micro_mapping: Dict[Hashable, Array] = {}
        generator = generate_items(
            self.simulation.unique_market_ids, market_factory, SimulationResultsMarket.safely_compute_micro
        )
        for t, (micro_t, errors_t) in generator:
            micro_mapping[t] = micro_t
            errors.extend(errors_t)

        # average micro moments across all markets (this is done after market-by-market computation to preserve
        #   numerical stability with different market orderings)
        micro = np.zeros((moments.MM, 1), options.dtype)
        for t in self.simulation.unique_market_ids:
            indices = moments.market_indices[t]
            micro[indices] += micro_mapping[t] / moments.market_counts[indices]

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # output how long it took to compute the micro moments
        end_time = time.time()
        output("")
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return micro
