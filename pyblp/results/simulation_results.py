"""Economy-level structuring of BLP simulation results."""

import time
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

from .. import exceptions, options
from ..configurations.formulation import Formulation
from ..configurations.integration import Integration
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

    The :meth:`SimulationResults.to_problem` method can be used to convert the full set of simulated data and configured
    information into a :class:`Problem`.

    Attributes
    ----------
    simulation : `Simulation`
        :class:`Simulation` that created these results.
    product_data : `recarray`
        Simulated :attr:`Simulation.product_data` that are updated with synthetic prices and shares.
    computation_time : `float`
        Number of seconds it took to compute synthetic prices and shares.
    fp_converged : `ndarray`
        Flags for convergence of the iteration routine used to compute synthetic prices in each market. Flags are in
        the same order as :attr:`Simulation.unique_market_ids`.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute synthetic prices in each market.
        Counts are in the same order as :attr:`Simulation.unique_market_ids`.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute synthetic prices was evaluated in each market. Counts are in the
        same order as :attr:`Simulation.unique_market_ids`.

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
                'product_data', 'delta', 'computation_time', 'fp_converged', 'fp_iterations', 'contraction_evaluations'
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
            agent_data: Optional[Mapping] = None, integration: Optional[Integration] = None) -> 'Problem':
        """Convert the solved simulation into a problem.

        Parameters are the same as those of :class:`Problem`. By default, the structure of the problem will be the same
        as that of the solved simulation.

        Parameters
        ----------
        product_formulations : `Formulation or sequence of Formulation, optional`
            By default, :attr:`Simulation.product_formulations`.
        product_data : `structured array-like, optional`
            By default, :attr:`SimulationResults.product_data`.
        agent_formulation : `Formulation, optional`
            By default, :attr:`Simulation.agent_formulation`.
        agent_data : `structured array-like, optional`
            By default, :attr:`Simulation.agent_data`.
        integration : `Integration, optional`
            By default, this is unspecified.

        Returns
        -------
        `Problem`
            A BLP problem.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        from ..economies.problem import Problem  # noqa
        if product_formulations is None:
            product_formulations = self.simulation.product_formulations
        if product_data is None:
            product_data = self.product_data
        if agent_formulation is None:
            agent_formulation = self.simulation.agent_formulation
        if agent_data is None:
            agent_data = self.simulation.agent_data
        assert product_formulations is not None and product_data is not None
        return Problem(product_formulations, product_data, agent_formulation, agent_data, integration)

    def compute_micro(self, micro_moments: Sequence[Moment]) -> Array:
        r"""Compute averaged micro moment values, :math:`\bar{g}_M`.

        Typically, this method is used to compute the values that micro moments aim to match. This can be done by
        setting ``value=0`` in each of the configured ``micro_moments``.

        Parameters
        ----------
        micro_moments : `sequence of ProductsAgentsCovarianceMoment`
            Configurations for the averaged micro moments that will be computed. The only type of micro moment currently
            supported is the :class:`ProductsAgentsCovarianceMoment`.

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
