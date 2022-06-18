"""Economy-level structuring of bootstrapped BLP problem results."""

import itertools
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .economy_results import SimpleEconomyResults
from .problem_results import ProblemResults
from .. import exceptions, options
from ..configurations.integration import Integration
from ..markets.economy_results_market import EconomyResultsMarket
from ..primitives import Agents
from ..utilities.basics import (
    Array, Error, SolverStats, format_seconds, format_table, generate_items, get_indices, output, output_progress
)


class BootstrappedResults(SimpleEconomyResults):
    r"""Bootstrapped results of a solved problem.

    This class has slightly modified versions of the following :class:`ProblemResults` methods:

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

    The difference is that each method returns an array with an extra first dimension along which bootstrapped results
    are stacked These stacked results can be used to construct, for example, confidence intervals for
    post-estimation outputs. Similarly, arrays of data (except for firm IDs and ownership matrices) passed as arguments
    to methods should have an extra first dimension of size :attr:`BootstrappedResults.draws`.

    Attributes
    ----------
    problem_results : `ProblemResults`
        :class:`ProblemResults` that was used to compute these bootstrapped results.
    bootstrapped_sigma : `ndarray`
        Bootstrapped Cholesky decomposition of the covariance matrix for unobserved taste heterogeneity, :math:`\Sigma`.
    bootstrapped_pi : `ndarray`
        Bootstrapped parameters that measures how agent tastes vary with demographics, :math:`\Pi`.
    bootstrapped_rho : `ndarray`
        Bootstrapped parameters that measure within nesting group correlations, :math:`\rho`.
    bootstrapped_beta : `ndarray`
        Bootstrapped demand-side linear parameters, :math:`\beta`.
    bootstrapped_gamma : `ndarray`
        Bootstrapped supply-side linear parameters, :math:`\gamma`.
    bootstrapped_prices : `ndarray`
        Bootstrapped prices, :math:`p`. If a supply side was not estimated, these are unchanged prices. Otherwise, they
        are equilibrium prices implied by each draw.
    bootstrapped_shares : `ndarray`
        Bootstrapped market shares, :math:`s`, implied by each draw.
    bootstrapped_delta : `ndarray`
        Bootstrapped mean utility, :math:`\delta`, implied by each draw.
    computation_time : `float`
        Number of seconds it took to compute the bootstrapped results.
    draws : `int`
        Number of bootstrap draws.
    fp_converged : `ndarray`
        Flags for convergence of the iteration routine used to compute equilibrium prices in each market. Rows are in
        the same order as :attr:`Problem.unique_market_ids` and column indices correspond to draws.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute equilibrium prices in each market
        for each draw. Rows are in the same order as :attr:`Problem.unique_market_ids` and column indices correspond to
        draws.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute equilibrium prices was evaluated in each market for each draw.
        Rows are in the same order as :attr:`Problem.unique_market_ids` and column indices correspond to draws.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    problem_results: ProblemResults
    bootstrapped_sigma: Array
    bootstrapped_pi: Array
    bootstrapped_rho: Array
    bootstrapped_beta: Array
    bootstrapped_gamma: Array
    bootstrapped_prices: Array
    bootstrapped_shares: Array
    bootstrapped_delta: Array
    computation_time: float
    draws: int
    fp_converged: Array
    fp_iterations: Array
    contraction_evaluations: Array

    def __init__(
            self, problem_results: ProblemResults, bootstrapped_sigma: Array, bootstrapped_pi: Array,
            bootstrapped_rho: Array, bootstrapped_beta: Array, bootstrapped_gamma: Array, bootstrapped_prices: Array,
            bootstrapped_shares: Array, bootstrapped_delta: Array, start_time: float, end_time: float, draws: int,
            iteration_stats: Mapping[Hashable, SolverStats]) -> None:
        """Structure bootstrapped problem results."""
        super().__init__(problem_results.problem, problem_results._parameters)
        self.problem_results = problem_results
        self.bootstrapped_sigma = bootstrapped_sigma
        self.bootstrapped_pi = bootstrapped_pi
        self.bootstrapped_rho = bootstrapped_rho
        self.bootstrapped_beta = bootstrapped_beta
        self.bootstrapped_gamma = bootstrapped_gamma
        self.bootstrapped_prices = bootstrapped_prices
        self.bootstrapped_shares = bootstrapped_shares
        self.bootstrapped_delta = bootstrapped_delta
        self.computation_time = end_time - start_time
        self.draws = draws
        unique_market_ids = problem_results.problem.unique_market_ids
        self.fp_converged = np.array(
            [[iteration_stats[(d, t)].converged for d in range(self.draws)] for t in unique_market_ids],
            dtype=np.bool_,
        )
        self.fp_iterations = np.array(
            [[iteration_stats[(d, t)].iterations for d in range(self.draws)] for t in unique_market_ids],
            dtype=np.int64,
        )
        self.contraction_evaluations = np.array(
            [[iteration_stats[(d, t)].evaluations for d in range(self.draws)] for t in unique_market_ids],
            dtype=np.int64,
        )

    def __str__(self) -> str:
        """Format bootstrapped problem results as a string."""
        header = [("Computation", "Time"), ("Bootstrap", "Draws")]
        values = [format_seconds(self.computation_time), self.draws]
        if self.fp_iterations.sum() > 0 or self.contraction_evaluations.sum() > 0:
            header.extend([("Fixed Point", "Iterations"), ("Contraction", "Evaluations")])
            values.extend([self.fp_iterations.sum(), self.contraction_evaluations.sum()])
        return format_table(header, values, title="Bootstrapped Results Summary")

    def _combine_arrays(
            self, compute_market_results: Callable, market_ids: Array, fixed_args: Sequence = (),
            market_args: Sequence = (), agent_data: Optional[Mapping] = None,
            integration: Optional[Integration] = None) -> Array:
        """Compute arrays for one or all markets and stack them into a single tensor. An array for a single market is
        computed by passing fixed_args (identical for all markets) and market_args (matrices with as many rows as there
        are products that are restricted to the market) to compute_market_results, a ResultsMarket method that returns
        the output for the market and any errors encountered during computation. Agent data and an integration
        configuration can be optionally specified to override agent data.
        """
        errors: List[Error] = []

        # keep track of how long it takes to compute the arrays
        start_time = time.time()

        # structure or construct different agent data
        if agent_data is None and integration is None:
            agents = self._economy.agents
            agents_market_indices = self._economy._agent_market_indices
        else:
            agents = Agents(self._economy.products, self._economy.agent_formulation, agent_data, integration)
            agents_market_indices = get_indices(agents.market_ids)

        def market_factory(pair: Tuple[int, Hashable]) -> tuple:
            """Build a market with bootstrapped data along with arguments used to compute arrays."""
            c, s = pair
            data_override_c = {
                'prices': self.bootstrapped_prices[c],
                'shares': self.bootstrapped_shares[c]
            }
            market_cs = EconomyResultsMarket(
                self._economy, s, self._parameters, self.bootstrapped_sigma[c], self.bootstrapped_pi[c],
                self.bootstrapped_rho[c], self.bootstrapped_beta[c], self.bootstrapped_gamma[c],
                self.bootstrapped_delta[c], data_override=data_override_c,
                agents_override=agents[agents_market_indices[s]]
            )
            args_cs: List[Optional[Array]] = []
            for market_arg in market_args:
                if market_arg is None:
                    args_cs.append(market_arg)
                elif len(market_arg.shape) == 2:
                    if market_ids.size == 1:
                        args_cs.append(market_arg)
                    else:
                        args_cs.append(market_arg[self._economy._product_market_indices[s]])
                else:
                    assert len(market_arg.shape) == 3
                    if market_ids.size == 1:
                        args_cs.append(market_arg[c])
                    else:
                        args_cs.append(market_arg[c, self._economy._product_market_indices[s]])
            return (market_cs, *fixed_args, *args_cs)

        # construct a mapping from draws and market IDs to market-specific arrays and compute the full matrix size
        array_mapping: Dict[Tuple[int, Hashable], Array] = {}
        pairs = itertools.product(range(self.draws), market_ids)
        generator = generate_items(pairs, market_factory, compute_market_results)
        if self.draws > 1 or market_ids.size > 1:
            generator = output_progress(generator, self.draws * market_ids.size, start_time)
        for (d, t), (array_dt, errors_dt) in generator:
            array_mapping[(d, t)] = np.c_[array_dt]
            errors.extend(errors_dt)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # determine the sizes of dimensions
        dimension_sizes = []
        for dimension in range(len(array_mapping[(0, market_ids[0])].shape)):
            if dimension == 0:
                dimension_sizes.append(sum(array_mapping[(0, t)].shape[dimension] for t in market_ids))
            else:
                dimension_sizes.append(max(array_mapping[(0, t)].shape[dimension] for t in market_ids))

        # preserve the original product order or the sorted market order when stacking the arrays
        combined = np.full((self.draws, *dimension_sizes), np.nan, options.dtype)
        for (d, t), array_dt in array_mapping.items():
            slices = (slice(0, s) for s in array_dt.shape[1:])
            if dimension_sizes[0] == market_ids.size:
                combined[(d, market_ids == t, *slices)] = array_dt
            elif dimension_sizes[0] == self._economy.N:
                combined[(d, self._economy._product_market_indices[t], *slices)] = array_dt
            else:
                assert market_ids.size == 1
                combined[d] = array_dt

        # output how long it took to compute the arrays
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return combined

    def _coerce_matrices(self, matrices: Any, market_ids: Array) -> Array:
        """Coerce array-like stacked matrix tensors into a stacked matrix tensor and validate it."""
        matrices = np.atleast_3d(np.asarray(matrices, options.dtype))
        rows = sum(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        columns = max(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        if matrices.shape != (self.draws, rows, columns):
            raise ValueError(f"matrices must be {self.draws} by {rows} by {columns}.")
        return matrices

    def _coerce_optional_delta(self, delta: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like mean utilities into a column vector tensor and validate it."""
        if delta is None:
            return None
        delta = np.atleast_3d(np.asarray(delta, options.dtype))
        rows = sum(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        if delta.shape != (self.draws, rows, 1):
            raise ValueError(f"delta must be None or {self.draws} by {rows}.")
        return delta

    def _coerce_optional_costs(self, costs: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like costs into a column vector tensor and validate it."""
        if costs is None:
            return None
        costs = np.atleast_3d(np.asarray(costs, options.dtype))
        rows = sum(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        if costs.shape != (self.draws, rows, 1):
            raise ValueError(f"costs must be None or {self.draws} by {rows}.")
        return costs

    def _coerce_optional_prices(self, prices: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like prices into a column vector tensor and validate it."""
        if prices is None:
            return None
        prices = np.atleast_3d(np.asarray(prices, options.dtype))
        rows = sum(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        if prices.shape != (self.draws, rows, 1):
            raise ValueError(f"prices must be None or {self.draws} by {rows}.")
        return prices

    def _coerce_optional_shares(self, shares: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like shares into a column vector tensor and validate it."""
        if shares is None:
            return shares
        shares = np.atleast_3d(np.asarray(shares, options.dtype))
        rows = sum(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        if shares.shape != (self.draws, rows, 1):
            raise ValueError(f"shares must be None or {self.draws} by {rows}.")
        return shares

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
                'bootstrapped_sigma', 'bootstrapped_pi', 'bootstrapped_rho', 'bootstrapped_beta', 'bootstrapped_gamma',
                'bootstrapped_prices', 'bootstrapped_shares', 'bootstrapped_delta', 'computation_time', 'draws',
                'fp_converged', 'fp_iterations', 'contraction_evaluations'
            )) -> dict:
        """Convert these results into a dictionary that maps attribute names to values.

        Parameters
        ----------
        attributes : `sequence of str, optional`
            Name of attributes that will be added to the dictionary. By default, all :class:`BootstrappedResults`
            attributes are added except for :attr:`BootstrappedResults.problem_results`.

        Returns
        -------
        `dict`
            Mapping from attribute names to values.

        """
        return {k: getattr(self, k) for k in attributes}
