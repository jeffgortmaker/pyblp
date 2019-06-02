"""Economy-level structuring of bootstrapped BLP problem results."""

import itertools
import time
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .problem_results import ProblemResults
from .results import Results
from .. import exceptions, options
from ..markets.results_market import ResultsMarket
from ..utilities.basics import (
    Array, Error, SolverStats, format_seconds, format_table, generate_items, output, output_progress
)


class BootstrappedResults(Results):
    r"""Bootstrapped results of a solved problem.

    This class has same methods as :class:`ProblemResults` that compute post-estimation outputs in one or more markets,
    but not other methods like :meth:`ProblemResults.compute_optimal_instruments` that do not make sense in a
    bootstrapped dataset. The only other difference is that methods return arrays with an extra first dimension along
    which bootstrapped results are stacked (these stacked results can be used to construct, for example, confidence
    intervals for post-estimation outputs). Similarly, arrays of data (except for firm IDs and ownership matrices)
    passed as arguments to methods should have an extra first dimension of size :attr:`BootstrappedResults.draws`.

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
        Bootstrapped marketshares, :math:`s`, implied by each draw.
    bootstrapped_delta : `ndarray`
        Bootstrapped mean utility, :math:`\delta`, implied by each draw.
    bootstrapped_costs : `ndarray`
        Bootstrapped marginal costs, :math:`c`, implied by each draw.
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
    bootstrapped_costs: Array
    computation_time: float
    draws: int
    fp_converged: Array
    fp_iterations: Array
    contraction_evaluations: Array

    def __init__(
            self, problem_results: ProblemResults, bootstrapped_sigma: Array, bootstrapped_pi: Array,
            bootstrapped_rho: Array, bootstrapped_beta: Array, bootstrapped_gamma: Array, bootstrapped_prices: Array,
            bootstrapped_shares: Array, bootstrapped_delta: Array, bootstrapped_costs: Array, start_time: float,
            end_time: float, draws: int, iteration_stats: Sequence[Mapping[Hashable, SolverStats]]) -> None:
        """Structure bootstrapped problem results."""
        super().__init__(problem_results.problem, problem_results._parameters, problem_results._moments)
        self.problem_results = problem_results
        self.bootstrapped_sigma = bootstrapped_sigma
        self.bootstrapped_pi = bootstrapped_pi
        self.bootstrapped_rho = bootstrapped_rho
        self.bootstrapped_beta = bootstrapped_beta
        self.bootstrapped_gamma = bootstrapped_gamma
        self.bootstrapped_prices = bootstrapped_prices
        self.bootstrapped_shares = bootstrapped_shares
        self.bootstrapped_delta = bootstrapped_delta
        self.bootstrapped_costs = bootstrapped_costs
        self.computation_time = end_time - start_time
        self.draws = draws
        unique_market_ids = problem_results.problem.unique_market_ids
        self.fp_converged = np.array(
            [[m[t].converged if m else True for m in iteration_stats] for t in unique_market_ids], dtype=np.bool
        )
        self.fp_iterations = np.array(
            [[m[t].iterations if m else 0 for m in iteration_stats] for t in unique_market_ids], dtype=np.int
        )
        self.contraction_evaluations = np.array(
            [[m[t].evaluations if m else 0 for m in iteration_stats] for t in unique_market_ids], dtype=np.int
        )

    def __str__(self) -> str:
        """Format bootstrapped problem results as a string."""
        header = [("Computation", "Time"), ("Bootstrap", "Draws")]
        values = [format_seconds(self.computation_time), self.draws]
        if self.fp_iterations.sum() > 0 or self.contraction_evaluations.sum() > 0:
            header.extend([("Fixed Point", "Iterations"), ("Contraction", "Evaluations")])
            values.extend([self.fp_iterations.sum(), self.contraction_evaluations.sum()])
        return format_table(header, values, title="Bootstrapped Results Summary")

    def _coerce_matrices(self, matrices: Any) -> Array:
        """Coerce array-like stacked matrix tensors into a stacked matrix tensor and validate it."""
        matrices = np.atleast_3d(np.asarray(matrices, options.dtype))
        if matrices.shape != (self.draws, self.problem.N, self.problem._max_J):
            raise ValueError(f"matrices must be {self.draws} by {self.problem.N} by {self.problem._max_J}.")
        return matrices

    def _coerce_optional_costs(self, costs: Optional[Any]) -> Array:
        """Coerce optional array-like costs into a column vector tensor and validate it."""
        if costs is not None:
            costs = np.atleast_3d(np.asarray(costs, options.dtype))
            if costs.shape != (self.draws, self.problem.N, 1):
                raise ValueError(f"costs must be None or {self.draws} by {self.problem.N}.")
        return costs

    def _coerce_optional_prices(self, prices: Optional[Any]) -> Array:
        """Coerce optional array-like prices into a column vector tensor and validate it."""
        if prices is not None:
            prices = np.atleast_3d(np.asarray(prices, options.dtype))
            if prices.shape != (self.draws, self.problem.N, 1):
                raise ValueError(f"prices must be None or {self.draws} by {self.problem.N}.")
        return prices

    def _coerce_optional_shares(self, shares: Optional[Any]) -> Array:
        """Coerce optional array-like shares into a column vector tensor and validate it."""
        if shares is not None:
            shares = np.atleast_3d(np.asarray(shares, options.dtype))
            if shares.shape != (self.draws, self.problem.N, 1):
                raise ValueError(f"shares must be None or {self.draws} by {self.problem.N}.")
        return shares

    def _combine_arrays(
            self, compute_market_results: Callable, fixed_args: Sequence = (), market_args: Sequence = ()) -> Array:
        """Compute an array for each market and stack them into a single tensor. An array for a single market is
        computed by passing fixed_args (identical for all markets) and market_args (matrices with as many rows as there
        are products that are restricted to the market) to compute_market_results, a ResultsMarket method that returns
        the output for the market and a set of any errors encountered during computation.
        """
        errors: List[Error] = []

        # keep track of how long it takes to compute the arrays
        start_time = time.time()

        # define a factory for computing bootstrapped arrays in markets
        def market_factory(pair: Tuple[int, Hashable]) -> tuple:
            """Build a market with bootstrapped data along with arguments used to compute arrays."""
            c, s = pair
            data_override_cs = {
                'prices': self.bootstrapped_prices[c, self.problem._product_market_indices[s]],
                'shares': self.bootstrapped_shares[c, self.problem._product_market_indices[s]]
            }
            market_js = ResultsMarket(
                self.problem, s, self._parameters, self.bootstrapped_sigma[c], self.bootstrapped_pi[c],
                self.bootstrapped_rho[c], self.bootstrapped_beta[c], self.bootstrapped_delta[c], self._moments,
                data_override_cs
            )
            args_cs: List[Optional[Array]] = []
            for market_arg in market_args:
                if market_arg is None:
                    args_cs.append(market_arg)
                elif len(market_arg.shape) == 2:
                    args_cs.append(market_arg[self.problem._product_market_indices[s]])
                else:
                    assert len(market_arg.shape) == 3
                    args_cs.append(market_arg[c, self.problem._product_market_indices[s]])
            return (market_js, *fixed_args, *args_cs)

        # construct a mapping from draws and market IDs to market-specific arrays and compute the full matrix size
        matrix_mapping: Dict[Tuple[int, Hashable], Array] = {}
        pairs = itertools.product(range(self.draws), self.problem.unique_market_ids)
        generator = output_progress(
            generate_items(pairs, market_factory, compute_market_results), self.draws * self.problem.T, start_time
        )
        for (d, t), (array_dt, errors_dt) in generator:
            matrix_mapping[(d, t)] = np.c_[array_dt]
            errors.extend(errors_dt)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # determine the number of rows and columns
        row_count = sum(matrix_mapping[(0, t)].shape[0] for t in self.problem.unique_market_ids)
        column_count = max(matrix_mapping[(0, t)].shape[1] for t in self.problem.unique_market_ids)

        # preserve the original product order or the sorted market order when stacking the arrays
        combined = np.full((self.draws, row_count, column_count), np.nan, options.dtype)
        for (d, t), matrix_dt in matrix_mapping.items():
            if row_count == self.problem.N:
                combined[d, self.problem._product_market_indices[t], :matrix_dt.shape[1]] = matrix_dt
            else:
                combined[d, self.problem.unique_market_ids == t, :matrix_dt.shape[1]] = matrix_dt

        # output how long it took to compute the arrays
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return combined

    def to_dict(
            self, attributes: Sequence[str] = (
                'bootstrapped_sigma', 'bootstrapped_pi', 'bootstrapped_rho', 'bootstrapped_beta', 'bootstrapped_gamma',
                'bootstrapped_prices', 'bootstrapped_shares', 'bootstrapped_delta', 'bootstrapped_costs',
                'computation_time', 'draws', 'fp_converged', 'fp_iterations', 'contraction_evaluations'
            )) -> dict:
        """Convert these results into a dictionary that maps attribute names to values.

        Once converted to a dictionary, these results can be saved to a file with :func:`pickle.dump`.

        Parameters
        ----------
        attributes : `tuple of str, optional`
            Name of attributes that will be added to the dictionary. By default, all :class:`BootstrappedResults`
            attributes are added except for :attr:`BootstrappedResults.problem_results`.

        Returns
        -------
        `dict`
            Mapping from attribute names to values.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        return {k: getattr(self, k) for k in attributes}
