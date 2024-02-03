"""Economy-level structuring of abstract BLP problem results."""

import abc
import collections
import time
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import scipy.sparse

from .. import exceptions, options
from ..configurations.integration import Integration
from ..configurations.iteration import Iteration
from ..markets.economy_results_market import EconomyResultsMarket
from ..micro import MicroDataset, MicroMoment, Moments
from ..parameters import Parameters
from ..primitives import Agents, MicroAgents, build_demographics
from ..utilities.basics import (
    Array, Error, RecArray, StringRepresentation, format_seconds, generate_items, get_indices, output, output_progress,
    structure_matrices
)


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.economy import Economy  # noqa


class SimpleEconomyResults(abc.ABC, StringRepresentation):
    """Abstract results for an economy underlying the BLP model, supporting simple methods that return arrays."""

    _economy: 'Economy'
    _parameters: Parameters

    def __init__(self, economy: 'Economy', parameters: Parameters) -> None:
        """Store the underlying economy and parameter information."""
        self._economy = economy
        self._parameters = parameters

    @abc.abstractmethod
    def _combine_arrays(
            self, compute_market_results: Callable, market_ids: Array, fixed_args: Sequence = (),
            market_args: Sequence = (), agent_data: Optional[Mapping] = None,
            integration: Optional[Integration] = None) -> Array:
        """Combine arrays from one or all markets, which are computed by passing fixed_args (identical for all markets)
        and market_args (arrays that need to be restricted to markets) to compute_market_results, a market method
        that returns the output for the market and any errors encountered during computation. Agent data and an
        integration configuration can be optionally specified to override agent data.
        """

    def _select_market_ids(self, market_id: Optional[Any] = None) -> Array:
        """Select either a single market ID or all unique IDs."""
        if market_id is None:
            return self._economy.unique_market_ids
        if market_id in self._economy.unique_market_ids:
            return np.atleast_1d(np.array(market_id, np.object_))
        raise ValueError(f"market_id must be None or one of {list(sorted(self._economy.unique_market_ids))}.")

    def _coerce_matrices(self, matrices: Any, market_ids: Array) -> Array:
        """Coerce array-like stacked matrices into a stacked matrix and validate it."""
        matrices = np.c_[np.asarray(matrices, options.dtype)]
        rows = sum(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        columns = max(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        if matrices.shape != (rows, columns):
            raise ValueError(f"matrices must be {rows} by {columns}.")
        return matrices

    def _coerce_optional_delta(self, delta: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like mean utilities into a column vector and validate it."""
        if delta is None:
            return None
        delta = np.c_[np.asarray(delta, options.dtype)]
        rows = sum(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        if delta.shape != (rows, 1):
            raise ValueError(f"delta must be None or a {rows}-vector.")
        return delta

    def _coerce_optional_costs(self, costs: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like costs into a column vector and validate it."""
        if costs is None:
            return None
        costs = np.c_[np.asarray(costs, options.dtype)]
        rows = sum(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        if costs.shape != (rows, 1):
            raise ValueError(f"costs must be None or a {rows}-vector.")
        return costs

    def _coerce_optional_prices(self, prices: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like prices into a column vector and validate it."""
        if prices is None:
            return None
        prices = np.c_[np.asarray(prices, options.dtype)]
        rows = sum(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        if prices.shape != (rows, 1):
            raise ValueError(f"prices must be None or a {rows}-vector.")
        return prices

    def _coerce_optional_shares(self, shares: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like shares into a column vector and validate it."""
        if shares is None:
            return None
        shares = np.c_[np.asarray(shares, options.dtype)]
        rows = sum(i.size for t, i in self._economy._product_market_indices.items() if t in market_ids)
        if shares.shape != (rows, 1):
            raise ValueError(f"shares must be None or a {rows}-vector.")
        return shares

    def compute_aggregate_elasticities(
            self, factor: float = 0.1, name: Optional[str] = 'prices', market_id: Optional[Any] = None) -> Array:
        r"""Estimate aggregate elasticities of demand, :math:`\mathscr{E}`, with respect to a variable, :math:`x`.

        In market :math:`t`, the aggregate elasticity of demand is

        .. math:: \mathscr{E} = \sum_{j \in J_t} \frac{s_{jt}(x + \Delta x) - s_{jt}}{\Delta},

        in which :math:`\Delta` is a scalar factor and :math:`s_{jt}(x + \Delta x)` is the share of product :math:`j` in
        market :math:`t`, evaluated at the scaled values of the variable.

        Parameters
        ----------
        factor : `float, optional`
            The scalar factor, :math:`\Delta`.
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices. If this is ``None``, the variable will
            be :math:`x = \delta`, the mean utility.
        market_id : `object, optional`
            ID of the market in which to compute aggregate elasticities. By default, aggregate elasticities are computed
            in all markets and stacked.

        Returns
        -------
        `ndarray`
            Estimates of aggregate elasticities of demand, :math:`\mathscr{E}`. If ``market_id`` was not specified, rows
            are in the same order as :attr:`Problem.unique_market_ids`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output(f"Computing aggregate elasticities with respect to {name or 'the mean utility'} ...")
        if not isinstance(factor, float):
            raise ValueError("factor must be a float.")
        self._economy._validate_name(name)
        market_ids = self._select_market_ids(market_id)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_aggregate_elasticity, market_ids, fixed_args=[factor, name]
        )

    def compute_elasticities(self, name: Optional[str] = 'prices', market_id: Optional[Any] = None) -> Array:
        r"""Estimate matrices of elasticities of demand, :math:`\varepsilon`, with respect to a variable, :math:`x`.

        In market :math:`t`, the value in row :math:`j` and column :math:`k` of :math:`\varepsilon` is

        .. math:: \varepsilon_{jk} = \frac{x_{kt}}{s_{jt}}\frac{\partial s_{jt}}{\partial x_{kt}}.

        Parameters
        ----------
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices. If this is ``None``, the variable will
            be :math:`x = \delta`, the mean utility.
        market_id : `object, optional`
            ID of the market in which to compute elasticities. By default, elasticities are computed in all markets and
            stacked.

        Returns
        -------
        `ndarray`
            Estimated :math:`J_t \times J_t` matrices of elasticities of demand, :math:`\varepsilon`. If ``market_id``
            was not specified, matrices are estimated in each market :math:`t` and stacked. Columns for a market are in
            the same order as products for the market. If a market has fewer products than others, extra columns will
            contain ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output(f"Computing elasticities with respect to {name or 'the mean utility'} ...")
        self._economy._validate_name(name)
        market_ids = self._select_market_ids(market_id)
        return self._combine_arrays(EconomyResultsMarket.safely_compute_elasticities, market_ids, fixed_args=[name])

    def compute_demand_jacobians(self, name: str = 'prices', market_id: Optional[Any] = None) -> Array:
        r"""Estimate matrices of derivatives of demand with respect to a variable, :math:`x`.

        In market :math:`t`, the value in row :math:`j` and column :math:`k` is

        .. math:: \frac{\partial s_{jt}}{\partial x_{kt}}.

        Parameters
        ----------
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices.
        market_id : `object, optional`
            ID of the market in which to compute Jacobians. By default, Jacobians are computed in all markets and
            stacked.

        Returns
        -------
        `ndarray`
            Estimated :math:`J_t \times J_t` matrices of derivatives of demand. If ``market_id`` was not specified,
            matrices are estimated in each market :math:`t` and stacked. Columns for a market are in the same order as
            products for the market. If a market has fewer products than others, extra columns will contain
            ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output(f"Computing derivatives of demand with respect to {name or 'the mean utility'} ...")
        self._economy._validate_name(name, none_valid=False)
        market_ids = self._select_market_ids(market_id)
        return self._combine_arrays(EconomyResultsMarket.safely_compute_demand_jacobian, market_ids, fixed_args=[name])

    def compute_demand_hessians(self, name: str = 'prices', market_id: Optional[Any] = None) -> Array:
        r"""Estimate arrays of second derivatives of demand with respect to a variable, :math:`x`.

        In market :math:`t`, the value indexed by :math:`(j, k, \ell)` is

        .. math:: \frac{\partial^2 s_{jt}}{\partial x_{kt} \partial x_{\ell t}}.

        Parameters
        ----------
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices.
        market_id : `object, optional`
            ID of the market in which to compute Hessians. By default, Hessians are computed in all markets and
            stacked.

        Returns
        -------
        `ndarray`
            Estimated :math:`J_t \times J_t \times J_t` arrays of second derivatives of demand. If ``market_id`` was not
            specified, arrays are estimated in each market :math:`t` and stacked. Indices for a market are in the same
            order as products for the market. If a market has fewer products than others, extra indices will contain
            ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output(f"Computing second derivatives of demand with respect to {name or 'the mean utility'} ...")
        self._economy._validate_name(name, none_valid=False)
        market_ids = self._select_market_ids(market_id)
        return self._combine_arrays(EconomyResultsMarket.safely_compute_demand_hessian, market_ids, fixed_args=[name])

    def compute_diversion_ratios(self, name: Optional[str] = 'prices', market_id: Optional[Any] = None) -> Array:
        r"""Estimate matrices of diversion ratios, :math:`\mathscr{D}`, with respect to a variable, :math:`x`.

        In market :math:`t`, the value in row :math:`j` and column :math:`k \neq j` is

        .. math::

           \mathscr{D}_{jk} =
           -\frac{\partial s_{kt}}{\partial x_{jt}} \Big/ \frac{\partial s_{jt}}{\partial x_{jt}}.

        Diversion ratios for the outside good are reported on diagonals:

        .. math::

           \mathscr{D}_{jj} =
           -\frac{\partial s_{0t}}{\partial x_{jt}} \Big/ \frac{\partial s_{jt}}{\partial x_{jt}}.

        Unlike :meth:`ProblemResults.compute_long_run_diversion_ratios`, this gives the marginal treatment effect (MTE)
        version of the diversion ratio. For more information, see :ref:`references:Conlon and Mortimer (2018)`.

        Parameters
        ----------
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices. If this is ``None``, the variable will
            be :math:`x = \delta`, the mean utility.
        market_id : `object, optional`
            ID of the market in which to compute diversion ratios. By default, diversion ratios are computed in all
            markets and stacked.

        Returns
        -------
        `ndarray`
            Estimated :math:`J_t \times J_t` matrices of diversion ratios, :math:`\mathscr{D}`. If ``market_id`` was not
            specified, matrices are estimated in each market :math:`t` and stacked. Columns for a market are in the same
            order as products for the market. If a market has fewer products than others, extra columns will contain
            ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output(f"Computing diversion ratios with respect to {name or 'the mean utility'} ...")
        self._economy._validate_name(name)
        market_ids = self._select_market_ids(market_id)
        return self._combine_arrays(EconomyResultsMarket.safely_compute_diversion_ratios, market_ids, fixed_args=[name])

    def compute_long_run_diversion_ratios(self, market_id: Optional[Any] = None) -> Array:
        r"""Estimate matrices of long-run diversion ratios, :math:`\bar{\mathscr{D}}`.

        In market :math:`t`, the value in row :math:`j` and column :math:`k \neq j` is

        .. math:: \bar{\mathscr{D}}_{jk} = \frac{s_{k(-j)t} - s_{kt}}{s_{jt}},

        in which :math:`s_{k(-j)t}` is the share of product :math:`k` computed with :math:`j` removed from the choice
        set. Long-run diversion ratios for the outside good are reported on diagonals:

        .. math:: \bar{\mathscr{D}}_{jj} = \frac{s_{0(-j)t} - s_0}{s_{jt}}.

        Unlike :meth:`ProblemResults.compute_diversion_ratios`, this gives the average treatment effect (ATE) version of
        the diversion ratio. For more information, see :ref:`references:Conlon and Mortimer (2018)`.

        Parameters
        ----------
        market_id : `object, optional`
            ID of the market in which to compute long-run diversion ratios. By default, long-run diversion ratios are
            computed in all markets and stacked.

        Returns
        -------
        `ndarray`
            Estimated :math:`J_t \times J_t` matrices of long-run diversion ratios, :math:`\bar{\mathscr{D}}`. If
            ``market_id`` was not specified, matrices are estimated in each market :math:`t` and stacked. Columns for a
            market are in the same order as products for the market. If a market has fewer products than others, extra
            columns will contain ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing long run mean diversion ratios ...")
        market_ids = self._select_market_ids(market_id)
        return self._combine_arrays(EconomyResultsMarket.safely_compute_long_run_diversion_ratios, market_ids)

    def compute_probabilities(
            self, prices: Optional[Any] = None, delta: Optional[Any] = None, agent_data: Optional[Mapping] = None,
            integration: Optional[Integration] = None, market_id: Optional[Any] = None) -> Array:
        r"""Estimate matrices of choice probabilities.

        For each market, the value in row :math:`j` and column `i` is given by :eq:`probabilities` when there are random
        coefficients, and by :eq:`nested_probabilities` when there is additionally a nested structure. For the logit
        and nested logit models, choice probabilities are market shares.

        It may be desirable to compute the probabilities associated with equilibrium prices that have been computed, for
        example, by :meth:`ProblemResults.compute_prices`.

        .. note::

           To compute equilibrium shares (and prices) associated with a more complicated counterfactual, a
           :class:`Simulation` for the counterfactual can be initialized with the estimated parameters, structural
           errors, and marginal costs from these results, and then solved with :meth:`Simulation.replace_endogenous`.

        Alternatively, this method can also be used to evaluate the performance of different numerical integration
        configurations. One way to do so is to use :meth:`ProblemResults.compute_delta` to compute mean utilities
        with a very precise integration rule (one that is infeasible to use during estimation), use these same mean
        utilities and integration rule to precisely compute probabilities, and then compare error between these
        precisely-computed probabilities and probabilities computed with less precise (but feasible to use during
        estimation) integration rules, still using the precisely-computed mean utilities.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices at which to evaluate probabilities, such as equilibrium prices, :math:`p^*`, computed by
            :meth:`ProblemResults.compute_prices`. By default, unchanged prices are used.
        delta : `array-like, optional`
            Mean utilities that will be used to evaluate probabilities, such as those computed more precisely by
            :meth:`ProblemResults.compute_delta`. By default, the estimated :attr:`ProblemResults.delta` is used,
            and updated with any specified ``prices``.
        agent_data : `structured array-like, optional`
            Agent data that will be used to compute probabilities. By default, ``agent_data`` in :class:`Problem` is
            used. For more information, refer to :class:`Problem`.
        integration : `Integration, optional`
            :class:`Integration` configuration that will be used to compute probabilities, which will replace any
            ``nodes`` field in ``agent_data``. This configuration is required if ``agent_data`` is specified without a
            nodes field. By default, ``agent_data`` in :class:`Problem` is used. For more information, refer to
            :class:`Problem`.
        market_id : `object, optional`
            ID of the market in which to compute choice probabilities. By default, choice probabilities are computed in
            all markets and stacked.

        Returns
        -------
        `ndarray`
            Estimated :math:`J_t \times I_t` matrices of choice probabilities. If ``market_id`` was not specified,
            matrices are estimated in each market :math:`t` and stacked. Columns for a market are in the same order as
            agents for the market. If a market has fewer agents than others, extra columns will contain ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing choice probabilities ...")
        market_ids = self._select_market_ids(market_id)
        prices = self._coerce_optional_prices(prices, market_ids)
        delta = self._coerce_optional_delta(delta, market_ids)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_probabilities, market_ids, market_args=[prices, delta],
            agent_data=agent_data, integration=integration
        )

    def extract_diagonals(self, matrices: Any, market_id: Optional[Any] = None) -> Array:
        r"""Extract diagonals from stacked :math:`J_t \times J_t` matrices for each market :math:`t`.

        Parameters
        ----------
        matrices : `array-like`
            Stacked matrices, such as estimates of :math:`\varepsilon`, computed by
            :meth:`ProblemResults.compute_elasticities`; :math:`\mathscr{D}`, computed by
            :meth:`ProblemResults.compute_diversion_ratios`; :math:`\bar{\mathscr{D}}`, computed by
            :meth:`ProblemResults.compute_long_run_diversion_ratios`; or :math:`s_{ijt}` computed by
            :meth:`ProblemResults.compute_probabilities`.
        market_id : `object, optional`
            ID of the market in which to extract diagonals. By default, diagonals are extracted in all markets and
            stacked.

        Returns
        -------
        `ndarray`
            Stacked matrix diagonals. If ``market_id`` was not specified, diagonals are extracted in each market
            :math:`t` and stacked. If the matrices are estimates of :math:`\varepsilon`, a diagonal is a market's own
            elasticities of demand; if they are estimates of :math:`\mathscr{D}` or :math:`\bar{\mathscr{D}}`, a
            diagonal is a market's diversion ratios to the outside good.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Extracting diagonals ...")
        market_ids = self._select_market_ids(market_id)
        matrices = self._coerce_matrices(matrices, market_ids)
        return self._combine_arrays(EconomyResultsMarket.safely_extract_diagonal, market_ids, market_args=[matrices])

    def extract_diagonal_means(self, matrices: Any, market_id: Optional[Any] = None) -> Array:
        r"""Extract means of diagonals from stacked :math:`J_t \times J_t` matrices for each market :math:`t`.

        Parameters
        ----------
        matrices : `array-like`
            Stacked matrices, such as estimates of :math:`\varepsilon`, computed by
            :meth:`ProblemResults.compute_elasticities`; :math:`\mathscr{D}`, computed by
            :meth:`ProblemResults.compute_diversion_ratios`; :math:`\bar{\mathscr{D}}`, computed by
            :meth:`ProblemResults.compute_long_run_diversion_ratios`; or :math:`s_{ijt}` computed by
            :meth:`ProblemResults.compute_probabilities`.
        market_id : `object, optional`
            ID of the market in which to extract diagonal means. By default, diagonal means are extracted in all markets
            and stacked.

        Returns
        -------
        `ndarray`
            Stacked diagonal means. If ``market_id`` was not specified, diagonal means are extracted in each market
            :math:`t` and stacked. If the matrices are estimates of :math:`\varepsilon`, the mean of a diagonal is a
            market's mean own elasticity of demand; if they are estimates of :math:`\mathscr{D}` or
            :math:`\bar{\mathscr{D}}`, the mean of a diagonal is a market's mean diversion ratio to the outside good.
            Rows are in the same order as :attr:`Problem.unique_market_ids`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Extracting diagonal means ...")
        market_ids = self._select_market_ids(market_id)
        matrices = self._coerce_matrices(matrices, market_ids)
        return self._combine_arrays(
            EconomyResultsMarket.safely_extract_diagonal_mean, market_ids, market_args=[matrices]
        )

    def compute_costs(
            self, firm_ids: Optional[Any] = None, ownership: Optional[Any] = None,
            market_id: Optional[Any] = None) -> Array:
        r"""Estimate marginal costs, :math:`c`.

        Marginal costs are computed with the :math:`\eta`-markup equation in :eq:`eta`:

        .. math:: c = p - \eta.

        Parameters
        ----------
        firm_ids : `array-like, optional`
            Firm IDs. By default, the ``firm_ids`` field of ``product_data`` in :class:`Problem` will be used.
        ownership : `array-like, optional`
            Ownership matrices. By default, standard ownership matrices based on ``firm_ids`` will be used unless the
            ``ownership`` field of ``product_data`` in :class:`Problem` was specified.
        market_id : `object, optional`
            ID of the market in which to compute marginal costs. By default, marginal costs are computed in all markets
            and stacked.

        Returns
        -------
        `ndarray`
            Marginal costs, :math:`c`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing marginal costs ...")
        market_ids = self._select_market_ids(market_id)
        firm_ids = self._economy._coerce_optional_firm_ids(firm_ids, market_ids)
        ownership = self._economy._coerce_optional_ownership(ownership, market_ids)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_costs, market_ids, market_args=[firm_ids, ownership]
        )

    def compute_passthrough(
            self, firm_ids: Optional[Any] = None, ownership: Optional[Any] = None,
            market_id: Optional[Any] = None) -> Array:
        r"""Estimate matrices of passthrough of marginal costs to equilibrium prices, :math:`\Upsilon`.

        In market :math:`t`, the value in row :math:`j` and column :math:`k` of :math:`\Upsilon` is

        .. math:: \Upsilon_{jk} = \frac{\partial p_j}{\partial c_k}.

        Parameters
        ----------
        firm_ids : `array-like, optional`
            Firm IDs. By default, the ``firm_ids`` field of ``product_data`` in :class:`Problem` will be used.
        ownership : `array-like, optional`
            Ownership matrices. By default, standard ownership matrices based on ``firm_ids`` will be used unless the
            ``ownership`` field of ``product_data`` in :class:`Problem` was specified.
        market_id : `object, optional`
            ID of the market in which to compute passthrough. By default, passthrough matrices are computed in all
            markets and stacked.

        Returns
        -------
        `ndarray`
            Estimated :math:`J_t \times J_t` passthrough matrices, :math:`\Upsilon`. If ``market_id``
            was not specified, matrices are estimated in each market :math:`t` and stacked. Columns for a market are in
            the same order as products for the market. If a market has fewer products than others, extra columns will
            contain ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing passthrough ...")
        market_ids = self._select_market_ids(market_id)
        firm_ids = self._economy._coerce_optional_firm_ids(firm_ids, market_ids)
        ownership = self._economy._coerce_optional_ownership(ownership, market_ids)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_passthrough, market_ids, market_args=[firm_ids, ownership]
        )

    def compute_delta(
            self, agent_data: Optional[Mapping] = None, integration: Optional[Integration] = None,
            iteration: Optional[Iteration] = None, fp_type: str = 'safe_linear',
            shares_bounds: Optional[Tuple[Any, Any]] = (1e-300, None), market_id: Optional[Any] = None) -> Array:
        r"""Estimate mean utilities, :math:`\delta`.

        This method can be used to compute mean utilities at the estimated parameters with a different integration
        configuration or with different fixed point iteration settings than those used during estimation. The estimated
        :attr:`ProblemResults.delta` will be used as starting values for the fixed point routine.

        A more precisely estimated mean utility can be used, for example, by
        :meth:`ProblemResults.importance_sampling`. It can also be used to :meth:`ProblemResults.compute_shares` to
        compare the performance of different integration routines.

        Parameters
        ----------
        agent_data : `structured array-like, optional`
            Agent data that will be used to compute :math:`\delta`. By default, ``agent_data`` in :class:`Problem` is
            used. For more information, refer to :class:`Problem`.
        integration : `Integration, optional`
            :class:`Integration` configuration that will be used to compute :math:`\delta`, which will replace any
            ``nodes`` field in ``agent_data``. This configuration is required if ``agent_data`` is specified without a
            nodes field. By default, ``agent_data`` in :class:`Problem` is used. For more information, refer to
            :class:`Problem`.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem used to compute :math:`\delta` in
            each market. By default, ``Iteration('squarem', {'atol': 1e-14})`` is used. For more information, refer to
            :meth:`Problem.solve`.
        fp_type : `str, optional`
            Configuration for the type of contraction mapping used to compute :math:`\delta` in each market. By default,
            ``'safe_linear'`` is used. For more information, refer to :meth:`Problem.solve`.
        shares_bounds : `tuple, optional`
            Configuration for :math:`s_{jt}(\delta, \theta)` bounds of the form ``(lb, ub)``, in which both ``lb`` and
            ``ub`` are floats or ``None``. By default, simulated shares are bounded from below by ``1e-300``. This is
            only relevant if ``fp_type`` is ``'safe_linear'`` or ``'linear'``. Bounding shares in the contraction does
            nothing with a nonlinear fixed point. For more information, refer to :meth:`Problem.solve`.
        market_id : `object, optional`
            ID of the market in which to compute mean utilities. By default, mean utilities is computed in all markets
            and stacked.

        Returns
        -------
        `ndarray`
           Mean utilities, :math:`\delta`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing delta ...")
        market_ids = self._select_market_ids(market_id)
        iteration = self._economy._coerce_optional_delta_iteration(iteration)
        self._economy._validate_fp_type(fp_type)
        shares_bounds = self._economy._coerce_optional_bounds(shares_bounds, 'shares_bounds')
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_delta, market_ids, fixed_args=[iteration, fp_type, shares_bounds],
            agent_data=agent_data, integration=integration
        )

    def compute_approximate_prices(
            self, firm_ids: Optional[Any] = None, ownership: Optional[Any] = None, costs: Optional[Any] = None,
            market_id: Optional[Any] = None) -> Array:
        r"""Approximate equilibrium prices after firm or cost changes, :math:`p^*`, under the assumption that shares and
        their price derivatives are unaffected by such changes.

        This approximation is in the spirit of :ref:`references:Hausman, Leonard, and Zona (1994)` and
        :ref:`references:Werden (1997)`. Prices in each market are computed according to the :math:`\eta`-markup
        equation in :eq:`eta`:

        .. math:: p^* = c^* + \eta^*,

        in which the markup term is approximated with

        .. math:: \eta^* \approx -\left(\mathscr{H}^* \odot \frac{\partial s}{\partial p}\right)^{-1}s

        where :math:`\mathscr{H}^*` is the ownership or product holding matrix associated with firm changes.

        Parameters
        ----------
        firm_ids : `array-like, optional`
            Potentially changed firm IDs. By default, the unchanged ``firm_ids`` field of ``product_data`` in
            :class:`Problem` will be used.
        ownership : `array-like, optional`
            Potentially changed ownership matrices. By default, standard ownership matrices based on ``firm_ids`` will
            be used unless the ``ownership`` field of ``product_data`` in :class:`Problem` was specified.
        costs : `array-like, optional`
            Potentially changed marginal costs, :math:`c^*`. By default, unchanged marginal costs are computed with
            :meth:`ProblemResults.compute_costs`. Costs under a changed ownership structure can be computed by
            specifying the ``firm_ids`` or ``ownership`` arguments of :meth:`ProblemResults.compute_costs`.
        market_id : `object, optional`
            ID of the market in which to compute approximate equilibrium prices. By default, approximate equilibrium
            prices are computed in all markets and stacked.

        Returns
        -------
        `ndarray`
            Approximation of equilibrium prices after any firm or cost changes, :math:`p^*`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Solving for approximate equilibrium prices ...")
        market_ids = self._select_market_ids(market_id)
        firm_ids = self._economy._coerce_optional_firm_ids(firm_ids, market_ids)
        ownership = self._economy._coerce_optional_ownership(ownership, market_ids)
        costs = self._coerce_optional_costs(costs, market_ids)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_approximate_equilibrium_prices, market_ids,
            market_args=[firm_ids, ownership, costs]
        )

    def compute_prices(
            self, firm_ids: Optional[Any] = None, ownership: Optional[Any] = None, costs: Optional[Any] = None,
            prices: Optional[Any] = None, iteration: Optional[Iteration] = None, constant_costs: bool = True,
            market_id: Optional[Any] = None) -> Array:
        r"""Estimate equilibrium prices after firm or cost changes, :math:`p^*`.

        .. note::

           To compute equilibrium prices (and shares) associated with a more complicated counterfactual, a
           :class:`Simulation` for the counterfactual can be initialized with the estimated parameters, structural
           errors, and marginal costs from these results, and then solved with :meth:`Simulation.replace_endogenous`.
           The returned :class:`SimulationResults` gives more information about the contraction than this method, such
           as the number of contraction evaluations. It also automatically reports first and second order conditions.

        Prices are computed in each market by iterating over the :math:`\zeta`-markup contraction in
        :eq:`zeta_contraction`:

        .. math:: p^* \leftarrow c^* + \zeta^*(p^*),

        in which the markup term from :eq:`zeta` is

        .. math::

           \zeta^*(p^*) = \Lambda^{-1}(p^*)[\mathscr{H}^* \odot \Gamma(p^*)]'(p^* - c^*) - \Lambda^{-1}(p^*)s(p^*)

        where :math:`\mathscr{H}^*` is the ownership matrix associated with firm changes.

        Parameters
        ----------
        firm_ids : `array-like, optional`
            Potentially changed firm IDs. By default, the unchanged ``firm_ids`` field of ``product_data`` in
            :class:`Problem` will be used.
        ownership : `array-like, optional`
            Potentially changed ownership matrices. By default, standard ownership matrices based on ``firm_ids`` will
            be used unless the ``ownership`` field of ``product_data`` in :class:`Problem` was specified.
        costs : `array-like`
            Potentially changed marginal costs, :math:`c^*`. By default, unchanged marginal costs are computed with
            :meth:`ProblemResults.compute_costs`. Costs under a changed ownership structure can be computed by
            specifying the ``firm_ids`` or ``ownership`` arguments of :meth:`ProblemResults.compute_costs`. If marginal
            costs depend on prices through market shares, they will be updated to reflect different prices during each
            iteration of the routine. Updated marginal costs can be obtained by instead using
            :meth:`Simulation.replace_endogenous`.
        prices : `array-like, optional`
            Prices at which the fixed point iteration routine will start. By default, unchanged prices, :math:`p`, are
            used as starting values. Other reasonable starting prices include the approximate equilibrium prices
            computed by :meth:`ProblemResults.compute_approximate_prices`.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem in each market. By default,
            ``Iteration('simple', {'atol': 1e-12})`` is used.
        constant_costs : `bool, optional`
            Whether to assume that marginal costs, :math:`c`, remain constant as equilibrium prices and shares change.
            By default this is ``True``, which means that firms treat marginal costs as constant (equal to ``costs``)
            when setting prices. This assumption is implicit in how :meth:`ProblemResults.compute_costs` computes
            marginal costs. If set to ``False``, marginal costs will be allowed to adjust if ``shares`` was included in
            the formulation for :math:`X_3` in :class:`Problem`.
        market_id : `object, optional`
            ID of the market in which to compute equilibrium prices. By default, equilibrium prices are computed in all
            markets and stacked.

        Returns
        -------
        `ndarray`
            Estimates of equilibrium prices after any firm or cost changes, :math:`p^*`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Solving for equilibrium prices ...")
        market_ids = self._select_market_ids(market_id)
        firm_ids = self._economy._coerce_optional_firm_ids(firm_ids, market_ids)
        ownership = self._economy._coerce_optional_ownership(ownership, market_ids)
        costs = self._coerce_optional_costs(costs, market_ids)
        prices = self._coerce_optional_prices(prices, market_ids)
        iteration = self._economy._coerce_optional_prices_iteration(iteration)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_prices, market_ids, fixed_args=[iteration, constant_costs],
            market_args=[firm_ids, ownership, costs, prices]
        )

    def compute_shares(
            self, prices: Optional[Any] = None, delta: Optional[Any] = None, agent_data: Optional[Mapping] = None,
            integration: Optional[Integration] = None, market_id: Optional[Any] = None) -> Array:
        r"""Estimate shares.

        It may be desirable to compute the shares associated with equilibrium prices that have been computed, for
        example, by :meth:`ProblemResults.compute_prices`.

        .. note::

           To compute equilibrium shares (and prices) associated with a more complicated counterfactual, a
           :class:`Simulation` for the counterfactual can be initialized with the estimated parameters, structural
           errors, and marginal costs from these results, and then solved with :meth:`Simulation.replace_endogenous`.

        Alternatively, this method can also be used to evaluate the performance of different numerical integration
        configurations. One way to do so is to use :meth:`ProblemResults.compute_delta` to compute mean utilities
        with a very precise integration rule (one that is infeasible to use during estimation), use these same mean
        utilities and integration rule to precisely compute shares, and then compare error between these
        precisely-computed shares and shares computed with less precise (but feasible to use during estimation)
        integration rules, still using the precisely-computed mean utilities.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices at which to evaluate shares, such as equilibrium prices, :math:`p^*`, computed by
            :meth:`ProblemResults.compute_prices`. By default, unchanged prices are used.
        delta : `array-like, optional`
            Mean utilities that will be used to evaluate shares, such as those computed more precisely by
            :meth:`ProblemResults.compute_delta`. By default, the estimated :attr:`ProblemResults.delta` is used,
            and updated with any specified ``prices``.
        agent_data : `structured array-like, optional`
            Agent data that will be used to compute shares. By default, ``agent_data`` in :class:`Problem` is used. For
            more information, refer to :class:`Problem`.
        integration : `Integration, optional`
            :class:`Integration` configuration that will be used to compute shares, which will replace any ``nodes``
            field in ``agent_data``. This configuration is required if ``agent_data`` is specified without a nodes
            field. By default, ``agent_data`` in :class:`Problem` is used. For more information, refer to
            :class:`Problem`.
        market_id : `object, optional`
            ID of the market in which to compute shares. By default, shares are computed in all markets and stacked.

        Returns
        -------
        `ndarray`
            Estimates of shares.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing shares ...")
        market_ids = self._select_market_ids(market_id)
        prices = self._coerce_optional_prices(prices, market_ids)
        delta = self._coerce_optional_delta(delta, market_ids)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_shares, market_ids, market_args=[prices, delta], agent_data=agent_data,
            integration=integration
        )

    def compute_hhi(
            self, firm_ids: Optional[Any] = None, shares: Optional[Any] = None, market_id: Optional[Any] = None) -> (
            Array):
        r"""Estimate Herfindahl-Hirschman Indices, :math:`\text{HHI}`.

        The index in market :math:`t` is

        .. math:: \text{HHI} = \text{10,000} \times \sum_{f \in F_t} \left(\sum_{j \in J_{ft}} s_{jt}\right)^2.

        Parameters
        ----------
        firm_ids : `array-like, optional`
            Firm IDs. By default, the unchanged ``firm_ids`` field of ``product_data`` in :class:`Problem` will be used.
        shares : `array-like, optional`
            Shares, :math:`s`, such as those computed by :meth:`ProblemResults.compute_shares`. By default, unchanged
            shares are used.
        market_id : `object, optional`
            ID of the market in which to compute the index. By default, indices are computed in all markets and stacked.

        Returns
        -------
        `ndarray`
            Estimated Herfindahl-Hirschman Indices, :math:`\text{HHI}`. If ``market_ids`` was not specified, rows are in
            the same order as :attr:`Problem.unique_market_ids`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing HHI ...")
        market_ids = self._select_market_ids(market_id)
        firm_ids = self._economy._coerce_optional_firm_ids(firm_ids, market_ids)
        shares = self._coerce_optional_shares(shares, market_ids)
        return self._combine_arrays(EconomyResultsMarket.safely_compute_hhi, market_ids, market_args=[firm_ids, shares])

    def compute_markups(
            self, prices: Optional[Any] = None, costs: Optional[Any] = None, market_id: Optional[Any] = None) -> Array:
        r"""Estimate markups, :math:`\mathscr{M}`.

        The markup of product :math:`j` in market :math:`t` is

        .. math:: \mathscr{M}_{jt} = \frac{p_{jt} - c_{jt}}{p_{jt}}.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices, :math:`p`, such as equilibrium prices, :math:`p^*`, computed by
            :meth:`ProblemResults.compute_prices`. By default, unchanged prices are used.
        costs : `array-like`
            Marginal costs, :math:`c`. By default, marginal costs are computed with
            :meth:`ProblemResults.compute_costs`. Costs under a changed ownership structure can be computed by
            specifying the ``firm_ids`` or ``ownership`` arguments of :meth:`ProblemResults.compute_costs`.
        market_id : `object, optional`
            ID of the market in which to compute markups. By default, markups are computed in all markets and stacked.

        Returns
        -------
        `ndarray`
            Estimated markups, :math:`\mathscr{M}`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing markups ...")
        market_ids = self._select_market_ids(market_id)
        prices = self._coerce_optional_prices(prices, market_ids)
        costs = self._coerce_optional_costs(costs, market_ids)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_markups, market_ids, market_args=[prices, costs]
        )

    def compute_profits(
            self, prices: Optional[Any] = None, shares: Optional[Any] = None, costs: Optional[Any] = None,
            market_id: Optional[Any] = None) -> Array:
        r"""Estimate population-normalized gross expected profits, :math:`\pi`.

        With constant costs, the profit from product :math:`j` in market :math:`t` is

        .. math:: \pi_{jt} = (p_{jt} - c_{jt})s_{jt}.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices, :math:`p`, such as equilibrium prices, :math:`p^*`, computed by
            :meth:`ProblemResults.compute_prices`. By default, unchanged prices are used.
        shares : `array-like, optional`
            Shares, :math:`s`, such as those computed by :meth:`ProblemResults.compute_shares`. By default, unchanged
            shares are used.
        costs : `array-like`
            Marginal costs, :math:`c`. By default, marginal costs are computed with
            :meth:`ProblemResults.compute_costs`. Costs under a changed ownership structure can be computed by
            specifying the ``firm_ids`` or ``ownership`` arguments of :meth:`ProblemResults.compute_costs`.
        market_id : `object, optional`
            ID of the market in which to compute profits. By default, profits are computed in all markets and stacked.

        Returns
        -------
        `ndarray`
            Estimated population-normalized gross expected profits, :math:`\pi`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing profits ...")
        market_ids = self._select_market_ids(market_id)
        prices = self._coerce_optional_prices(prices, market_ids)
        shares = self._coerce_optional_shares(shares, market_ids)
        costs = self._coerce_optional_costs(costs, market_ids)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_profits, market_ids, market_args=[prices, shares, costs]
        )

    def compute_profit_hessians(
            self, prices: Optional[Any] = None, costs: Optional[Array] = None, market_id: Optional[Any] = None) -> (
            Array):
        r"""Estimate arrays of second derivatives of profits with respect to a prices.

        In market :math:`t`, the value indexed by :math:`(j, k, \ell)` is

        .. math:: \frac{\partial^2 \pi_{jt}}{\partial p_{kt} \partial p_{\ell t}}.

        Profit Hessians can be used to check second order conditions for firms' pricing problem. See
        :attr:`SimulationResults.profit_hessians` and :attr:`SimulationResults.profit_hessian_eigenvalues` for more
        information.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices, :math:`p`, such as equilibrium prices, :math:`p^*`, computed by
            :meth:`ProblemResults.compute_prices`. By default, unchanged prices are used.
        costs : `array-like`
            Marginal costs, :math:`c`. By default, marginal costs are computed with
            :meth:`ProblemResults.compute_costs`. Costs under a changed ownership structure can be computed by
            specifying the ``firm_ids`` or ``ownership`` arguments of :meth:`ProblemResults.compute_costs`.
        market_id : `object, optional`
            ID of the market in which to compute Hessians. By default, Hessians are computed in all markets and
            stacked.

        Returns
        -------
        `ndarray`
            Estimated :math:`J_t \times J_t \times J_t` arrays of second derivatives of profits. If ``market_id`` was
            not specified, arrays are estimated in each market :math:`t` and stacked. Indices for a market are in the
            same order as products for the market. If a market has fewer products than others, extra indices will
            contain ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output(f"Computing second derivatives of profits with respect to prices ...")
        market_ids = self._select_market_ids(market_id)
        prices = self._coerce_optional_prices(prices, market_ids)
        costs = self._coerce_optional_costs(costs, market_ids)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_profit_hessian, market_ids, market_args=[prices, costs]
        )

    def compute_consumer_surpluses(
            self, prices: Optional[Any] = None, keep_all: bool = False, eliminate_product_ids: Optional[Any] = None,
            product_ids_index: int = 0, market_id: Optional[Any] = None) -> Array:
        r"""Estimate population-normalized consumer surpluses, :math:`\text{CS}`.

        Assuming away nonlinear income effects, the surplus in market :math:`t` is

        .. math:: \text{CS} = \sum_{i \in I_t} w_{it}\text{CS}_{it},

        in which the consumer surplus for individual :math:`i` is

        .. math::

           \text{CS}_{it} =
           \log\left(1 + \sum_{j \in J_t} \exp V_{ijt}\right) \Big/
           \left(-\frac{\partial V_{i1t}}{\partial p_{1t}}\right),

        or with nesting parameters,

        .. math::

           \text{CS}_{it} =
           \log\left(1 + \sum_{h \in H} \exp V_{iht}\right) \Big/
           \left(-\frac{\partial V_{i1t}}{\partial p_{1t}}\right)

        where :math:`V_{ijt}` is defined in :eq:`utilities` and :math:`V_{iht}` is defined in :eq:`inclusive_value`.

        .. warning::

           :math:`\frac{\partial V_{1ti}}{\partial p_{1t}}` is the derivative of utility for the first product with
           respect to its price. The first product is chosen arbitrarily because this method assumes that there are no
           nonlinear income effects, which implies that this derivative is the same for all products. Computed consumer
           surpluses will likely be incorrect if prices are formulated in a nonlinear fashion like ``log(prices)``.

        Comparing consumer surpluses with the same values computed after eliminating one or more products from the
        agents' choice sets (i.e. setting :math:`\exp V_{ijt} = 0` for eliminated products :math:`j`) gives a measure of
        willingness to pay. This can be done with the ``eliminate_product_ids`` argument.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices at which utilities and price derivatives will be evaluated, such as equilibrium prices, :math:`p^*`,
            computed by :meth:`ProblemResults.compute_prices`. By default, unchanged prices are used.
        keep_all : `bool, optional`
            Whether to keep all individuals' surpluses :math:`\text{CS}_{it}` or just market-level surpluses.
            By default only market-level surpluses are returned, but returning all surpluses will be important for
            analysis by agent type or demographic category.
        eliminate_product_ids : `sequence of object, optional`
            IDs of the products to eliminate from the choice set. These IDs should show up in the ``product_ids`` field
            of ``product_data`` in :class:`Problem`. Eliminating one or more products and comparing consumer surpluses
            gives a measure of willingness to pay for these products.
        product_ids_index : `int, optional`
            Index between ``0`` and the number of columns in the ``product_ids`` field of ``product_data`` minus one,
            inclusive, which determines which column of product IDs ``eliminate_product_ids`` refers to. By default, it
            refers to the first column, which is index ``0``.
        market_id : `object, optional`
            ID of the market in which to compute consumer surplus. By default, consumer surpluses are computed in all
            markets and stacked.

        Returns
        -------
        `ndarray`
            Estimated population-normalized consumer surpluses, :math:`\text{CS}` (or individuals' surpluses if
            ``keep_all`` is ``True``). If ``market_ids`` was not specified, rows are in the same order as
            :attr:`Problem.unique_market_ids`. If ``keep_all`` is ``True``, columns for a market are in the same order
            as agents for the market. If a market has fewer agents than others, extra columns will contain
            ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing consumer surpluses with the equation that assumes away nonlinear income effects ...")
        if eliminate_product_ids is not None:
            self._economy._validate_product_ids_index(product_ids_index)
        market_ids = self._select_market_ids(market_id)
        prices = self._coerce_optional_prices(prices, market_ids)
        return self._combine_arrays(
            EconomyResultsMarket.safely_compute_consumer_surplus, market_ids,
            fixed_args=[keep_all, eliminate_product_ids, product_ids_index], market_args=[prices]
        )


class EconomyResults(SimpleEconomyResults):
    """Abstract results for an economy underlying the BLP model, supporting more complicated methods."""

    _sigma: Array
    _pi: Array
    _rho: Array
    _beta: Array
    _gamma: Array
    _delta: Array
    _data_override: Optional[Dict[str, Array]]

    def __init__(
            self, economy: 'Economy', parameters: Parameters, sigma: Array, pi: Array, rho: Array, beta: Array,
            gamma: Array, delta: Array, data_override: Optional[Dict[str, Array]] = None) -> None:
        """Store the underlying economy and parameter information."""
        super().__init__(economy, parameters)
        self._sigma = sigma
        self._pi = pi
        self._rho = rho
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._data_override = data_override

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
            agents = self._economy.agents
            agents_market_indices = self._economy._agent_market_indices
        else:
            agents = Agents(self._economy.products, self._economy.agent_formulation, agent_data, integration)
            agents_market_indices = get_indices(agents.market_ids)

        def market_factory(s: Hashable) -> tuple:
            """Build a market along with arguments used to compute arrays."""
            indices_s = self._economy._product_market_indices[s]
            market_s = EconomyResultsMarket(
                self._economy, s, self._parameters, self._sigma, self._pi, self._rho, self._beta, self._gamma,
                self._delta, self._data_override, agents_override=agents[agents_market_indices[s]]
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
            elif dimension_sizes[0] == self._economy.N:
                combined[(self._economy._product_market_indices[t], *slices)] = array_t
            else:
                assert market_ids.size == 1
                combined = array_t

        # output how long it took to compute the arrays
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return combined

    def compute_micro_values(self, micro_moments: Sequence[MicroMoment]) -> Array:
        r"""Estimate micro moment values, :math:`f_m(v)`.

        Parameters
        ----------
        micro_moments : `sequence of MicroMoment`
            :class:`MicroMoment` instances. The ``value`` argument is ignored.

        Returns
        -------
        `ndarray`
            Micro moment values :math:`f_m(v)`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        errors: List[Error] = []

        # keep track of long it takes to compute micro moment values
        output("Computing micro moment values ...")
        start_time = time.time()

        # validate and structure micro moments
        moments = Moments(micro_moments, self._economy)
        if moments.MM == 0:
            return np.array([], options.dtype)

        def market_factory(s: Hashable) -> Tuple[EconomyResultsMarket, Moments]:
            """Build a market along with arguments used to compute micro moment values."""
            market_s = EconomyResultsMarket(
                self._economy, s, self._parameters, self._sigma, self._pi, self._rho, self._beta, self._gamma,
                self._delta, self._data_override
            )
            return market_s, moments

        # compute micro moment part contributions market-by-market
        parts_numerator_mapping: Dict[Hashable, Array] = {}
        parts_denominator_mapping: Dict[Hashable, Array] = {}
        generator = generate_items(
            self._economy.unique_market_ids, market_factory, EconomyResultsMarket.safely_compute_micro_contributions
        )
        for t, (parts_numerator_t, parts_denominator_t, errors_t) in generator:
            parts_numerator_mapping[t] = scipy.sparse.csr_matrix(parts_numerator_t)
            parts_denominator_mapping[t] = scipy.sparse.csr_matrix(parts_denominator_t)
            errors.extend(errors_t)

        # aggregate micro moments across all markets (this is done after market-by-market computation to preserve
        #   numerical stability with different market orderings)
        with np.errstate(all='ignore'):
            # construct micro moment parts
            parts_numerator = scipy.sparse.csr_matrix((moments.PM, 1), dtype=options.dtype)
            parts_denominator = scipy.sparse.csr_matrix((moments.PM, 1), dtype=options.dtype)
            for t in self._economy.unique_market_ids:
                parts_numerator += parts_numerator_mapping[t]
                parts_denominator += parts_denominator_mapping[t]

            parts_numerator = parts_numerator.toarray()
            parts_denominator = parts_denominator.toarray()
            parts_values = parts_numerator / parts_denominator

            # from the parts, construct micro moment values
            micro_values = np.zeros((moments.MM, 1), options.dtype)
            for m, moment in enumerate(moments.micro_moments):
                part_indices = [moments.micro_parts.index(p) for p in moment.parts]
                micro_value = moment.compute_value(parts_values[part_indices])
                micro_value = np.asarray(micro_value).flatten()
                if micro_value.size != 1:
                    raise TypeError(f"compute_value of micro moment '{moment}' should return a float.")

                micro_values[m] = micro_value

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # output how long it took to compute the micro values
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return micro_values.flatten()

    def compute_micro_scores(
            self, dataset: MicroDataset, micro_data: Mapping, integration: Optional[Integration] = None) -> List[Array]:
        r"""Compute scores for observations :math:`n \in N_d` from a micro dataset :math:`d`.

        The score for observation :math:`n \in N_d` is

        .. math::
           :label: score

           \mathscr{S}_n = \frac{\partial\log\mathscr{P}_n}{\partial\theta'},

        in which the conditional probability of observation :math:`n` is

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
            The :class:`MicroDataset` for which scores will be computed. The ``compute_weights`` function is called
            separately for each observation :math:`n`.
        micro_data : `structured array-like`
            Each row corresponds either to an observation :math:`n` or if there are multiple rows per observation, to
            an :math:`i \in I_n` that integrates over unobserved heterogeneity. In addition to the names of any
            demographics used in the ``agent_formulation`` and any specification of agent-specific product
            ``'availability'``, the following fields are required:

                - **market_ids** : (`object`) - Market IDs :math:`t_n` for each observation :math:`n`.

                - **choice_indices** : (`int`) - Within-market indices of choices :math:`j_n`. If ``compute_weights``
                  passed to the ``dataset`` returns an array with :math:`J_t` elements in its second axis, then choice
                  indices take on values from :math:`0` to :math:`J_t - 1` where :math:`0` corresponds to the first
                  inside good. If it returns an array with :math:`1 + J_t` elements in its second axis, then choice
                  indices take on values from :math:`0` to :math:`J_t` where :math:`0` corresponds to the outside good.

            If the ``dataset`` is configured to support second choice data, second choices are also required:

                - **second_choice_indices** : (`int, optional`) - Within-market indices of second choices :math:`k_n`.
                  If ``compute_weights`` passed to the ``dataset`` returns an array with :math:`J_t` elements in its
                  third axis, then second choice indices take on values from :math:`0` to :math:`J_t - 1` where
                  :math:`0` corresponds to the first inside good. If it returns an array with :math:`1 + J_t` elements
                  in its third axis, then second choice indices take on values from :math:`0` to :math:`J_t` where
                  :math:`0` corresponds to the outside good.

            The following fields are required if ``integration`` is not specified:

                - **micro_ids** : (`object, optional`) - IDs corresponding to observations :math:`n`, which should be
                  pre-sorted, from smallest to largest.

                - **weights** : (`numeric, optional`) - Integration weights, :math:`w_{it_n}`, for integration over
                  unobserved heterogeneity :math:`i \in I_n`.

                - **nodes** : (`numeric, optional`) - Unobserved agent characteristics called integration nodes,
                  :math:`\nu`. If there are more than :math:`K_2` columns (the number of demand-side nonlinear product
                  characteristics), only the first :math:`K_2` will be retained. If any columns of ``sigma`` are fixed
                  at zero, only the first few columns of these nodes will be used.

            If these fields are specified, each row corresponds to an :math:`i \in I_n`, and there should generally be
            multiple rows per observation :math:`n`.

            The convenience function :func:`build_integration` can be useful when constructing custom nodes and weights.

            .. note::

               If ``nodes`` has multiple columns, it can be specified as a matrix or broken up into multiple
               one-dimensional fields with column index suffixes that start at zero. For example, if there are three
               columns of nodes, a ``nodes`` field with three columns can be replaced by three one-dimensional fields:
               ``nodes0``, ``nodes1``, and ``nodes2``.

        integration : `Integration, optional`
            :class:`Integration` configuration for how to build ``nodes`` and ``weights`` fields in ``micro_data`` for
            each observation :math:`n`. If this configuration is specified, any ``micro_ids``, ``weights``, and
            ``nodes`` in ``micro_data`` will be ignored.

            If specified, each row of ``micro_data`` is treated as corresponding to a unique observation :math:`n`,
            and will be duplicated by as many rows of nodes as are created by the :class:`Integration` configuration.
            Specifically, up to :math:`K_2` columns of nodes (the number of demand-side nonlinear product
            characteristics) will be built for each observation :math:`n`. If there are zeros on the diagonal of
            :math:`\Sigma`, nodes will not be built for those characteristics, to cut down on memory usage.

        Returns
        -------
        `list`
            Scores :math:`\mathscr{S}_n`. The list is in the same order as :attr:`ProblemResults.theta` (also see
            :attr:`ProblemResults.theta_labels`). Each element of the list is an array of scores for the corresponding
            parameter. The array is in the same order as observations appear in the ``micro_data``. Note that it is
            possible for parameters in :attr:`ProblemResults.theta` to mechanically have zero scores, for example if
            they are on a constant demographic.

            Taking the mean of a parameter's scores delivers the observed ``value`` for an optimal
            :class:`MicroMoment` that matches the score for that parameter.

            If any scores are ``numpy.nan``, this means that the probability of that observation is
            :math:`\mathscr{P}_n = 0`, suggesting that the observation was not generated by the sampling process
            defined by the ``dataset``.

        """

        # keep track of long it takes to compute scores
        output("Computing micro scores ...")
        start_time = time.time()

        # build agents and verify that they have choices
        demographics, demographics_formulations = build_demographics(
            self._economy.products, micro_data, self._economy.agent_formulation
        )
        micro_agents = MicroAgents(
            self._economy.products, self._parameters, micro_data, demographics, demographics_formulations, integration
        )
        if micro_agents.choice_indices.size == 0:
            raise KeyError("micro_data must have choice_indices.")

        # compute the contributions
        numerator_mapping, numerator_jacobian_mapping, denominator_mapping, denominator_jacobian_mapping, errors = (
            self._compute_scores(dataset, micro_agents)
        )

        # compute the denominator contributions
        unique_market_ids = np.unique(micro_agents.market_ids)
        denominator = np.stack([denominator_mapping[t] for t in unique_market_ids]).sum()
        denominator_jacobian = np.stack([denominator_jacobian_mapping[t] for t in unique_market_ids]).sum(axis=0)

        # stack the numerator contributions
        unique_micro_ids = np.unique(micro_agents.micro_ids)
        numerator = np.stack([numerator_mapping[n] for n in unique_micro_ids])
        numerator_jacobian = np.stack([numerator_jacobian_mapping[n] for n in unique_micro_ids])

        # construct the scores
        scores = []
        for p in range(self._parameters.P):
            with np.errstate(all='ignore'):
                scores_p = numerator_jacobian[..., p] / numerator - denominator_jacobian[..., p] / denominator
            scores_p[~np.isfinite(scores_p)] = np.nan
            scores.append(scores_p)

        # output how long it took to compute scores
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return scores

    def compute_agent_scores(
            self, dataset: MicroDataset, micro_data: Optional[Mapping] = None,
            integration: Optional[Integration] = None) -> Array:
        r"""Compute scores for all agent-choices, treated as observations :math:`n \in N_d` from a micro dataset
        :math:`d`.

        This method is the same as :meth:`ProblemResults.compute_micro_scores`, except it computes scores for all
        possible choices of all :attr:`Problem.agents`. Each agent-choice is treated as a separate observation
        :math:`n`. Instead of returning an array, this method returns a mapping from market IDs to scores, to facilitate
        use by ``compute_values`` of an optimal :class:`MicroMoment`.

        Parameters
        ----------
        dataset : `MicroDataset`
            The :class:`MicroDataset` for which scores will be computed. The ``compute_weights`` function is called
            separately for each observation :math:`n`.
        micro_data : `structured array-like, optional`
            By default, each row in :attr:`Problem.agents` and each possible choice is treated as an observation
            :math:`n`. In this case, ``integration`` should generally be specified to define integration
            :math:`i \in I_n` over unobserved heterogeneity.

            If ``micro_data`` is specified, it should be of the form required by
            :meth:`ProblemResults.compute_micro_scores`, except without ``choice_indices`` or ``second_choice_indices``,
            since scores will be computed for all choices.

        integration : `Integration, optional`
            :class:`Integration` configuration of the form required by :meth:`ProblemResults.compute_micro_scores`.

        Returns
        -------
        `list`
            Scores :math:`\mathscr{S}_n`. The list is in the same order as :attr:`ProblemResults.theta` (also see
            :attr:`ProblemResults.theta_labels`). Each element of the list is a mapping from market IDs supported by the
            ``dataset`` to an array of scores for the corresponding parameter and market. The array's dimensions
            correspond to the dimensions of the weights returned by ``compute_weights`` passed to ``dataset``. Note that
            it is possible for parameters in :attr:`ProblemResults.theta` to mechanically have zero scores, for example
            if they are on a constant demographic.

            To build an optimal :class:`MicroMoment` that matches the score for a parameter, ``compute_values``
            in its single :class:`MicroPart` should select the array corresponding to that parameter and the requested
            market ``t``. Any ``numpy.nan`` values in this array correspond to agent-choices that are assigned a
            probability of :math:`\mathscr{P}_n = 0` by the sampling process defined by ``dataset``, so should be
            replaced by some arbitrary number (e.g., by passing the array of scores through ``numpy.nan_to_num``).

        """

        # keep track of long it takes to compute scores
        output("Computing agent scores ...")
        start_time = time.time()

        # build micro data
        if micro_data is not None:
            demographics, demographics_formulations = build_demographics(
                self._economy.products, micro_data, self._economy.agent_formulation
            )
            micro_agents = MicroAgents(
                self._economy.products, self._parameters, micro_data, demographics, demographics_formulations,
                integration
            )
        else:
            if dataset.market_ids is None:
                agents = self._economy.agents
            else:
                agent_indices = np.concatenate([self._economy._agent_market_indices[t] for t in dataset.market_ids])
                agents = self._economy.agents[agent_indices]

            demographics = agents.demographics
            demographics_formulations = agents.dtype.fields['demographics'][2]
            micro_data = {
                'micro_ids': np.arange(agents.size),
                'market_ids': agents.market_ids,
                'agent_ids': agents.agent_ids,
                'nodes': agents.nodes,
                'weights': np.ones(agents.size),
                'availability': agents.availability,
            }
            micro_agents = MicroAgents(
                self._economy.products, self._parameters, micro_data, demographics, demographics_formulations,
                integration
            )

        # compute the contributions
        numerator_mapping, numerator_jacobian_mapping, denominator_mapping, denominator_jacobian_mapping, errors = (
            self._compute_scores(dataset, micro_agents)
        )

        # compute the denominator contributions
        unique_market_ids = np.unique(micro_agents.market_ids)
        denominator = np.stack([denominator_mapping[t] for t in unique_market_ids]).sum()
        denominator_jacobian = np.stack([denominator_jacobian_mapping[t] for t in unique_market_ids]).sum(axis=0)

        # construct the scores
        scores: List[Dict[Hashable, Array]] = []
        for t, indices in get_indices(micro_agents.market_ids).items():
            market_micro_ids = np.unique(micro_agents.micro_ids[indices])
            numerator = np.stack([numerator_mapping[n] for n in market_micro_ids])
            numerator_jacobian = np.stack([numerator_jacobian_mapping[n] for n in market_micro_ids])
            for p in range(self._parameters.P):
                if p == len(scores):
                    scores.append({})
                with np.errstate(all='ignore'):
                    scores_pt = numerator_jacobian[..., p] / numerator - denominator_jacobian[..., p] / denominator
                scores_pt[~np.isfinite(scores_pt)] = np.nan
                scores[p][t] = scores_pt

        # output how long it took to compute scores
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return scores

    def _compute_scores(
            self, dataset: MicroDataset, micro_agents: RecArray) -> (
            Tuple[
                Dict[Hashable, Array], Dict[Hashable, Array], Dict[Hashable, Array], Dict[Hashable, Array], List[Error]
            ]):
        """Compute scores for observations from a micro dataset."""
        errors: List[Error] = []

        # validate the micro dataset
        if not isinstance(dataset, MicroDataset):
            raise TypeError("dataset must be a MicroDataset.")
        dataset._validate(self._economy)

        # collect information about micro and market IDs
        unique_micro_ids = np.unique(micro_agents.micro_ids)
        unique_market_ids = np.unique(micro_agents.market_ids)
        micro_indices = get_indices(micro_agents.micro_ids)

        # verify that the micro data only has market IDs supported by the dataset
        dataset_market_ids = dataset.market_ids
        if dataset_market_ids is None:
            dataset_market_ids = set(self._economy.unique_market_ids)
        if set(unique_market_ids) - dataset_market_ids:
            raise ValueError("The market_ids field of micro_data must not have IDs not supported by the dataset.")

        def denominator_market_factory(s: Hashable) -> Tuple[EconomyResultsMarket, MicroDataset]:
            """Build a market along with arguments used to compute denominator contributions."""
            market_s = EconomyResultsMarket(
                self._economy, s, self._parameters, self._sigma, self._pi, self._rho, self._beta, self._gamma,
                self._delta, self._data_override
            )
            return market_s, dataset

        # construct mappings from market IDs to xi Jacobians and denominator contributions
        xi_jacobian_mapping: Dict[Hashable, Array] = {}
        denominator_mapping: Dict[Hashable, Array] = {}
        denominator_jacobian_mapping: Dict[Hashable, Array] = {}
        generator = generate_items(
            unique_market_ids, denominator_market_factory,
            EconomyResultsMarket.safely_compute_score_denominator_contributions
        )
        for t, (xi_jacobian_t, denominator_t, denominator_jacobian_t, errors_t) in generator:
            xi_jacobian_mapping[t] = xi_jacobian_t
            denominator_mapping[t] = denominator_t
            denominator_jacobian_mapping[t] = denominator_jacobian_t
            errors.extend(errors_t)

        def numerator_market_factory(i: Hashable) -> tuple:
            """Build a market for a micro observation along with arguments used to numerator contributions."""
            t_i = micro_agents.market_ids[micro_indices[i]].take(0)
            j_i = k_i = None
            if micro_agents.choice_indices.size > 0:
                j_i = micro_agents.choice_indices[micro_indices[i]].take(0)
            if micro_agents.second_choice_indices.size > 0:
                k_i = micro_agents.second_choice_indices[micro_indices[i]].take(0)

            products_i = self._economy.products[self._economy._product_market_indices[t_i]]
            micro_agents_i = micro_agents[micro_indices[i]]
            market_i = EconomyResultsMarket(
                self._economy, t_i, self._parameters, self._sigma, self._pi, self._rho, self._beta, self._gamma,
                self._delta, self._data_override, products_i, micro_agents_i,
            )
            return market_i, dataset, j_i, k_i, xi_jacobian_mapping[t_i]

        # construct mappings from observations to numerator contributions
        numerator_mapping: Dict[Hashable, Array] = {}
        numerator_jacobian_mapping: Dict[Hashable, Array] = {}
        generator = generate_items(
            unique_micro_ids, numerator_market_factory,
            EconomyResultsMarket.safely_compute_score_numerator_contributions
        )
        if unique_micro_ids.size > 1:
            generator = output_progress(generator, unique_micro_ids.size, time.time())
        for n, (numerator_n, numerator_jacobian_n, errors_n) in generator:
            numerator_mapping[n] = numerator_n
            numerator_jacobian_mapping[n] = numerator_jacobian_n
            errors.extend(errors_n)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        return numerator_mapping, numerator_jacobian_mapping, denominator_mapping, denominator_jacobian_mapping, errors

    def simulate_micro_data(self, dataset: MicroDataset, seed: Optional[int] = None) -> RecArray:
        r"""Simulate observations :math:`n \in N_d` from a micro dataset :math:`d`.

        Each micro observation :math:`n` underlying the dataset :math:`d` is simulated according to agent weights
        :math:`w_{it}`, choice probabilities :math:`s_{ijt}`, and survey weights :math:`w_{dijt}`.

        Parameters
        ----------
        dataset : `MicroDataset`
            The :class:`MicroDataset` for which micro data will be simulated.
        seed : `int, optional`
            Passed to :class:`numpy.random.RandomState` to seed the random number generator before data are simulated.
            By default, a seed is not passed to the random number generator.

        Returns
        -------
        `recarray`
            Micro data with as many rows as ``observations`` passed to the ``dataset``. Fields:

            - **micro_ids** : (`object`) - IDs corresponding to observations :math:`n`.

            - **market_ids** : (`object`) - Market IDs :math:`t_n` for each observation :math:`n`.

            - **agent_indices** : (`int`) - Within-market indices of agents :math:`i_n` that take on values from
              :math:`0` to :math:`I_t - 1`.

            - **choice_indices** : (`int`) - Within-market indices of simulated choices :math:`j_n`. If
              ``compute_weights`` passed to the ``dataset`` returns an array with :math:`J_t` elements in its second
              axis, then choice indices take on values from :math:`0` to :math:`J_t - 1` where :math:`0` corresponds to
              the first inside good. If it returns an array with :math:`1 + J_t` elements in its second axis, then
              choice indices take on values from :math:`0` to :math:`J_t` where :math:`0` corresponds to the outside
              good.

            If the ``dataset`` is configured to support second choice data, second choices will also be simulated:

            - **second_choice_indices** : (`int`) - Within-market indices of simulated second choices :math:`k_n`. If
              ``compute_weights`` passed to the ``dataset`` returns an array with :math:`J_t` elements in its third
              axis, then second choice indices take on values from :math:`0` to :math:`J_t - 1` where :math:`0`
              corresponds to the first inside good. If it returns an array with :math:`1 + J_t` elements in its third
              axis, then second choice indices take on values from :math:`0` to :math:`J_t` where :math:`0` corresponds
              to the outside good.

            Integration nodes and demographics can be merged in on the ``market_ids`` and ``agent_indices`` fields.
            Product characteristics can be merged in on the ``market_ids`` and ``choice_indices`` fields. Product
            characteristics of any second choices can be merged in on the ``market_ids`` and ``second_choice_indices``
            fields.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        errors: List[Error] = []

        # keep track of long it takes to simulate micro data
        output("Simulating micro data ...")
        start_time = time.time()

        # validate the micro dataset
        if not isinstance(dataset, MicroDataset):
            raise TypeError("dataset must be a MicroDataset.")
        dataset._validate(self._economy)

        # determine the datatypes to use to conserve on memory
        agent_dtype = choice_dtype = np.uint64
        for dtype in [np.uint32, np.uint8]:
            if self._economy._max_I <= np.iinfo(dtype).max:
                agent_dtype = dtype
            if self._economy._max_J <= np.iinfo(dtype).max:
                choice_dtype = dtype

        # collect the relevant market ids
        if dataset.market_ids is None:
            market_ids = self._economy.unique_market_ids
        else:
            market_ids = np.asarray(list(dataset.market_ids))

        def market_factory(s: Hashable) -> Tuple[EconomyResultsMarket, MicroDataset]:
            """Build a market along with arguments used to compute weights needed for simulation."""
            assert dataset is not None
            market_s = EconomyResultsMarket(
                self._economy, s, self._parameters, self._sigma, self._pi, self._rho, self._beta, self._gamma,
                self._delta, self._data_override
            )
            return market_s, dataset

        # construct mappings from market IDs to probabilities, IDs, and indices needed for simulation
        weights_mapping: Dict[Hashable, Array] = {}
        agent_indices_mapping: Dict[Hashable, Array] = {}
        choice_indices_mapping: Dict[Hashable, Array] = {}
        second_choice_indices_mapping: Dict[Hashable, Array] = {}
        generator = generate_items(market_ids, market_factory, EconomyResultsMarket.safely_compute_micro_weights)
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
        micro_data_mapping: Dict[str, Tuple[Array, Any]] = collections.OrderedDict()
        micro_data_mapping['micro_ids'] = (np.arange(dataset.observations), np.object_)
        micro_data_mapping['market_ids'] = (
            np.concatenate([np.full(agent_indices_mapping[t].size, t) for t in market_ids])[choices], np.object_
        )
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
        micro_data = structure_matrices(micro_data_mapping)

        # output how long this took
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return micro_data
