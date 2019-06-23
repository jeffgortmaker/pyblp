"""Economy-level structuring of abstract BLP problem results."""

import abc
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

import numpy as np

from ..configurations.iteration import Iteration
from ..markets.results_market import ResultsMarket
from ..moments import EconomyMoments
from ..parameters import Parameters
from ..utilities.basics import Array, StringRepresentation, output


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import ProblemEconomy  # noqa


class Results(abc.ABC, StringRepresentation):
    """Abstract results of a solved BLP problem."""

    problem: 'ProblemEconomy'
    _parameters: Parameters
    _moments: EconomyMoments

    def __init__(self, problem: 'ProblemEconomy', parameters: Parameters, moments: EconomyMoments) -> None:
        """Store the underlying problem and parameter information."""
        self.problem = problem
        self._parameters = parameters
        self._moments = moments

    def _select_market_ids(self, market_id: Optional[Any] = None) -> Array:
        """Select either a single market ID or all unique IDs."""
        if market_id is None:
            return self.problem.unique_market_ids
        if market_id in self.problem.unique_market_ids:
            return np.array(market_id, np.object)
        raise ValueError(f"market_id must be None or one of {list(sorted(self.problem.unique_market_ids))}.")

    @abc.abstractmethod
    def _coerce_matrices(self, matrices: Any, market_ids: Array) -> Array:
        """Coerce array-like stacked arrays into a stacked array and validate it."""

    @abc.abstractmethod
    def _coerce_optional_costs(self, costs: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like costs into an array and validate it."""

    @abc.abstractmethod
    def _coerce_optional_prices(self, prices: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like prices into an array and validate it."""

    @abc.abstractmethod
    def _coerce_optional_shares(self, shares: Optional[Any], market_ids: Array) -> Array:
        """Coerce optional array-like shares into an array and validate it."""

    @abc.abstractmethod
    def _combine_arrays(
            self, compute_market_results: Callable, market_ids: Array, fixed_args: Sequence = (),
            market_args: Sequence = ()) -> Array:
        """Combine arrays from one or all markets, which are computed by passing fixed_args (identical for all markets)
        and market_args (arrays that need to be restricted to markets) to compute_market_results, a ResultsMarket method
        that returns the output for the market and any errors encountered during computation.
        """

    def compute_aggregate_elasticities(
            self, factor: float = 0.1, name: str = 'prices', market_id: Optional[Any] = None) -> Array:
        r"""Estimate aggregate elasticities of demand, :math:`\mathscr{E}`, with respect to a variable, :math:`x`.

        In market :math:`t`, the aggregate elasticity of demand is

        .. math:: \mathscr{E} = \sum_{j=1}^{J_t} \frac{s_{jt}(x + \Delta x) - s_{jt}}{\Delta},

        in which :math:`\Delta` is a scalar factor and :math:`s_{jt}(x + \Delta x)` is the share of product :math:`j` in
        market :math:`t`, evaluated at the scaled values of the variable.

        Parameters
        ----------
        factor : `float, optional`
            The scalar factor, :math:`\Delta`.
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices.
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
        output(f"Computing aggregate elasticities with respect to {name} ...")
        if not isinstance(factor, float):
            raise ValueError("factor must be a float.")
        self.problem._validate_name(name)
        market_ids = self._select_market_ids(market_id)
        return self._combine_arrays(
            ResultsMarket.safely_compute_aggregate_elasticity, market_ids, fixed_args=[factor, name]
        )

    def compute_elasticities(self, name: str = 'prices', market_id: Optional[Any] = None) -> Array:
        r"""Estimate matrices of elasticities of demand, :math:`\varepsilon`, with respect to a variable, :math:`x`.

        For each market, the value in row :math:`j` and column :math:`k` of :math:`\varepsilon` is

        .. math:: \varepsilon_{jk} = \frac{x_k}{s_j}\frac{\partial s_j}{\partial x_k}.

        Parameters
        ----------
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices.
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
        output(f"Computing elasticities with respect to {name} ...")
        self.problem._validate_name(name)
        market_ids = self._select_market_ids(market_id)
        return self._combine_arrays(ResultsMarket.safely_compute_elasticities, market_ids, fixed_args=[name])

    def compute_diversion_ratios(self, name: str = 'prices', market_id: Optional[Any] = None) -> Array:
        r"""Estimate matrices of diversion ratios, :math:`\mathscr{D}`, with respect to a variable, :math:`x`.

        Diversion ratios to the outside good are reported on diagonals. For each market, the value in row :math:`j` and
        column :math:`k` is

        .. math:: \mathscr{D}_{jk} = -\frac{\partial s_{k(j)}}{\partial x_j} \Big/ \frac{\partial s_j}{\partial x_j},

        in which :math:`s_{k(j)}` is :math:`s_0 = 1 - \sum_j s_j` if :math:`j = k`, and is :math:`s_k` otherwise.

        Parameters
        ----------
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices.
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
        output(f"Computing diversion ratios with respect to {name} ...")
        self.problem._validate_name(name)
        market_ids = self._select_market_ids(market_id)
        return self._combine_arrays(ResultsMarket.safely_compute_diversion_ratios, market_ids, fixed_args=[name])

    def compute_long_run_diversion_ratios(self, market_id: Optional[Any] = None) -> Array:
        r"""Estimate matrices of long-run diversion ratios, :math:`\bar{\mathscr{D}}`.

        Long-run diversion ratios to the outside good are reported on diagonals. For each market, the value in row
        :math:`j` and column :math:`k` is

        .. math:: \bar{\mathscr{D}}_{jk} = \frac{s_{k(-j)} - s_k}{s_j},

        in which :math:`s_{k(-j)}` is the share of product :math:`k` computed with the outside option removed from the
        choice set if :math:`j = k`, and with product :math:`j` removed otherwise.

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
        return self._combine_arrays(ResultsMarket.safely_compute_long_run_diversion_ratios, market_ids)

    def compute_probabilities(self, market_id: Optional[Any] = None) -> Array:
        r"""Estimate matrices of choice probabilities.

        For each market, the value in row :math:`j` and column `i` is given by :eq:`probabilities` when there are random
        coefficients, and by :eq:`nested_probabilities` when there is additionally a nested structure. For the logit
        and nested logit models, choice probabilities are marketshares.

        Parameters
        ----------
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
        return self._combine_arrays(ResultsMarket.safely_compute_probabilities, market_ids)

    def extract_diagonals(self, matrices: Any) -> Array:
        r"""Extract diagonals from stacked :math:`J_t \times J_t` matrices for each market :math:`t`.

        Parameters
        ----------
        matrices : `array-like`
            Stacked matrices, such as estimates of :math:`\varepsilon`, computed by
            :meth:`ProblemResults.compute_elasticities`; :math:`\mathscr{D}`, computed by
            :meth:`ProblemResults.compute_diversion_ratios`; :math:`\bar{\mathscr{D}}`, computed by
            :meth:`ProblemResults.compute_long_run_diversion_ratios`; or :math:`s_{jti}` computed by
            :meth:`ProblemResults.compute_probabilities`.

        Returns
        -------
        `ndarray`
            Stacked matrix diagonals. If the matrices are estimates of :math:`\varepsilon`, a diagonal is a market's own
            elasticities of demand; if they are estimates of :math:`\mathscr{D}` or :math:`\bar{\mathscr{D}}`, a
            diagonal is a market's diversion ratios to the outside good.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Extracting diagonals ...")
        market_ids = self._select_market_ids()
        matrices = self._coerce_matrices(matrices, market_ids)
        return self._combine_arrays(ResultsMarket.safely_extract_diagonal, market_ids, market_args=[matrices])

    def extract_diagonal_means(self, matrices: Any) -> Array:
        r"""Extract means of diagonals from stacked :math:`J_t \times J_t` matrices for each market :math:`t`.

        Parameters
        ----------
        matrices : `array-like`
            Stacked matrices, such as estimates of :math:`\varepsilon`, computed by
            :meth:`ProblemResults.compute_elasticities`; :math:`\mathscr{D}`, computed by
            :meth:`ProblemResults.compute_diversion_ratios`; :math:`\bar{\mathscr{D}}`, computed by
            :meth:`ProblemResults.compute_long_run_diversion_ratios`; or :math:`s_{jti}` computed by
            :meth:`ProblemResults.compute_probabilities`.

        Returns
        -------
        `ndarray`
            Stacked diagonal means. If the matrices are estimates of :math:`\varepsilon`, the mean of a diagonal is a
            market's mean own elasticity of demand; if they are estimates of :math:`\mathscr{D}` or
            :math:`\bar{\mathscr{D}}`, the mean of a diagonal is a market's mean diversion ratio to the outside good.
            Rows are in the same order as :attr:`Problem.unique_market_ids`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Extracting diagonal means ...")
        market_ids = self._select_market_ids()
        matrices = self._coerce_matrices(matrices, market_ids)
        return self._combine_arrays(ResultsMarket.safely_extract_diagonal_mean, market_ids, market_args=[matrices])

    def compute_costs(self, market_id: Optional[Any] = None) -> Array:
        r"""Estimate marginal costs, :math:`c`.

        Marginal costs are computed with the :math:`\eta`-markup equation in :eq:`eta`:

        .. math:: c = p - \eta.

        Parameters
        ----------
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
        return self._combine_arrays(ResultsMarket.safely_compute_costs, market_ids)

    def compute_approximate_prices(
            self, firm_ids: Optional[Any] = None, ownership: Optional[Any] = None, costs: Optional[Any] = None,
            market_id: Optional[Any] = None) -> Array:
        r"""Approximate equilibrium prices after firm or cost changes, :math:`p^*`, under the assumption that shares and
        their price derivatives are unaffected by such changes.

        This approximation is discussed in, for example, :ref:`references:Nevo (1997)`. Prices in each market are
        computed according to the :math:`\eta`-markup equation in :eq:`eta`:

        .. math:: p^* = c^* + \eta^*,

        in which the markup term is approximated with

        .. math:: \eta^* \approx -\left(O^* \odot \frac{\partial s}{\partial p}\right)^{-1}s

        where :math:`O^*` is the ownership matrix associated with firm changes.

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
            :meth:`ProblemResults.compute_costs`.
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
        firm_ids = self.problem._coerce_optional_firm_ids(firm_ids, market_ids)
        ownership = self.problem._coerce_optional_ownership(ownership, market_ids)
        costs = self._coerce_optional_costs(costs, market_ids)
        return self._combine_arrays(
            ResultsMarket.safely_compute_approximate_equilibrium_prices, market_ids,
            market_args=[firm_ids, ownership, costs]
        )

    def compute_prices(
            self, firm_ids: Optional[Any] = None, ownership: Optional[Any] = None, costs: Optional[Any] = None,
            prices: Optional[Any] = None, iteration: Optional[Iteration] = None, market_id: Optional[Any] = None) -> (
            Array):
        r"""Estimate equilibrium prices after firm or cost changes, :math:`p^*`.

        .. note::

           To compute equilibrium prices (and shares) associated with a more complicated counterfactual, a
           :class:`Simulation` for the counterfactual can be initialized with the estimated parameters, structural
           errors, and marginal costs from these results, and then solved with :meth:`Simulation.solve`. The returned
           :class:`SimulationResults` gives more information about the contraction than this method, such as the number
           of contraction evaluations.

        Prices are computed in each market by iterating over the :math:`\zeta`-markup contraction in
        :eq:`zeta_contraction`:

        .. math:: p^* \leftarrow c^* + \zeta^*(p^*),

        in which the markup term from :eq:`zeta` is

        .. math:: \zeta^*(p^*) = \Lambda^{-1}(p^*)[O^* \odot \Gamma(p^*)]'(p^* - c^*) - \Lambda^{-1}(p^*)

        where :math:`O^*` is the ownership matrix associated with firm changes.

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
            :meth:`ProblemResults.compute_costs`.
        prices : `array-like, optional`
            Prices at which the fixed point iteration routine will start. By default, unchanged prices, :math:`p`, are
            used as starting values. Other reasonable starting prices include the approximate equilibrium prices
            computed by :meth:`ProblemResults.compute_approximate_prices`.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem in each market. By default,
            ``Iteration('simple', {'atol': 1e-12})`` is used. Analytic Jacobians are not supported for solving this
            system.
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
        firm_ids = self.problem._coerce_optional_firm_ids(firm_ids, market_ids)
        ownership = self.problem._coerce_optional_ownership(ownership, market_ids)
        costs = self._coerce_optional_costs(costs, market_ids)
        prices = self._coerce_optional_prices(prices, market_ids)
        if iteration is None:
            iteration = Iteration('simple', {'atol': 1e-12})
        elif not isinstance(iteration, Iteration):
            raise ValueError("iteration must None or an Iteration instance.")
        elif iteration._compute_jacobian:
            raise ValueError("Analytic Jacobians are not supported for solving this system.")
        return self._combine_arrays(
            ResultsMarket.safely_compute_prices, market_ids, fixed_args=[iteration],
            market_args=[firm_ids, ownership, costs, prices]
        )

    def compute_shares(self, prices: Optional[Any] = None, market_id: Optional[Any] = None) -> Array:
        r"""Estimate shares evaluated at specified prices.

        .. note::

           To compute equilibrium shares (and prices) associated with a more complicated counterfactual, a
           :class:`Simulation` for the counterfactual can be initialized with the estimated parameters, structural
           errors, and marginal costs from these results, and then solved with :meth:`Simulation.solve`.

        Parameters
        ----------
        prices : `array-like`
            Prices at which to evaluate shares, such as equilibrium prices, :math:`p^*`, computed by
            :meth:`ProblemResults.compute_prices`. By default, unchanged prices are used.
        market_id : `object, optional`
            ID of the market in which to compute shares. By default, shares are computed in all markets and stacked.

        Returns
        -------
        `ndarray`
            Estimates of shares evaluated at the specified prices.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing shares ...")
        market_ids = self._select_market_ids(market_id)
        prices = self._coerce_optional_prices(prices, market_ids)
        return self._combine_arrays(ResultsMarket.safely_compute_shares, market_ids, market_args=[prices])

    def compute_hhi(
            self, firm_ids: Optional[Any] = None, shares: Optional[Any] = None, market_id: Optional[Any] = None) -> (
            Array):
        r"""Estimate Herfindahl-Hirschman Indices, :math:`\text{HHI}`.

        The index in market :math:`t` is

        .. math:: \text{HHI} = 10,000 \times \sum_{f=1}^{F_t} \left(\sum_{j \in \mathscr{J}_{ft}} s_j\right)^2.

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
        firm_ids = self.problem._coerce_optional_firm_ids(firm_ids, market_ids)
        shares = self._coerce_optional_shares(shares, market_ids)
        return self._combine_arrays(ResultsMarket.safely_compute_hhi, market_ids, market_args=[firm_ids, shares])

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
            :meth:`ProblemResults.compute_costs`.
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
        return self._combine_arrays(ResultsMarket.safely_compute_markups, market_ids, market_args=[prices, costs])

    def compute_profits(
            self, prices: Optional[Any] = None, shares: Optional[Any] = None, costs: Optional[Any] = None,
            market_id: Optional[Any] = None) -> Array:
        r"""Estimate population-normalized gross expected profits, :math:`\pi`.

        The profit from product :math:`j` in market :math:`t` is

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
            :meth:`ProblemResults.compute_costs`.
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
            ResultsMarket.safely_compute_profits, market_ids, market_args=[prices, shares, costs]
        )

    def compute_consumer_surpluses(self, prices: Optional[Any] = None, market_id: Optional[Any] = None) -> Array:
        r"""Estimate population-normalized consumer surpluses, :math:`\text{CS}`.

        Assuming away nonlinear income effects, the surplus in market :math:`t` is

        .. math:: \text{CS} = \sum_{i=1}^{I_t} w_{it}\text{CS}_{it},

        in which the consumer surplus for individual :math:`i` is

        .. math::

           \text{CS}_{it} =
           \log\left(1 + \sum_{j=1}^{J_t} \exp V_{jti}\right) \Big/
           \left(-\frac{\partial V_{1ti}}{\partial p_{1t}}\right),

        or with nesting parameters,

        .. math::

           \text{CS}_{it} =
           \log\left(1 + \sum_{h=1}^H \exp V_{hti}\right) \Big/
           \left(-\frac{\partial V_{1ti}}{\partial p_{1t}}\right)

        where :math:`V_{jti}` is defined in :eq:`utilities` and :math:`V_{hti}` is defined in :eq:`inclusive_value`.

        .. warning::

           :math:`\frac{\partial V_{1ti}}{\partial p_{1t}}` is the derivative of utility for the first product with
           respect to its price. The first product is chosen arbitrarily because this method assumes that there are no
           nonlinear income effects, which implies that this derivative is the same for all products. Computed consumer
           surpluses will likely be incorrect if prices are formulated in a nonlinear fashion like ``log(prices)``.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices at which utilities and price derivatives will be evaluated, such as equilibrium prices, :math:`p^*`,
            computed by :meth:`ProblemResults.compute_prices`. By default, unchanged prices are used.
        market_id : `object, optional`
            ID of the market in which to compute consumer surplus. By default, consumer surpluses are computed in all
            markets and stacked.

        Returns
        -------
        `ndarray`
            Estimated population-normalized consumer surpluses, :math:`\text{CS}`. If ``market_ids`` was not specified,
            rows are in the same order as :attr:`Problem.unique_market_ids`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing consumer surpluses with the equation that assumes away nonlinear income effects ...")
        market_ids = self._select_market_ids(market_id)
        prices = self._coerce_optional_prices(prices, market_ids)
        return self._combine_arrays(ResultsMarket.safely_compute_consumer_surplus, market_ids, market_args=[prices])
