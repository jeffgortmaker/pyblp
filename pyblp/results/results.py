"""Economy-level structuring of abstract BLP problem results."""

import abc
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

from ..configurations.iteration import Iteration
from ..markets.results_market import ResultsMarket
from ..parameters import Parameters
from ..utilities.basics import Array, StringRepresentation, output


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import ProblemEconomy  # noqa


class Results(abc.ABC, StringRepresentation):
    """Abstract results of a solved BLP problem."""

    problem: 'ProblemEconomy'
    _parameters: Parameters

    def __init__(self, problem: 'ProblemEconomy', parameters: Parameters) -> None:
        """Store the underlying problem and parameter information."""
        self.problem = problem
        self._parameters = parameters

    @abc.abstractmethod
    def _coerce_matrices(self, matrices: Any) -> Array:
        """Coerce array-like stacked arrays into a stacked array and validate it."""

    @abc.abstractmethod
    def _coerce_optional_costs(self, costs: Optional[Any]) -> Array:
        """Coerce optional array-like costs into an array and validate it."""

    @abc.abstractmethod
    def _coerce_optional_prices(self, prices: Optional[Any]) -> Array:
        """Coerce optional array-like prices into an array and validate it."""

    @abc.abstractmethod
    def _coerce_optional_shares(self, shares: Optional[Any]) -> Array:
        """Coerce optional array-like shares into an array and validate it."""

    @abc.abstractmethod
    def _combine_arrays(
            self, compute_market_results: Callable, fixed_args: Sequence = (), market_args: Sequence = ()) -> Array:
        """Combine arrays for each market, which are computed by passing fixed_args (identical for all markets) and
        market_args (arrays that need to be restricted to markets) to compute_market_results, a ResultsMarket method
        that returns the output for the market and a set of any errors encountered during computation.
        """

    def compute_aggregate_elasticities(self, factor: float = 0.1, name: str = 'prices') -> Array:
        r"""Estimate aggregate elasticities of demand, :math:`E`, with respect to a variable, :math:`x`.

        In market :math:`t`, the aggregate elasticity of demand is

        .. math:: E = \sum_{j=1}^{J_t} \frac{s_{jt}(x + \Delta x) - s_{jt}}{\Delta},

        in which :math:`\Delta` is a scalar factor and :math:`s_{jt}(x + \Delta x)` is the share of product :math:`j` in
        market :math:`t`, evaluated at the scaled values of the variable.

        Parameters
        ----------
        factor : `float, optional`
            The scalar factor, :math:`\Delta`.
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices.

        Returns
        -------
        `ndarray`
            Estimates of aggregate elasticities of demand, :math:`E`, for all markets. Rows are in the same order as
            :attr:`Problem.unique_market_ids`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output(f"Computing aggregate elasticities with respect to {name} ...")
        if not isinstance(factor, float):
            raise ValueError("factor must be a float.")
        self.problem._validate_name(name)
        return self._combine_arrays(ResultsMarket.compute_aggregate_elasticity, fixed_args=[factor, name])

    def compute_elasticities(self, name: str = 'prices') -> Array:
        r"""Estimate matrices of elasticities of demand, :math:`\varepsilon`, with respect to a variable, :math:`x`.

        For each market, the value in row :math:`j` and column :math:`k` of :math:`\varepsilon` is

        .. math:: \varepsilon_{jk} = \frac{x_k}{s_j}\frac{\partial s_j}{\partial x_k}.

        Parameters
        ----------
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices.

        Returns
        -------
        `ndarray`
            Stacked :math:`J_t \times J_t` estimated matrices of elasticities of demand, :math:`\varepsilon`, for each
            market :math:`t`. Columns for a market are in the same order as products for the market. If a market has
            fewer products than others, extra columns will contain ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output(f"Computing elasticities with respect to {name} ...")
        self.problem._validate_name(name)
        return self._combine_arrays(ResultsMarket.compute_elasticities, fixed_args=[name])

    def compute_diversion_ratios(self, name: str = 'prices') -> Array:
        r"""Estimate matrices of diversion ratios, :math:`\mathscr{D}`, with respect to a variable, :math:`x`.

        Diversion ratios to the outside good are reported on diagonals. For each market, the value in row :math:`j` and
        column :math:`k` is

        .. math:: \mathscr{D}_{jk} = -\frac{\partial s_{k(j)} / \partial x_j}{\partial s_j / \partial x_j},

        in which :math:`s_{k(j)}` is :math:`s_0 = 1 - \sum_j s_j` if :math:`j = k`, and is :math:`s_k` otherwise.

        Parameters
        ----------
        name : `str, optional`
            Name of the variable, :math:`x`. By default, :math:`x = p`, prices.

        Returns
        -------
        `ndarray`
            Stacked :math:`J_t \times J_t` estimated matrices of diversion ratios, :math:`\mathscr{D}`, for all markets.
            Columns for a market are in the same order as products for the market. If a market has fewer products than
            others, extra columns will contain ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output(f"Computing diversion ratios with respect to {name} ...")
        self.problem._validate_name(name)
        return self._combine_arrays(ResultsMarket.compute_diversion_ratios, fixed_args=[name])

    def compute_long_run_diversion_ratios(self) -> Array:
        r"""Estimate matrices of long-run diversion ratios, :math:`\bar{\mathscr{D}}`.

        Long-run diversion ratios to the outside good are reported on diagonals. For each market, the value in row
        :math:`j` and column :math:`k` is

        .. math:: \bar{\mathscr{D}}_{jk} = \frac{s_{k(-j)} - s_k}{s_j},

        in which :math:`s_{k(-j)}` is the share of product :math:`k` computed with the outside option removed from the
        choice set if :math:`j = k`, and with product :math:`j` removed otherwise.

        Parameters
        ----------

        Returns
        -------
        `ndarray`
            Stacked :math:`J_t \times J_t` estimated matrices of long-run diversion ratios, :math:`\bar{\mathscr{D}}`,
            for all markets. Columns for a market are in the same order as products for the market. If a market has
            fewer products than others, extra columns will contain ``numpy.nan``.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing long run mean diversion ratios ...")
        return self._combine_arrays(ResultsMarket.compute_long_run_diversion_ratios)

    def extract_diagonals(self, matrices: Any) -> Array:
        r"""Extract diagonals from stacked :math:`J_t \times J_t` matrices for each market :math:`t`.

        Parameters
        ----------
        matrices : `array-like`
            Stacked matrices, such as estimates of :math:`\varepsilon`, computed by
            :meth:`ProblemResults.compute_elasticities`; :math:`\mathscr{D}`, computed by
            :meth:`ProblemResults.compute_diversion_ratios`; or :math:`\bar{\mathscr{D}}`, computed by
            :meth:`ProblemResults.compute_long_run_diversion_ratios`.

        Returns
        -------
        `ndarray`
            Stacked diagonals for all markets. If the matrices are estimates of :math:`\varepsilon`, a diagonal is a
            market's own elasticities of demand; if they are estimates of :math:`\mathscr{D}` or
            :math:`\bar{\mathscr{D}}`, a diagonal is a market's diversion ratios to the outside good.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Extracting diagonals ...")
        matrices = self._coerce_matrices(matrices)
        return self._combine_arrays(ResultsMarket.extract_diagonal, market_args=[matrices])

    def extract_diagonal_means(self, matrices: Any) -> Array:
        r"""Extract means of diagonals from stacked :math:`J_t \times J_t` matrices for each market :math:`t`.

        Parameters
        ----------
        matrices : `array-like`
            Stacked matrices, such as estimates of :math:`\varepsilon`, computed by
            :meth:`ProblemResults.compute_elasticities`; :math:`\mathscr{D}`, computed by
            :meth:`ProblemResults.compute_diversion_ratios`; or :math:`\bar{\mathscr{D}}`, computed by
            :meth:`ProblemResults.compute_long_run_diversion_ratios`.

        Returns
        -------
        `ndarray`
            Stacked means of diagonals for all markets. If the matrices are estimates of :math:`\varepsilon`, the mean
            of a diagonal is a market's mean own elasticity of demand; if they are estimates of :math:`\mathscr{D}` or
            :math:`\bar{\mathscr{D}}`, the mean of a diagonal is a market's mean diversion ratio to the outside good.
            Rows are in the same order as :attr:`Problem.unique_market_ids`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Extracting diagonal means ...")
        matrices = self._coerce_matrices(matrices)
        return self._combine_arrays(ResultsMarket.extract_diagonal_mean, market_args=[matrices])

    def compute_costs(self) -> Array:
        r"""Estimate marginal costs, :math:`c`.

        Marginal costs are computed with the BLP-markup equation,

        .. math:: c = p - \eta.

        Returns
        -------
        `ndarray`
            Marginal costs, :math:`c`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing marginal costs ...")
        return self._combine_arrays(ResultsMarket.compute_costs)

    def compute_approximate_prices(
            self, firm_ids: Optional[Any] = None, ownership: Optional[Any] = None, costs: Optional[Any] = None) -> (
            Array):
        r"""Estimate approximate equilibrium prices after firm changes, :math:`p^a`, under the assumption that shares
        and their price derivatives are unaffected by such changes.

        This approximation is discussed in, for example, :ref:`references:Nevo (1997)`. Prices in each market are
        computed according to the BLP-markup equation,

        .. math:: p^a = c + \eta^a,

        in which the approximate markup term is

        .. math:: \eta^a = -\left(O^* \circ \frac{\partial s}{\partial p}\right)^{-1}s

        where :math:`O^*` is the ownership matrix associated with firm changes.

        Parameters
        ----------
        firm_ids : `array-like, optional`
            Changed firm IDs. By default, the ``firm_ids`` field of ``product_data`` in :class:`Problem` will be used.
        ownership : `array-like, optional`
            Changed ownership matrices. By default, standard ownership matrices based on ``firm_ids`` will be used
            unless the ``ownership`` field of ``product_data`` in :class:`Problem` was specified.
        costs : `array-like, optional`
            Marginal costs, :math:`c`, computed by :meth:`ProblemResults.compute_costs`. By default, marginal costs are
            computed.

        Returns
        -------
        `ndarray`
            Estimates of approximate equilibrium prices after any firm ID changes, :math:`p^a`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Solving for approximate equilibrium prices ...")
        firm_ids = self.problem._coerce_optional_firm_ids(firm_ids)
        ownership = self.problem._coerce_optional_ownership(ownership)
        costs = self._coerce_optional_costs(costs)
        return self._combine_arrays(ResultsMarket.compute_approximate_prices, market_args=[firm_ids, ownership, costs])

    def compute_prices(
            self, firm_ids: Optional[Any] = None, ownership: Optional[Any] = None, costs: Optional[Any] = None,
            prices: Optional[Any] = None, iteration: Optional[Iteration] = None) -> Array:
        r"""Estimate equilibrium prices after firm changes, :math:`p^*`.

        Prices are computed in each market by iterating over the :math:`\zeta`-markup equation from
        :ref:`references:Morrow and Skerlos (2011)`,

        .. math:: p^* \leftarrow c + \zeta^*(p^*),

        in which the markup term is

        .. math:: \zeta^*(p^*) = \Lambda^{-1}(p^*)[O^* \circ \Gamma(p^*)]'(p^* - c) - \Lambda^{-1}(p^*)

        where :math:`O^*` is the ownership matrix associated with specified firm IDs.

        Parameters
        ----------
        firm_ids : `array-like, optional`
            Changed firm IDs. By default, the ``firm_ids`` field of ``product_data`` in :class:`Problem` will be used.
        ownership : `array-like, optional`
            Changed ownership matrices. By default, standard ownership matrices based on ``firm_ids`` will be used
            unless the ``ownership`` field of ``product_data`` in :class:`Problem` was specified.
        costs : `array-like`
            Marginal costs, :math:`c`, computed by :meth:`ProblemResults.compute_costs`. By default, marginal costs are
            computed.
        prices : `array-like, optional`
            Prices at which the fixed point iteration routine will start. By default, unchanged prices, :math:`p`, are
            used as starting values. Other reasonable starting prices include :math:`p^a`, computed by
            :meth:`ProblemResults.compute_approximate_prices`.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem in each market. By default,
            ``Iteration('simple', {'atol': 1e-12})`` is used. Analytic Jacobians are not supported for this contraction
            mapping.

        Returns
        -------
        `ndarray`
            Estimates of equilibrium prices after any firm ID changes, :math:`p^*`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Solving for equilibrium prices ...")
        firm_ids = self.problem._coerce_optional_firm_ids(firm_ids)
        ownership = self.problem._coerce_optional_ownership(ownership)
        costs = self._coerce_optional_costs(costs)
        prices = self._coerce_optional_prices(prices)
        if iteration is None:
            iteration = Iteration('simple', {'atol': 1e-12})
        elif not isinstance(iteration, Iteration):
            raise ValueError("iteration must None or an Iteration instance.")
        elif iteration._compute_jacobian:
            raise ValueError("Analytic Jacobians are not supported for this contraction mapping.")
        return self._combine_arrays(
            ResultsMarket.compute_prices, fixed_args=[iteration], market_args=[firm_ids, ownership, costs, prices]
        )

    def compute_shares(self, prices: Optional[Any] = None) -> Array:
        r"""Estimate shares evaluated at specified prices.

        Parameters
        ----------
        prices : `array-like`
            Prices at which to evaluate shares, such as equilibrium prices, :math:`p^*`, computed by
            :meth:`ProblemResults.compute_prices`, or approximate equilibrium prices, :math:`p^a`, computed by
            :meth:`ProblemResults.compute_approximate_prices`. By default, unchanged prices are used.

        Returns
        -------
        `ndarray`
            Estimates of shares evaluated at the specified prices.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing shares ...")
        prices = self._coerce_optional_prices(prices)
        return self._combine_arrays(ResultsMarket.compute_shares, market_args=[prices])

    def compute_hhi(self, firm_ids: Optional[Any] = None, shares: Optional[Any] = None) -> Array:
        r"""Estimate Herfindahl-Hirschman Indices, :math:`\text{HHI}`.

        The index in market :math:`t` is

        .. math:: \text{HHI} = 10,000 \times \sum_{f=1}^{F_t} \left(\sum_{j \in \mathscr{J}_{ft}} s_j\right)^2,

        in which :math:`\mathscr{J}_{ft}` is the set of products produced by firm :math:`f` in market :math:`t`.

        Parameters
        ----------
        firm_ids : `array-like, optional`
            Changed firm IDs. By default, the ``firm_ids`` field of ``product_data`` in :class:`Problem` will be used.
        shares : `array-like, optional`
            Shares, :math:`s`, such as those computed by :meth:`ProblemResults.compute_shares`. By default, unchanged
            shares are used.

        Returns
        -------
        `ndarray`
            Estimated Herfindahl-Hirschman Indices, :math:`\text{HHI}`, for all markets. Rows are in the same order as
            :attr:`Problem.unique_market_ids`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing HHI ...")
        firm_ids = self.problem._coerce_optional_firm_ids(firm_ids)
        shares = self._coerce_optional_shares(shares)
        return self._combine_arrays(ResultsMarket.compute_hhi, market_args=[firm_ids, shares])

    def compute_markups(self, prices: Optional[Any] = None, costs: Optional[Any] = None) -> Array:
        r"""Estimate markups, :math:`\mathscr{M}`.

        The markup of product :math:`j` in market :math:`t` is

        .. math:: \mathscr{M}_{jt} = \frac{p_{jt} - c_{jt}}{p_{jt}}.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices, :math:`p`, such as equilibrium prices, :math:`p^*`, computed by
            :meth:`ProblemResults.compute_prices`, or approximate equilibrium prices, :math:`p^a`, computed by
            :meth:`ProblemResults.compute_approximate_prices`. By default, unchanged prices are used.
        costs : `array-like`
            Marginal costs, :math:`c`, computed by :meth:`ProblemResults.compute_costs`. By default, marginal costs are
            computed.

        Returns
        -------
        `ndarray`
            Estimated markups, :math:`\mathscr{M}`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing markups ...")
        prices = self._coerce_optional_prices(prices)
        costs = self._coerce_optional_costs(costs)
        return self._combine_arrays(ResultsMarket.compute_markups, market_args=[prices, costs])

    def compute_profits(
            self, prices: Optional[Any] = None, shares: Optional[Any] = None, costs: Optional[Any] = None) -> Array:
        r"""Estimate population-normalized gross expected profits, :math:`\pi`.

        The profit of product :math:`j` in market :math:`t` is

        .. math:: \pi_{jt} = p_{jt} - c_{jt}s_{jt}.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices, :math:`p`, such as equilibrium prices, :math:`p^*`, computed by
            :meth:`ProblemResults.compute_prices`, or approximate equilibrium prices, :math:`p^a`, computed by
            :meth:`ProblemResults.compute_approximate_prices`. By default, unchanged prices are used.
        shares : `array-like, optional`
            Shares, :math:`s`, such as those computed by :meth:`ProblemResults.compute_shares`. By default, unchanged
            shares are used.
        costs : `array-like`
            Marginal costs, :math:`c`, computed by :meth:`ProblemResults.compute_costs`. By default, marginal costs are
            computed.

        Returns
        -------
        `ndarray`
            Estimated population-normalized gross expected profits, :math:`\pi`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing profits ...")
        prices = self._coerce_optional_prices(prices)
        shares = self._coerce_optional_shares(shares)
        costs = self._coerce_optional_costs(costs)
        return self._combine_arrays(ResultsMarket.compute_profits, market_args=[prices, shares, costs])

    def compute_consumer_surpluses(self, prices: Optional[Any] = None) -> Array:
        r"""Estimate population-normalized consumer surpluses, :math:`\text{CS}`.

        Assuming away nonlinear income effects, the surplus in market :math:`t` is

        .. math:: \text{CS} = \sum_{i=1}^{I_t} w_i\text{CS}_i,

        in which, if there is no nesting, the consumer surplus for individual :math:`i` is

        .. math:: \text{CS}_i = \frac{\log\left(1 + \sum_{j=1}^{J_t} \exp V_{jti}\right)}{\partial U_{i1}/\partial p_1}

        where

        .. math:: V_{jti} = \delta_{jt} + \mu_{jti}.

        If there is nesting,

        .. math:: \text{CS}_i = \frac{\log\left(1 + \sum_{h=1}^H \exp V_{hti}\right)}{\partial U_{i1}/\partial p_1}

        where

        .. math:: V_{hti} = (1 - \rho_h)\log\sum_{j\in\mathscr{J}_{ht}} \exp[V_{jti} / (1 - \rho_h)].

        .. warning::

           Note that :math:`\partial U_{i1} / \partial p_1` is the derivative of utility for the first product with
           respect to its price. The first product is chosen arbitrarily because the consumer surpluses computed by this
           method assumes that there are not nonlinear income effects, which implies that this derivative is the same
           for all products. Computed consumer surpluses will be incorrect if a formulation contains, for example,
           ``log(prices)``.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices at which utilities and price derivatives will be evaluated, such as equilibrium prices, :math:`p^*`,
            computed by :meth:`ProblemResults.compute_prices`, or approximate equilibrium prices, :math:`p^a`, computed
            by :meth:`ProblemResults.compute_approximate_prices`. By default, unchanged prices are used.

        Returns
        -------
        `ndarray`
            Estimated population-normalized consumer surpluses, :math:`\text{CS}`, for all markets. Rows are in the same
            order as :attr:`Problem.unique_market_ids`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        output("Computing consumer surpluses with the equation that assumes away nonlinear income effects ...")
        prices = self._coerce_optional_prices(prices)
        return self._combine_arrays(ResultsMarket.compute_consumer_surplus, market_args=[prices])
