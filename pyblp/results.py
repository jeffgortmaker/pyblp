"""Structuring of BLP results and computation of post-estimation outputs."""

import collections
import itertools
import time
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, TYPE_CHECKING, Tuple, Union

import numpy as np
import scipy.linalg

from . import exceptions, options
from .configurations.formulation import ColumnFormulation, Formulation
from .configurations.integration import Integration
from .configurations.iteration import Iteration
from .economy import Market
from .parameters import LinearParameters, NonlinearParameters, PiParameter, RhoParameter, SigmaParameter
from .utilities.algebra import multiply_matrix_and_tensor
from .utilities.basics import (
    Array, Bounds, Error, Mapping, RecArray, StringRepresentation, TableFormatter, format_number, format_seconds,
    generate_items, output, output_progress, update_matrices
)
from .utilities.statistics import IV, compute_gmm_parameter_covariances, compute_gmm_weights


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .problem import _Problem, Problem, Progress  # noqa
    from .simulation import Simulation  # noqa


class _ProblemResults(StringRepresentation):
    """Results of a solved BLP problem."""

    problem: '_Problem'

    def __init__(self, problem: '_Problem') -> None:
        """Store the underlying problem."""
        self.problem = problem

    def _coerce_matrices(self, matrices: Any) -> Array:
        """Coerce array-like stacked arrays into a stacked array and validate it."""
        raise NotImplementedError

    def _coerce_optional_costs(self, costs: Optional[Any]) -> Array:
        """Coerce optional array-like costs into an array and validate it."""
        raise NotImplementedError

    def _coerce_optional_prices(self, prices: Optional[Any]) -> Array:
        """Coerce optional array-like prices into an array and validate it."""
        raise NotImplementedError

    def _coerce_optional_shares(self, shares: Optional[Any]) -> Array:
        """Coerce optional array-like shares into an array and validate it."""
        raise NotImplementedError

    def _combine_arrays(self, compute_market_results: Callable, fixed_args: Sequence, market_args: Sequence) -> Array:
        """Combine arrays for each market, which are computed by passing fixed_args (identical for all markets) and
        market_args (arrays that need to be restricted to markets) to compute_market_results, a ResultsMarket method
        that returns the output for the market and a set of any errors encountered during computation.
        """
        raise NotImplementedError

    def _compute_true_X1(self, data_override: Optional[Mapping] = None) -> Array:
        """Compute X1 without any absorbed demand-side fixed effects."""
        if self.problem.ED == 0 and not data_override:
            return self.problem.products.X1
        ones = np.ones((self.problem.N, 1), options.dtype)
        columns = (ones * f.evaluate(self.problem.products, data_override) for f in self.problem._X1_formulations)
        return np.column_stack(columns)

    def _compute_true_X3(self, data_override: Optional[Mapping] = None) -> Array:
        """Compute X3 without any absorbed supply-side fixed effects."""
        if self.problem.ES == 0 and not data_override:
            return self.problem.products.X3
        ones = np.ones((self.problem.N, 1), options.dtype)
        columns = (ones * f.evaluate(self.problem.products, data_override) for f in self.problem._X3_formulations)
        return np.column_stack(columns)

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

        """
        output(f"Computing aggregate elasticities with respect to {name} ...")
        if not isinstance(factor, float):
            raise ValueError("factor must be a float.")
        self.problem._validate_name(name)
        return self._combine_arrays(ResultsMarket.compute_aggregate_elasticity, [factor, name], [])

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

        """
        output(f"Computing elasticities with respect to {name} ...")
        self.problem._validate_name(name)
        return self._combine_arrays(ResultsMarket.compute_elasticities, [name], [])

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

        """
        output(f"Computing diversion ratios with respect to {name} ...")
        self.problem._validate_name(name)
        return self._combine_arrays(ResultsMarket.compute_diversion_ratios, [name], [])

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

        """
        output("Computing long run mean diversion ratios ...")
        return self._combine_arrays(ResultsMarket.compute_long_run_diversion_ratios, [], [])

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

        """
        output("Extracting diagonals ...")
        matrices = self._coerce_matrices(matrices)
        return self._combine_arrays(ResultsMarket.extract_diagonal, [], [matrices])

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

        """
        output("Extracting diagonal means ...")
        matrices = self._coerce_matrices(matrices)
        return self._combine_arrays(ResultsMarket.extract_diagonal_mean, [], [matrices])

    def compute_costs(self) -> Array:
        r"""Estimate marginal costs, :math:`c`.

        Marginal costs are computed with the BLP-markup equation,

        .. math:: c = p - \eta.

        Parameters
        ----------

        Returns
        -------
        `ndarray`
            Marginal costs, :math:`c`.

        """
        output("Computing marginal costs ...")
        return self._combine_arrays(ResultsMarket.compute_costs, [], [])

    def compute_approximate_prices(self, firms_index: int = 1, costs: Optional[Any] = None) -> Array:
        r"""Estimate approximate equilibrium prices after firm ID changes, :math:`p^a`, under the assumption that
        shares and their price derivatives are unaffected by such changes.

        This approximation is discussed in, for example, :ref:`Nevo (1997) <n97>`. Prices in each market are computed
        according to the BLP-markup equation,

        .. math:: p^a = c + \eta^a,

        in which the approximate markup term is

        .. math:: \eta^a = -\left(O^* \circ \frac{\partial s}{\partial p}\right)^{-1}s

        where :math:`O^*` is the ownership matrix associated with specified firm IDs.

        Parameters
        ----------
        firms_index : `int, optional`
            Column index of the firm IDs in the `firm_ids` field of `product_data` in :class:`Problem`. If an
            `ownership` field was specified, the corresponding stack of ownership matrices will be used.
        costs : `array-like, optional`
            Marginal costs, :math:`c`, computed by :meth:`ProblemResults.compute_costs`. By default, marginal costs are
            computed.

        Returns
        -------
        `ndarray`
            Estimates of approximate equilibrium prices after any firm ID changes, :math:`p^a`.

        """
        output("Solving for approximate equilibrium prices ...")
        self.problem._validate_firms_index(firms_index)
        costs = self._coerce_optional_costs(costs)
        return self._combine_arrays(ResultsMarket.compute_approximate_prices, [firms_index], [costs])

    def compute_prices(
            self, iteration: Optional[Iteration] = None, firms_index: int = 1, prices: Optional[Any] = None,
            costs: Optional[Any] = None) -> Array:
        r"""Estimate equilibrium prices after firm ID changes, :math:`p^*`.

        Prices are computed in each market by iterating over the :math:`\zeta`-markup equation from
        :ref:`Morrow and Skerlos (2011) <ms11>`,

        .. math:: p^* \leftarrow c + \zeta^*(p^*),

        in which the markup term is

        .. math:: \zeta^*(p^*) = \Lambda^{-1}(p^*)[O^* \circ \Gamma(p^*)]'(p^* - c) - \Lambda^{-1}(p^*)

        where :math:`O^*` is the ownership matrix associated with specified firm IDs.

        Parameters
        ----------
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem in each market. By default,
            ``Iteration('simple', {'tol': 1e-12})`` is used.
        firms_index : `int, optional`
            Column index of the firm IDs in the `firm_ids` field of `product_data` in :class:`Problem`. If an
            `ownership` field was specified, the corresponding stack of ownership matrices will be used.
        prices : `array-like, optional`
            Prices at which the fixed point iteration routine will start. By default, unchanged prices, :math:`p`, are
            used as starting values. Other reasonable starting prices include :math:`p^a`, computed by
            :meth:`ProblemResults.compute_approximate_prices`.
        costs : `array-like`
            Marginal costs, :math:`c`, computed by :meth:`ProblemResults.compute_costs`. By default, marginal costs are
            computed.

        Returns
        -------
        `ndarray`
            Estimates of equilibrium prices after any firm ID changes, :math:`p^*`.

        """
        output("Solving for equilibrium prices ...")
        if iteration is None:
            iteration = Iteration('simple', {'tol': 1e-12})
        elif not isinstance(iteration, Iteration):
            raise ValueError("iteration must None or an Iteration instance.")
        self.problem._validate_firms_index(firms_index)
        prices = self._coerce_optional_prices(prices)
        costs = self._coerce_optional_costs(costs)
        return self._combine_arrays(ResultsMarket.compute_prices, [iteration, firms_index], [prices, costs])

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

        """
        output("Computing shares ...")
        prices = self._coerce_optional_prices(prices)
        return self._combine_arrays(ResultsMarket.compute_shares, [], [prices])

    def compute_hhi(self, firms_index: int = 0, shares: Optional[Any] = None) -> Array:
        r"""Estimate Herfindahl-Hirschman Indices, :math:`\text{HHI}`.

        The index in market :math:`t` is

        .. math:: \text{HHI} = 10,000 \times \sum_{f=1}^{F_t} \left(\sum_{j \in \mathscr{J}_{ft}} s_j\right)^2,

        in which :math:`\mathscr{J}_{ft}` is the set of products produced by firm :math:`f` in market :math:`t`.

        Parameters
        ----------
        firms_index : `int, optional`
            Column index of the firm IDs in the `firm_ids` field of `product_data` in :class:`Problem`. By default,
            unchanged firm IDs are used.
        shares : `array-like, optional`
            Shares, :math:`s`, such as those computed by :meth:`ProblemResults.compute_shares`. By default, unchanged
            shares are used.

        Returns
        -------
        `ndarray`
            Estimated Herfindahl-Hirschman Indices, :math:`\text{HHI}`, for all markets. Rows are in the same order as
            :attr:`Problem.unique_market_ids`.

        """
        output("Computing HHI ...")
        self.problem._validate_firms_index(firms_index)
        shares = self._coerce_optional_shares(shares)
        return self._combine_arrays(ResultsMarket.compute_hhi, [firms_index], [shares])

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

        """
        output("Computing markups ...")
        prices = self._coerce_optional_prices(prices)
        costs = self._coerce_optional_costs(costs)
        return self._combine_arrays(ResultsMarket.compute_markups, [], [prices, costs])

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

        """
        output("Computing profits ...")
        prices = self._coerce_optional_prices(prices)
        shares = self._coerce_optional_shares(shares)
        costs = self._coerce_optional_costs(costs)
        return self._combine_arrays(ResultsMarket.compute_profits, [], [prices, shares, costs])

    def compute_consumer_surpluses(self, prices: Optional[Any] = None) -> Array:
        r"""Estimate population-normalized consumer surpluses, :math:`\text{CS}`.

        Assuming away nonlinear income effects, the surplus in market :math:`t` is

        .. math:: \text{CS} = \sum_{i=1}^{I_t} w_i\text{CS}_i,

        in which, if there is no nesting, the consumer surplus for individual :math:`i` is

        .. math:: \text{CS}_i = \frac{\log\left(1 + \sum_{j=1}^{J_t} \exp V_{jti}\right)}{\alpha + \alpha_i}

        where

        .. math:: V_{jti} = \delta_{jt} + \mu_{jti}.

        If there is nesting,

        .. math:: \text{CS}_i = \frac{\log\left(1 + \sum_{h=1}^H \exp V_{hti}\right)}{\alpha + \alpha_i}

        where

        .. math:: V_{hti} = (1 - \rho_h)\log\sum_{j\in\mathscr{J}_{ht}} \exp[V_{jti} / (1 - \rho_h)].

        .. warning::

           The consumer surpluses computed by this method are only correct when there are not nonlinear income effects.
           For example, computed consumer surpluses will be incorrect if a formulation contains ``log(prices)``.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices at which utilities, :math:`u`, and price derivatives, :math:`\alpha` and :math:`\alpha_i`, will be
            evaluated, such as equilibrium prices, :math:`p^*`, computed by :meth:`ProblemResults.compute_prices`, or
            approximate equilibrium prices, :math:`p^a`, computed by :meth:`ProblemResults.compute_approximate_prices`.
            By default, unchanged prices are used.

        Returns
        -------
        `ndarray`
            Estimated population-normalized consumer surpluses, :math:`\text{CS}`, for all markets. Rows are in the same
            order as :attr:`Problem.unique_market_ids`.

        """
        output("Computing consumer surpluses with the equation that assumes away nonlinear income effects ...")
        prices = self._coerce_optional_prices(prices)
        return self._combine_arrays(ResultsMarket.compute_consumer_surplus, [], [prices])


class ProblemResults(_ProblemResults):
    r"""Results of a solved BLP problem.

    Many results are class attributes. Other post-estimation outputs be computed by calling class methods.

    .. note::

       All methods in this class support :func:`parallel` processing. If multiprocessing is used, market-by-market
       computation of each post-estimation output will be distributed among the processes.

    Attributes
    ----------
    problem : `Problem`
        :class:`Problem` that created these results.
    last_results : `ProblemResults`
        :class:`ProblemResults` from the last GMM step.
    step : `int`
        GMM step that created these results.
    optimization_time : `float`
        Number of seconds it took the optimization routine to finish.
    cumulative_optimization_time : `float`
        Sum of :attr:`ProblemResults.optimization_time` for this step and all prior steps.
    total_time : `float`
        Sum of :attr:`ProblemResults.optimization_time` and the number of seconds it took to set up the GMM step and
        compute results after optimization had finished.
    cumulative_total_time : `float`
        Sum of :attr:`ProblemResults.total_time` for this step and all prior steps.
    optimization_iterations : `int`
        Number of major iterations completed by the optimization routine.
    cumulative_optimization_iterations : `int`
        Sum of :attr:`ProblemResults.optimization_iterations` for this step and all prior steps.
    objective_evaluations : `int`
        Number of GMM objective evaluations.
    cumulative_objective_evaluations : `int`
        Sum of :attr:`ProblemResults.objective_evaluations` for this step and all prior steps.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute :math:`\delta(\hat{\theta})` in
        each market during each objective evaluation. Rows are in the same order as
        :attr:`Problem.unique_market_ids` and column indices correspond to objective evaluations.
    cumulative_fp_iterations : `ndarray`
        Concatenation of :attr:`ProblemResults.fp_iterations` for this step and all prior steps.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute :math:`\delta(\hat{\theta})` was evaluated in each market during
        each objective evaluation. Rows are in the same order as :attr:`Problem.unique_market_ids` and column
        indices correspond to objective evaluations.
    cumulative_contraction_evaluations : `ndarray`
        Concatenation of :attr:`ProblemResults.contraction_evaluations` for this step and all prior steps.
    parameters : `ndarray`
        Stacked parameters: :math:`\hat{\theta}`, :math:`\hat{\beta}`, and :math:`\hat{\gamma}`, in that order.
    parameter_covariances : `ndarray`
        Estimated covariance matrix of the stacked parameters, from which standard errors are extracted.
    theta : `ndarray`
        Estimated unfixed nonlinear parameters, :math:`\hat{\theta}`.
    sigma : `ndarray`
        Estimated Cholesky decomposition of the covariance matrix that measures agents' random taste distribution,
        :math:`\hat{\Sigma}`.
    pi : `ndarray`
        Estimated parameters that measures how agent tastes vary with demographics, :math:`\hat{\Pi}`.
    rho : `ndarray`
        Estimated parameters that measure within nesting group correlations, :math:`\hat{\rho}`.
    beta : `ndarray`
        Estimated demand-side linear parameters, :math:`\hat{\beta}`.
    gamma : `ndarray`
        Estimated supply-side linear parameters, :math:`\hat{\gamma}`.
    sigma_se : `ndarray`
        Estimated standard errors for unknown :math:`\hat{\Sigma}` elements in :math:`\hat{\theta}`.
    pi_se : `ndarray`
        Estimated standard errors for unknown :math:`\hat{\Pi}` elements in :math:`\hat{\theta}`.
    rho_se : `ndarray`
        Estimated standard errors for unknown :math:`\hat{\rho}` elements in :math:`\hat{\theta}`.
    beta_se : `ndarray`
        Estimated standard errors for :math:`\hat{\beta}`.
    gamma_se : `ndarray`
        Estimated standard errors for :math:`\hat{\gamma}`.
    sigma_bounds : `tuple`
        Bounds for :math:`\Sigma` that were used during optimization, which are of the form ``(lb, ub)``.
    pi_bounds : `tuple`
        Bounds for :math:`\Pi` that were used during optimization, which are of the form ``(lb, ub)``.
    rho_bounds : `tuple`
        Bounds for :math:`\rho` that were used during optimization, which are of the form ``(lb, ub)``.
    delta : `ndarray`
        Estimated mean utility, :math:`\delta(\hat{\theta})`, which may have been residualized to absorb any demand-side
        fixed effects.
    true_delta : `ndarray`
        Estimated mean utility, :math:`\delta(\hat{\theta})`.
    tilde_costs : `ndarray`
        Estimated transformed marginal costs, :math:`\tilde{c}(\hat{\theta})`, which may have been residualized to
        absorb any demand-side fixed effects. Transformed marginal costs are simply :math:`\tilde{c} = c`, marginal
        costs, under a linear cost specification, and are :math:`\tilde{c} = \log c` under a log-linear specification.
        If `costs_bounds` were specified in :meth:`Problem.solve`, :math:`c` may have been clipped.
    true_tilde_costs : `ndarray`
        Estimated transformed marginal costs, :math:`\tilde{c}(\hat{\theta})`. Transformed marginal costs are simply
        :math:`\tilde{c} = c`, marginal costs, under a linear cost specification, and are :math:`\tilde{c} = \log c`
        under a log-linear specification. If `costs_bounds` were specified in :meth:`Problem.solve`, :math:`c` may have
        been clipped.
    clipped_costs : `ndarray`
        Vector of booleans indicating whether the associated marginal costs were clipped. All elements will be ``False``
        if `costs_bounds` in :meth:`Problem.solve` was not specified.
    xi : `ndarray`
        Estimated unobserved demand-side product characteristics, :math:`\xi(\hat{\theta})`, or equivalently, the
        demand-side structural error term, which includes the contribution of any absorbed demand-side fixed effects.
    true_xi : `ndarray
        Estimated unobserved demand-side product characteristics, :math:`\xi(\hat{\theta})`.
    omega : `ndarray`
        Estimated unobserved supply-side product characteristics, :math:`\omega(\hat{\theta})`, or equivalently, the
        supply-side structural error term, which includes the contribution of any absorbed supply-side fixed effects.
    true_omega : `ndarray`
        Estimated unobserved supply-side product characteristics, :math:`\omega(\hat{\theta})`, or equivalently, the
        supply-side structural error term.
    objective : `float`
        GMM objective value.
    xi_by_theta_jacobian : `ndarray`
        Estimated :math:`\partial\xi / \partial\theta = \partial\delta / \partial\theta`.
    omega_by_theta_jacobian : `ndarray`
        Estimated :math:`\partial\omega / \partial\theta = \partial\tilde{c} / \partial\theta`.
    omega_by_beta_jacobian : `ndarray`
        Estimated :math:`\partial\omega / \partial\beta = \partial\tilde{c} / \partial\beta`.
    gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to :math:`\theta`. This is still computed once at the end
        of an optimization routine that was configured to not use analytic gradients.
    gradient_norm : `ndarray`
        Infinity norm of :attr:`ProblemResults.gradient`.
    sigma_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to unknown :math:`\Sigma` elements in :math:`\theta`.
    pi_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to unknown :math:`\Pi` elements in :math:`\theta`.
    rho_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to unknown :math:`\rho` elements in :math:`\theta`.
    WD : `ndarray`
        Demand-side weighting matrix, :math:`W_D`, used to compute these results.
    WS : `ndarray`
        Supply-side weighting matrix, :math:`W_S`, used to compute these results.
    updated_WD : `ndarray`
        Updated demand-side weighting matrix.
    updated_WS : `ndarray`
        Updated supply-side weighting matrix.

    Examples
    --------
    For examples of how to use class methods, refer to the :doc:`Examples </examples>` section.

    """

    last_results: Optional['ProblemResults']
    step: int
    optimization_time: float
    cumulative_optimization_time: float
    total_time: float
    cumulative_total_time: float
    optimization_iterations: int
    cumulative_optimization_iterations: int
    objective_evaluations: int
    cumulative_objective_evaluations: int
    fp_iterations: Array
    cumulative_fp_iterations: Array
    contraction_evaluations: Array
    cumulative_contraction_evaluations: Array
    parameters: Array
    parameter_covariances: Array
    theta: Array
    sigma: Array
    pi: Array
    rho: Array
    beta: Array
    gamma: Array
    sigma_se: Array
    pi_se: Array
    rho_se: Array
    beta_se: Array
    gamma_se: Array
    sigma_bounds: tuple
    pi_bounds: tuple
    rho_bounds: tuple
    delta: Array
    true_delta: Array
    tilde_costs: Array
    true_tilde_costs: Array
    clipped_costs: Array
    xi: Array
    true_xi: Array
    omega: Array
    true_omega: Array
    objective: Array
    xi_by_theta_jacobian: Array
    omega_by_theta_jacobian: Array
    omega_by_beta_jacobian: Array
    gradient: Array
    gradient_norm: Array
    sigma_gradient: Array
    pi_gradient: Array
    rho_gradient: Array
    WD: Array
    WS: Array
    updated_WD: Array
    updated_WS: Array
    _costs_type: str
    _se_type: str
    _errors: List[Error]
    _linear_parameters: LinearParameters
    _nonlinear_parameters: NonlinearParameters

    def __init__(
            self, progress: 'Progress', last_results: Optional['ProblemResults'], step_start_time: float,
            optimization_start_time: float, optimization_end_time: float, iterations: int, evaluations: int,
            iteration_mappings: Sequence[Dict[Hashable, int]], evaluation_mappings: Sequence[Dict[Hashable, int]],
            costs_type: str, costs_bounds: Bounds, center_moments: bool, W_type: str, se_type: str) -> None:
        """Compute cumulative progress statistics, update weighting matrices, and estimate standard errors."""

        # initialize values from the progress structure
        super().__init__(progress.problem)
        self._errors = progress.errors
        self.problem = progress.problem
        self.WD = progress.WD
        self.WS = progress.WS
        self.theta = progress.theta
        self.true_delta = progress.true_delta
        self.true_tilde_costs = progress.true_tilde_costs
        self.xi_by_theta_jacobian = progress.xi_jacobian
        self.omega_by_theta_jacobian = progress.omega_jacobian
        self.delta = progress.delta
        self.tilde_costs = progress.tilde_costs
        self.true_xi = progress.true_xi
        self.true_omega = progress.true_omega
        self.beta = progress.beta
        self.gamma = progress.gamma
        self.objective = progress.objective
        self.gradient = progress.gradient
        self.gradient_norm = progress.gradient_norm

        # store information about cost bounds
        self._costs_bounds = costs_bounds
        self.clipped_costs = progress.clipped_costs
        assert self.clipped_costs is not None

        # initialize counts and times
        self.step = 1
        self.total_time = self.cumulative_total_time = time.time() - step_start_time
        self.optimization_time = self.cumulative_optimization_time = optimization_end_time - optimization_start_time
        self.optimization_iterations = self.cumulative_optimization_iterations = iterations
        self.objective_evaluations = self.cumulative_objective_evaluations = evaluations
        self.fp_iterations = self.cumulative_fp_iterations = np.array(
            [[m[t] if m else 0 for m in iteration_mappings] for t in self.problem.unique_market_ids]
        )
        self.contraction_evaluations = self.cumulative_contraction_evaluations = np.array(
            [[m[t] if m else 0 for m in evaluation_mappings] for t in self.problem.unique_market_ids]
        )

        # initialize last results and add to cumulative values
        self.last_results = last_results
        if last_results is not None:
            self.step += last_results.step
            self.cumulative_total_time += last_results.cumulative_total_time
            self.cumulative_optimization_time += last_results.cumulative_optimization_time
            self.cumulative_optimization_iterations += last_results.cumulative_optimization_iterations
            self.cumulative_objective_evaluations += last_results.cumulative_objective_evaluations
            self.cumulative_fp_iterations = np.c_[
                last_results.cumulative_fp_iterations, self.cumulative_fp_iterations
            ]
            self.cumulative_contraction_evaluations = np.c_[
                last_results.cumulative_contraction_evaluations, self.cumulative_contraction_evaluations
            ]

        # store parameter information
        self.parameters = np.r_[np.c_[self.theta], self.beta, self.gamma]
        self._linear_parameters = LinearParameters(self.problem, self.beta, self.gamma)
        self._nonlinear_parameters = progress.nonlinear_parameters
        self.sigma_bounds = self._nonlinear_parameters.sigma_bounds
        self.pi_bounds = self._nonlinear_parameters.pi_bounds
        self.rho_bounds = self._nonlinear_parameters.rho_bounds

        # expand the nonlinear parameters and their gradient
        self.sigma, self.pi, self.rho = self._nonlinear_parameters.expand(self.theta)
        self.sigma_gradient, self.pi_gradient, self.rho_gradient = self._nonlinear_parameters.expand(
            self.gradient, nullify=True
        )

        # compute a version of xi that includes the contribution of any demand-side fixed effects
        self.xi = self.true_xi
        if self.problem.ED > 0:
            self.xi = self.true_delta - self._compute_true_X1() @ self.beta

        # compute a version of omega that includes the contribution of any supply-side fixed effects
        self.omega = self.true_omega
        if self.problem.ES > 0:
            self.omega = self.true_tilde_costs - self._compute_true_X3() @ self.gamma

        # update the weighting matrices
        self.updated_WD, WD_errors = compute_gmm_weights(
            self.true_xi, self.problem.products.ZD, W_type, center_moments, self.problem.products.clustering_ids
        )
        self.updated_WS, WS_errors = compute_gmm_weights(
            self.true_omega, self.problem.products.ZS, W_type, center_moments, self.problem.products.clustering_ids
        )
        self._errors.extend(WD_errors + WS_errors)

        # compute the Jacobian of omega with respect to beta, which will be used for computing standard errors
        self.omega_by_beta_jacobian = np.full((self.problem.N, self.problem.K1), np.nan, options.dtype)
        if self.problem.K3 > 0:
            # define a factory for computing the Jacobian of omega with respect to beta in markets
            def market_factory(s: Hashable) -> Tuple[ResultsMarket, Array, str]:
                """Build a market along with arguments used to compute the Jacobian."""
                market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, self.true_delta)
                true_tilde_costs_s = self.true_tilde_costs[self.problem._product_market_indices[s]]
                return market_s, true_tilde_costs_s, costs_type

            # compute the Jacobian market-by-market
            generator = generate_items(
                self.problem.unique_market_ids, market_factory, ResultsMarket.compute_omega_by_beta_jacobian
            )
            for t, (omega_by_beta_jacobian_t, errors_t) in generator:
                self.omega_by_beta_jacobian[self.problem._product_market_indices[t]] = omega_by_beta_jacobian_t
                self._errors.extend(errors_t)

            # the Jacobian should be zero for any clipped marginal costs
            self.omega_by_beta_jacobian[self.clipped_costs.flat] = 0

        # stack errors, weights, instruments, Jacobian of the errors with respect to parameters, and clustering IDs
        if self.problem.K3 == 0:
            u = self.true_xi
            Z = self.problem.products.ZD
            W = self.WD
            jacobian = np.c_[self.xi_by_theta_jacobian, -self.problem.products.X1]
            stacked_clustering_ids = self.problem.products.clustering_ids
        else:
            u = np.r_[self.true_xi, self.true_omega]
            Z = scipy.linalg.block_diag(self.problem.products.ZD, self.problem.products.ZS)
            W = scipy.linalg.block_diag(self.WD, self.WS)
            jacobian = np.r_[
                np.c_[self.xi_by_theta_jacobian, -self.problem.products.X1, np.zeros_like(self.problem.products.X3)],
                np.c_[self.omega_by_theta_jacobian, self.omega_by_beta_jacobian, -self.problem.products.X3]
            ]
            stacked_clustering_ids = np.r_[self.problem.products.clustering_ids, self.problem.products.clustering_ids]

        # compute parameter covariances
        self.parameter_covariances, covariance_errors = compute_gmm_parameter_covariances(
            u, Z, W, jacobian, se_type, self.step, stacked_clustering_ids
        )
        self._errors.extend(covariance_errors)

        # extract standard errors
        with np.errstate(invalid='ignore'):
            se = np.sqrt(np.c_[self.parameter_covariances.diagonal()] / self.problem.N)
        self.sigma_se, self.pi_se, self.rho_se = self._nonlinear_parameters.expand(
            se[:self._nonlinear_parameters.P], nullify=True
        )
        self.beta_se = se[self._nonlinear_parameters.P:self._nonlinear_parameters.P + self.problem.K1]
        self.gamma_se = se[self._nonlinear_parameters.P + self.problem.K1:]
        if np.isnan(se).any():
            self._errors.append(exceptions.InvalidParameterCovariancesError())

        # store types that are used in other methods
        self._costs_type = costs_type
        self._se_type = se_type

    def __str__(self) -> str:
        """Format problem results as a string."""

        # construct a section containing summary information
        header = [
            ("Cumulative", "Total Time"), ("GMM", "Step"), ("Optimization", "Iterations"),
            ("Objective", "Evaluations"), ("Total Fixed Point", "Iterations"), ("Total Contraction", "Evaluations"),
            ("Objective", "Value"), ("Gradient", "Infinity Norm"),
        ]
        values = [
            format_seconds(self.cumulative_total_time),
            self.step,
            self.optimization_iterations,
            self.objective_evaluations,
            self.fp_iterations.sum(),
            self.contraction_evaluations.sum(),
            format_number(float(self.objective)),
            format_number(float(self.gradient_norm))
        ]
        if np.isfinite(self._costs_bounds).any():
            header.append(("Clipped", "Marginal Costs"))
            values.append(self.clipped_costs.sum())
        widths = [max(len(k1), len(k2), options.digits + 6 if i > 5 else 0) for i, (k1, k2) in enumerate(header)]
        formatter = TableFormatter(widths)
        sections = [[
            "Problem Results Summary:",
            formatter.line(),
            formatter([k[0] for k in header]),
            formatter([k[1] for k in header], underline=True),
            formatter(values),
            formatter.line()
        ]]

        # construct a standard error description
        if self._se_type == 'unadjusted':
            se_description = "Unadjusted SEs"
        elif self._se_type == 'robust':
            se_description = "Robust SEs"
        else:
            assert self._se_type == 'clustered'
            se_description = f'Robust SEs Adjusted for {np.unique(self.problem.products.clustering_ids).size} Clusters'

        # construct a section containing linear estimates
        sections.append([
            f"Linear Estimates ({se_description} in Parentheses):",
            self._linear_parameters.format_estimates(self.beta, self.gamma, self.beta_se, self.gamma_se)
        ])

        # construct a section containing nonlinear estimates
        if self.problem.K2 > 0 or self.problem.H > 0:
            sections.append([
                f"Nonlinear Estimates ({se_description} in Parentheses):",
                self._nonlinear_parameters.format_estimates(
                    self.sigma, self.pi, self.rho, self.sigma_se, self.pi_se, self.rho_se
                )
            ])

        # combine the sections into one string
        return "\n\n".join("\n".join(s) for s in sections)

    def bootstrap(
            self, draws: int = 1000, seed: Optional[int] = None, iteration: Optional[Iteration] = None) -> (
            'BootstrappedProblemResults'):
        """Use a parametric bootstrap to create a distribution of results.

        The constructed :class:`BootstrappedProblemResults` can be used just like :class:`ProblemResults` to compute
        various post-estimation outputs. The only difference is that :class:`BootstrappedProblemResults` methods return
        arrays with an extra first dimension, along which bootstrapped results are stacked. These stacked results can
        be used to construct, for example, confidence intervals for post-estimation outputs.

        For each bootstrap draw, parameters are drawn from the estimated multivariate normal distribution of all
        parameters defined by :attr:`ProblemResults.parameters` and :attr:`ProblemResults.parameter_covariances`. Note
        that any bounds configured during the optimization routine will be used to bound parameter draws. These
        parameters are used to compute the implied mean utility, :math:`\delta`, and shares, :math:`s`. If a supply side
        was estimated, the implied marginal costs, :math:`c`, and prices, :math:`p`, are computed as well. Specifically,
        if a supply side was estimated, equilibrium prices and shares are computed by iterating over the
        :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>`.

        .. note::

           By default, the bootstrapping procedure can use a lot of memory. This is because it stores in memory all
           bootstrapped results (for all `draws`) at the same time. To reduce the memory footprint of the procedure,
           call this method in a loop with `draws` set to ``1``. In each iteration of the loop, compute the desired
           post-estimation output with the proper method of the returned :class:`BootstrappedProblemResults` class and
           store these outputs.

        Parameters
        ----------
        draws : `int, optional`
            The number of draws that will be taken from the joint distribution of the parameters. The default is
            ``1000``.
        seed : `int, optional`
            Passed to :class:`numpy.random.RandomState` to seed the random number generator before any draws are taken.
            By default, a seed is not passed to the random number generator.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration used to compute bootstrapped prices by iterating over the
            :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>`. By default, if a supply side was
            estimated, this is ``Iteration('simple', {'tol': 1e-12})``. It is not used if a supply side was not
            estimated.

        Returns
        -------
        `BootstrappedProblemResults`
            Computed :class:`BootstrappedProblemResults`.

        """
        errors: List[Error] = []

        # keep track of long it takes to bootstrap results
        output("Bootstrapping results ...")
        start_time = time.time()

        # validate the number of draws
        if not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive int.")

        # validate the iteration configuration
        if self.problem.K3 == 0:
            iteration = None
        elif iteration is None:
            iteration = Iteration('simple', {'tol': 1e-12})
        elif not isinstance(iteration, Iteration):
            raise TypeError("iteration must be None or an iteration instance.")

        # draw from the asymptotic distribution implied by the estimated parameters
        state = np.random.RandomState(seed)
        bootstrapped_parameters = np.atleast_3d(state.multivariate_normal(
            self.parameters.flatten(), self.parameter_covariances, draws
        ))

        # extract the parameters
        split_indices = [self._nonlinear_parameters.P, self._nonlinear_parameters.P + self.problem.K1]
        bootstrapped_theta, bootstrapped_beta, bootstrapped_gamma = np.split(
            bootstrapped_parameters, split_indices, axis=1
        )

        # extract bootstrapped nonlinear parameters
        bootstrapped_sigma = np.repeat(np.zeros_like(self.sigma[None]), draws, axis=0)
        bootstrapped_pi = np.repeat(np.zeros_like(self.pi[None]), draws, axis=0)
        bootstrapped_rho = np.repeat(np.zeros_like(self.rho[None]), draws, axis=0)
        for d in range(draws):
            bootstrapped_sigma[d], bootstrapped_pi[d], bootstrapped_rho[d] = self._nonlinear_parameters.expand(
                bootstrapped_theta[d]
            )
            bootstrapped_sigma[d] = np.clip(bootstrapped_sigma[d], *self.sigma_bounds)
            bootstrapped_pi[d] = np.clip(bootstrapped_pi[d], *self.pi_bounds)
            bootstrapped_rho[d] = np.clip(bootstrapped_rho[d], *self.rho_bounds)

        # compute bootstrapped prices, shares, delta and marginal costs
        iteration_mappings: List[Dict[Hashable, int]] = []
        evaluation_mappings: List[Dict[Hashable, int]] = []
        bootstrapped_prices = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_shares = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_delta = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_costs = np.zeros((draws, self.problem.N, int(self.problem.K3 > 0)), options.dtype)
        for d in output_progress(range(draws), draws, start_time):
            prices_d, shares_d, delta_d, costs_d, iterations_d, evaluations_d, errors_d = self._compute_bootstrap(
                iteration, bootstrapped_sigma[d], bootstrapped_pi[d], bootstrapped_rho[d], bootstrapped_beta[d],
                bootstrapped_gamma[d]
            )
            bootstrapped_prices[d] = prices_d
            bootstrapped_shares[d] = shares_d
            bootstrapped_delta[d] = delta_d
            bootstrapped_costs[d] = costs_d
            iteration_mappings.append(iterations_d)
            evaluation_mappings.append(evaluations_d)
            errors.extend(errors_d)

        # structure the results
        results = BootstrappedProblemResults(
            self, bootstrapped_sigma, bootstrapped_pi, bootstrapped_rho, bootstrapped_beta, bootstrapped_gamma,
            bootstrapped_prices, bootstrapped_shares, bootstrapped_delta, bootstrapped_costs, start_time, time.time(),
            draws, iteration_mappings, evaluation_mappings
        )
        output(f"Bootstrapped results after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    def _compute_bootstrap(
            self, iteration: Optional[Iteration], sigma: Array, pi: Array, rho: Array, beta: Array, gamma: Array) -> (
            Tuple[Array, Array, Array, Array, Dict[Hashable, int], Dict[Hashable, int], List[Error]]):
        """Compute the equilibrium prices, shares, marginal costs, and delta associated with bootstrapped parameters
        market-by-market
        """
        errors: List[Error] = []

        # compute delta (which will change under equilibrium prices) and marginal costs (which won't change)
        delta = self.true_delta + self._compute_true_X1() @ (beta - self.beta)
        costs = self.true_tilde_costs + self._compute_true_X3() @ (gamma - self.gamma)
        if self._costs_type == 'log':
            costs = np.exp(costs)

        # prices will only change if there is an iteration configuration
        prices = self.problem.products.prices if iteration is None else None

        # define a factory for computing bootstrapped prices, shares, and delta in markets
        def market_factory(s: Hashable) -> Tuple[ResultsMarket, Array, Optional[Array], Optional[Iteration]]:
            """Build a market along with arguments used to compute equilibrium prices and shares along with delta."""
            market_s = ResultsMarket(self.problem, s, sigma, pi, rho, beta, delta)
            costs_s = costs[self.problem._product_market_indices[s]]
            prices_s = prices[self.problem._product_market_indices[s]] if prices is not None else None
            return market_s, costs_s, prices_s, iteration

        # compute bootstrapped prices, shares, and delta market-by-market
        iteration_mapping: Dict[Hashable, int] = {}
        evaluation_mapping: Dict[Hashable, int] = {}
        equilibrium_prices = np.zeros_like(self.problem.products.prices)
        equilibrium_shares = np.zeros_like(self.problem.products.shares)
        generator = generate_items(self.problem.unique_market_ids, market_factory, ResultsMarket.solve_equilibrium)
        for t, (prices_t, shares_t, delta_t, errors_t, iterations_t, evaluations_t) in generator:
            equilibrium_prices[self.problem._product_market_indices[t]] = prices_t
            equilibrium_shares[self.problem._product_market_indices[t]] = shares_t
            delta[self.problem._product_market_indices[t]] = delta_t
            errors.extend(errors_t)
            iteration_mapping[t] = iterations_t
            evaluation_mapping[t] = evaluations_t

        # return all of the information associated with this bootstrap draw
        return equilibrium_prices, equilibrium_shares, delta, costs, iteration_mapping, evaluation_mapping, errors

    def compute_optimal_instruments(
            self, method: str = 'normal', draws: int = 100, seed: Optional[int] = None,
            expected_prices: Optional[Any] = None, iteration: Optional[Iteration] = None) -> 'OptimalInstrumentResults':
        r"""Estimate the set of optimal or efficient instruments, :math:`\mathscr{Z}_D` and :math:`\mathscr{Z}_S`.

        Optimal instruments have been shown, for example, by :ref:`Reynaert and Verboven (2014) <rv14>`, to not only
        reduce bias in the BLP problem, but also to improve efficiency and stability.

        The optimal instruments in the spirit of :ref:`Chamberlain (1987) <c87>` are

        .. math::

           \begin{bmatrix}
               \mathscr{Z}_D \\
               \mathscr{Z}_S
           \end{bmatrix}_{jt}
           = \text{Var}(\xi, \omega)^{-1}\operatorname{\mathbb{E}}\left[
           \begin{matrix}
               \frac{\partial\xi_{jt}}{\partial\theta} &
               \frac{\partial\xi_{jt}}{\partial\beta} &
               0 \\
               \frac{\partial\omega_{jt}}{\partial\theta} &
               \frac{\partial\omega_{jt}}{\partial\beta} &
               \frac{\partial\omega_{jt}}{\partial\gamma}
           \end{matrix}
           \mathrel{\Bigg|} Z \right],

        The expectation is taken by integrating over the joint density of :math:`\xi` and :math:`\omega`. For each error
        term realization, if not already estimated, equilibrium prices are computed via iteration over the
        :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>`. Associated shares and :math:`\delta`
        are then computed before each Jacobian is evaluated. Note that :math:`\partial\xi / \partial\beta = -X_1` and
        :math:`\partial\omega / \partial\gamma = -X_3`.

        The expected Jacobians are estimated with the average over all computed Jacobian realizations. The normalizing
        matrix :math:`\text{Var}(\xi, \omega)^{-1}` is estimated with the sample covariance matrix of the error terms.

        Parameters
        ----------
        method : `str, optional`
            The method by which the integral over the joint density of :math:`\xi` and :math:`\omega` is computed. The
            following methods are supported:

                - ``'normal'`` (default) - Draw from the normal approximation to the joint distribution of the error
                  terms and take the average over the computed Jacobians (`draws` determines the number of draws).

                - ``'empirical'`` - Draw with replacement from the empirical joint distribution of the error terms and
                  take the average over the computed Jacobians (`draws` determines the number of draws).

                - ``'approximate'`` - Evaluate the Jacobians at the expected value of the error terms: zero (`draws`
                  will be ignored).

        draws : `int, optional`
            The number of draws that will be taken from the joint distribution of the error terms. This is ignored if
            `method` is ``'approximate'``. The default is ``100``.
        seed : `int, optional`
            Passed to :class:`numpy.random.RandomState` to seed the random number generator before any draws are taken.
            By default, a seed is not passed to the random number generator.
        expected_prices : `array-like, optional`
            Vector of expected prices conditional on all exogenous variables,
            :math:`\operatorname{\mathbb{E}}[p \mid Z]`, which is required if a supply side was not estimated. A common
            way to estimate this vector is with the fitted values from a reduced form regression of endogenous prices
            onto all exogenous variables, including instruments. An example is given in the documentation for the
            convenience function :func:`compute_fitted_values`.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration used to estimate expected prices by iterating over the :math:`\zeta`-markup
            equation from :ref:`Morrow and Skerlos (2011) <ms11>`. By default, if a supply side was estimated, this is
            ``Iteration('simple', {'tol': 1e-12})``. It is not used if `expected_prices` is specified.

        Returns
        -------
        `OptimalInstrumentResults`
           Computed :class:`OptimalInstrumentResults`.

        """
        errors: List[Error] = []

        # keep track of long it takes to compute optimal instruments
        output("Computing optimal instruments ...")
        start_time = time.time()

        # validate the method and create a function that samples from the error distribution
        if method == 'approximate':
            sample = lambda: (np.zeros_like(self.xi), np.zeros_like(self.omega))
        else:
            state = np.random.RandomState(seed)
            if method == 'normal':
                if self.problem.K3 == 0:
                    variance = np.var(self.true_xi)
                    sample = lambda: (np.c_[state.normal(0, variance, self.problem.N)], self.omega)
                else:
                    covariances = np.cov(self.true_xi, self.true_omega, rowvar=False)
                    sample = lambda: np.hsplit(state.multivariate_normal([0, 0], covariances, self.problem.N), 2)
            elif method == 'empirical':
                if self.problem.K3 == 0:
                    sample = lambda: (self.true_xi[state.choice(self.problem.N, self.problem.N)], self.omega)
                else:
                    joint = np.c_[self.true_xi, self.true_omega]
                    sample = lambda: np.hsplit(joint[state.choice(self.problem.N, self.problem.N)], 2)
            else:
                raise ValueError("method must be 'approximate', 'normal', or 'empirical'.")

        # validate the number of draws (there will be only one for the approximate method)
        if method == 'approximate':
            draws = 1
        if not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive int.")

        # validate expected prices or their iteration configuration
        if expected_prices is None:
            if self.problem.K3 == 0:
                raise TypeError("A supply side was not estimated, so expected_prices must be specified.")
            if iteration is None:
                iteration = Iteration('simple', {'tol': 1e-12})
            elif not isinstance(iteration, Iteration):
                raise TypeError("iteration must be None or an Iteration instance.")
        else:
            iteration = None
            expected_prices = np.c_[np.asarray(expected_prices, options.dtype)]
            if expected_prices.shape != (self.problem.N, 1):
                raise ValueError(f"expected_prices must be a {self.problem.N}-vector.")

        # average over Jacobian realizations
        iteration_mappings: List[Dict[Hashable, int]] = []
        evaluation_mappings: List[Dict[Hashable, int]] = []
        expected_xi_by_theta = np.zeros_like(self.xi_by_theta_jacobian)
        expected_xi_by_beta = np.zeros_like(self.problem.products.X1)
        expected_omega_by_theta = np.zeros_like(self.omega_by_theta_jacobian)
        expected_omega_by_beta = np.zeros_like(self.omega_by_beta_jacobian)
        for _ in output_progress(range(draws), draws, start_time):
            xi_by_theta_i, xi_by_beta_i, omega_by_theta_i, omega_by_beta_i, iterations_i, evaluations_i, errors_i = (
                self._compute_realizations(expected_prices, iteration, *sample())
            )
            expected_xi_by_theta += xi_by_theta_i / draws
            expected_xi_by_beta += xi_by_beta_i / draws
            expected_omega_by_theta += omega_by_theta_i / draws
            expected_omega_by_beta += omega_by_beta_i / draws
            iteration_mappings.append(iterations_i)
            evaluation_mappings.append(evaluations_i)
            errors.extend(errors_i)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # compute the optimal instruments
        if self.problem.K3 == 0:
            inverse_covariance_matrix = np.c_[1 / np.var(self.true_xi)]
            demand_instruments = inverse_covariance_matrix * np.c_[expected_xi_by_theta, expected_xi_by_beta]
            supply_instruments = np.full((self.problem.N, 0), np.nan, options.dtype)
        else:
            inverse_covariance_matrix = np.c_[scipy.linalg.inv(np.cov(self.true_xi, self.true_omega, rowvar=False))]
            jacobian = np.r_[
                np.c_[expected_xi_by_theta, expected_xi_by_beta, np.zeros_like(self.problem.products.X3)],
                np.c_[expected_omega_by_theta, expected_omega_by_beta, -self._compute_true_X3()]
            ]
            tensor = multiply_matrix_and_tensor(inverse_covariance_matrix, np.stack(np.split(jacobian, 2), axis=1))
            demand_instruments, supply_instruments = np.split(tensor.reshape((self.problem.N, -1)), 2, axis=1)

        # structure the results
        results = OptimalInstrumentResults(
            self, demand_instruments, supply_instruments, inverse_covariance_matrix, expected_xi_by_theta,
            expected_xi_by_beta, expected_omega_by_theta, expected_omega_by_beta, start_time, time.time(), draws,
            iteration_mappings, evaluation_mappings
        )
        output(f"Computed optimal instruments after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    def _compute_realizations(
            self, expected_prices: Optional[Array], iteration: Optional[Iteration], xi: Array, omega: Array) -> (
            Tuple[Array, Array, Array, Array, Dict[Hashable, int], Dict[Hashable, int], List[Error]]):
        """If they have not already been estimated, compute the equilibrium prices, shares, and delta associated with a
        realization of xi and omega market-by-market. Then, compute realizations of Jacobians of xi and omega with
        respect to theta and beta.
        """
        errors: List[Error] = []

        # compute delta (which will change under equilibrium prices) and marginal costs (which won't change)
        delta = self.true_delta - self.true_xi + xi
        costs = tilde_costs = self.true_tilde_costs - self.true_omega + omega
        if self._costs_type == 'log':
            costs = np.exp(costs)

        # define a factory for computing realizations of prices, shares, and delta in markets
        def market_factory(s: Hashable) -> Tuple[ResultsMarket, Array, Optional[Array], Optional[Iteration]]:
            """Build a market along with arguments used to compute equilibrium prices and shares along with delta."""
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, delta)
            costs_s = costs[self.problem._product_market_indices[s]]
            prices_s = expected_prices[self.problem._product_market_indices[s]] if expected_prices is not None else None
            return market_s, costs_s, prices_s, iteration

        # compute realizations of prices, shares, and delta market-by-market
        iteration_mapping: Dict[Hashable, int] = {}
        evaluation_mapping: Dict[Hashable, int] = {}
        equilibrium_prices = np.zeros_like(self.problem.products.prices)
        equilibrium_shares = np.zeros_like(self.problem.products.shares)
        generator = generate_items(self.problem.unique_market_ids, market_factory, ResultsMarket.solve_equilibrium)
        for t, (prices_t, shares_t, delta_t, errors_t, iterations_t, evaluations_t) in generator:
            equilibrium_prices[self.problem._product_market_indices[t]] = prices_t
            equilibrium_shares[self.problem._product_market_indices[t]] = shares_t
            delta[self.problem._product_market_indices[t]] = delta_t
            errors.extend(errors_t)
            iteration_mapping[t] = iterations_t
            evaluation_mapping[t] = evaluations_t

        # compute the Jacobian of xi with respect to theta
        xi_by_theta_jacobian, demand_errors = self._compute_demand_realizations(
            equilibrium_prices, equilibrium_shares, delta
        )
        errors.extend(demand_errors)

        # compute the Jacobian of xi with respect to beta (prices just need to be replaced in X1)
        xi_by_beta_jacobian = -self._compute_true_X1({'prices': equilibrium_prices})

        # compute the Jacobians of omega with respect to theta and beta
        omega_by_theta_jacobian = np.full((self.problem.N, self._nonlinear_parameters.P), np.nan, options.dtype)
        omega_by_beta_jacobian = np.full((self.problem.N, self.problem.K1), np.nan, options.dtype)
        if self.problem.K3 > 0:
            omega_by_theta_jacobian, omega_by_beta_jacobian, supply_errors = self._compute_supply_realizations(
                equilibrium_prices, equilibrium_shares, delta, tilde_costs, xi_by_theta_jacobian
            )
            errors.extend(supply_errors)

        # return all of the information associated with this realization
        return (
            xi_by_theta_jacobian, xi_by_beta_jacobian, omega_by_theta_jacobian, omega_by_beta_jacobian,
            iteration_mapping, evaluation_mapping, errors
        )

    def _compute_demand_realizations(
            self, equilibrium_prices: Array, equilibrium_shares: Array, delta: Array) -> Tuple[Array, List[Error]]:
        """Compute a realization of the Jacobian of xi with respect to theta market-by-market. If necessary, revert
        problematic elements to their estimated values.
        """
        errors: List[Error] = []

        # check if the Jacobian does not need to be computed
        xi_by_theta_jacobian = np.full((self.problem.N, self._nonlinear_parameters.P), np.nan, options.dtype)
        if self._nonlinear_parameters.P == 0:
            return xi_by_theta_jacobian, errors

        # define a factory for computing the Jacobian of xi with respect to theta in markets
        def market_factory(s: Hashable) -> Tuple[ResultsMarket, NonlinearParameters]:
            """Build a market with the data realization along with arguments used to compute the Jacobian."""
            data_override_s = {
                'prices': equilibrium_prices[self.problem._product_market_indices[s]],
                'shares': equilibrium_shares[self.problem._product_market_indices[s]]
            }
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, delta, data_override_s)
            return market_s, self._nonlinear_parameters

        # compute the Jacobian market-by-market
        generator = generate_items(
            self.problem.unique_market_ids, market_factory, ResultsMarket.compute_xi_by_theta_jacobian
        )
        for t, (xi_by_theta_jacobian_t, errors_t) in generator:
            xi_by_theta_jacobian[self.problem._product_market_indices[t]] = xi_by_theta_jacobian_t
            errors.extend(errors_t)

        # replace invalid elements
        bad_indices = ~np.isfinite(xi_by_theta_jacobian)
        if np.any(bad_indices):
            xi_by_theta_jacobian[bad_indices] = self.xi_by_theta_jacobian[bad_indices]
            errors.append(exceptions.XiByThetaJacobianReversionError(bad_indices))
        return xi_by_theta_jacobian, errors

    def _compute_supply_realizations(
            self, equilibrium_prices: Array, equilibrium_shares: Array, delta: Array, tilde_costs: Array,
            xi_jacobian: Array) -> Tuple[Array, Array, List[Error]]:
        """Compute realizations of the Jacobians of omega with respect to theta and beta market-by-market. If necessary,
        revert problematic elements to their estimated values.
        """
        errors: List[Error] = []

        # compute the Jacobian of beta with respect to theta, which is needed to compute the Jacobian of omega with
        #   respect to theta
        demand_iv = IV(self.problem.products.X1, self.problem.products.ZD, self.WD)
        beta_jacobian = demand_iv.estimate(xi_jacobian, residuals=False)
        errors.extend(demand_iv.errors)

        # define a factory for computing the Jacobians of omega with respect to theta and beta in markets
        def market_factory(s: Hashable) -> Tuple[ResultsMarket, Array, Array, Array, NonlinearParameters, str]:
            """Build a market with the data realization along with arguments used to compute the Jacobians."""
            data_override_s = {
                'prices': equilibrium_prices[self.problem._product_market_indices[s]],
                'shares': equilibrium_shares[self.problem._product_market_indices[s]]
            }
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, delta, data_override_s)
            tilde_costs_s = tilde_costs[self.problem._product_market_indices[s]]
            xi_jacobian_s = xi_jacobian[self.problem._product_market_indices[s]]
            return market_s, tilde_costs_s, xi_jacobian_s, beta_jacobian, self._nonlinear_parameters, self._costs_type

        # compute the Jacobians market-by-market
        omega_by_theta_jacobian = np.full((self.problem.N, self._nonlinear_parameters.P), np.nan, options.dtype)
        omega_by_beta_jacobian = np.full((self.problem.N, self.problem.K1), np.nan, options.dtype)
        generator = generate_items(
            self.problem.unique_market_ids, market_factory, ResultsMarket.compute_omega_jacobians
        )
        for t, (omega_by_theta_jacobian_t, omega_by_beta_jacobian_t, errors_t) in generator:
            omega_by_theta_jacobian[self.problem._product_market_indices[t]] = omega_by_theta_jacobian_t
            omega_by_beta_jacobian[self.problem._product_market_indices[t]] = omega_by_beta_jacobian_t
            errors.extend(errors_t)

        # the Jacobians should be zero for any clipped marginal costs
        omega_by_theta_jacobian[self.clipped_costs.flat] = 0
        omega_by_beta_jacobian[self.clipped_costs.flat] = 0

        # replace invalid elements in the Jacobian of omega with respect to theta
        bad_indices = ~np.isfinite(omega_by_theta_jacobian)
        if np.any(bad_indices):
            omega_by_theta_jacobian[bad_indices] = self.omega_by_theta_jacobian[bad_indices]
            errors.append(exceptions.OmegaByThetaJacobianReversionError(bad_indices))

        # replace invalid elements in the Jacobian of omega with respect to beta
        bad_indices = ~np.isfinite(omega_by_theta_jacobian)
        if np.any(bad_indices):
            omega_by_beta_jacobian[bad_indices] = self.omega_by_beta_jacobian[bad_indices]
            errors.append(exceptions.OmegaByBetaJacobianReversionError(bad_indices))
        return omega_by_theta_jacobian, omega_by_beta_jacobian, errors

    def _coerce_matrices(self, matrices: Any) -> Array:
        """Coerce array-like stacked matrices into a stacked matrix and validate it."""
        matrices = np.c_[np.asarray(matrices, options.dtype)]
        if matrices.shape != (self.problem.N, self.problem._max_J):
            raise ValueError(f"matrices must be {self.problem.N} by {self.problem._max_J}.")
        return matrices

    def _coerce_optional_costs(self, costs: Optional[Any]) -> Array:
        """Coerce optional array-like costs into a column vector and validate it."""
        if costs is not None:
            costs = np.c_[np.asarray(costs, options.dtype)]
            if costs.shape != (self.problem.N, 1):
                raise ValueError(f"costs must be None or a {self.problem.N}-vector.")
        return costs

    def _coerce_optional_prices(self, prices: Optional[Any]) -> Array:
        """Coerce optional array-like prices into a column vector and validate it."""
        if prices is not None:
            prices = np.c_[np.asarray(prices, options.dtype)]
            if prices.shape != (self.problem.N, 1):
                raise ValueError(f"prices must be None or a {self.problem.N}-vector.")
        return prices

    def _coerce_optional_shares(self, shares: Optional[Any]) -> Array:
        """Coerce optional array-like shares into a column vector and validate it."""
        if shares is not None:
            shares = np.c_[np.asarray(shares, options.dtype)]
            if shares.shape != (self.problem.N, 1):
                raise ValueError(f"shares must be None or a {self.problem.N}-vector.")
        return shares

    def _combine_arrays(self, compute_market_results: Callable, fixed_args: Sequence, market_args: Sequence) -> Array:
        """Compute an array for each market and stack them into a single matrix. An array for a single market is
        computed by passing fixed_args (identical for all markets) and market_args (matrices with as many rows as there
        are products that are restricted to the market) to compute_market_results, a ResultsMarket method that returns
        the output for the market and a set of any errors encountered during computation.
        """
        errors: List[Error] = []

        # keep track of how long it takes to compute the arrays
        start_time = time.time()

        # define a factory for computing arrays in markets
        def market_factory(s: Hashable) -> tuple:
            """Build a market along with arguments used to compute arrays."""
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, self.true_delta)
            args_s = [None if a is None else a[self.problem._product_market_indices[s]] for a in market_args]
            return (market_s, *fixed_args, *args_s)

        # construct a mapping from market IDs to market-specific arrays
        matrix_mapping: Dict[Hashable, Array] = {}
        generator = output_progress(
            generate_items(self.problem.unique_market_ids, market_factory, compute_market_results), self.problem.T,
            start_time
        )
        for t, (array_t, errors_t) in generator:
            matrix_mapping[t] = np.c_[array_t]
            errors.extend(errors_t)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # determine the number of rows and columns
        row_count = sum(matrix_mapping[t].shape[0] for t in self.problem.unique_market_ids)
        column_count = max(matrix_mapping[t].shape[1] for t in self.problem.unique_market_ids)

        # preserve the original product order or the sorted market order when stacking the arrays
        combined = np.full((row_count, column_count), np.nan, options.dtype)
        for t, matrix_t in matrix_mapping.items():
            if row_count == self.problem.N:
                combined[self.problem._product_market_indices[t], :matrix_t.shape[1]] = matrix_t
            else:
                combined[self.problem.unique_market_ids == t, :matrix_t.shape[1]] = matrix_t

        # output how long it took to compute the arrays
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return combined


class BootstrappedProblemResults(_ProblemResults):
    r"""Bootstrapped results of a solved BLP problem.

    This class has all of the same methods as :class:`ProblemResults`, for except :meth:`ProblemResults.bootstrap` and
    :meth:`ProblemResults.compute_optimal_instruments`. The only difference is that methods return arrays with an extra
    first dimension, along which bootstrapped results are stacked (these stacked results can be used to construct, for
    example, confidence intervals for post-estimation outputs). Similarly, arrays of data passed as arguments to methods
    should have an extra first dimension of size :attr:`BootstrappedProblemResults.draws`.

    Attributes
    ----------
    problem_results : `ProblemResults`
        :class:`ProblemResults` that was used to compute these optimal instrument results.
    bootstrapped_sigma : `ndarray`
        Bootstrapped Cholesky decomposition of the covariance matrix that measures agents' random taste distribution,
        :math:\Sigma`.
    bootstrapped_pi : `ndarray`
        Bootstrapped parameters that measures how agent tastes vary with demographics, :math:\Pi`.
    bootstrapped_rho : `ndarray`
        Bootstrapped parameters that measure within nesting group correlations, :math:\rho`.
    bootstrapped_beta : `ndarray`
        Bootstrapped demand-side linear parameters, :math:\beta`.
    bootstrapped_gamma : `ndarray`
        Bootstrapped supply-side linear parameters, :math:\gamma`.
    bootstrapped_prices : `ndarray`
        Bootstrapped prices, :math:`p`. If a supply-side was not estimated, these are unchanged prices. Otherwise, they
        are equilibrium prices implied by each draw.
    bootstrapped_shares : `ndarray`
        Bootstrapped shares, :math:`p`, implied by each draw.
    bootstrapped_delta : `ndarray`
        Bootstrapped mean utility, :math:`\delta`, implied by each draw.
    bootstrapped_costs : `ndarray`
        Bootstrapped marginal costs, :math:`c`, implied by each draw.
    computation_time : `float`
        Number of seconds it took to compute the bootstrapped results.
    draws : `int`
        Number of bootstrap draws.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute equilibrium prices in each market
        for each draw. Rows are in the same order as :attr:`Problem.unique_market_ids` and column indices correspond to
        draws.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute equilibrium prices was evaluated in each market for each draw.
        Rows are in the same order as :attr:`Problem.unique_market_ids` and column indices correspond to draws.

    Examples
    --------
    For an example of how to use this class, refer to the :doc:`Examples </examples>` section.

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
    fp_iterations: Array
    contraction_evaluations: Array

    def __init__(
            self, problem_results: ProblemResults, bootstrapped_sigma: Array, bootstrapped_pi: Array,
            bootstrapped_rho: Array, bootstrapped_beta: Array, bootstrapped_gamma: Array, bootstrapped_prices: Array,
            bootstrapped_shares: Array, bootstrapped_delta: Array, bootstrapped_costs: Array, start_time: float,
            end_time: float, draws: int, iteration_mappings: List[Dict[Hashable, int]],
            evaluation_mappings: List[Dict[Hashable, int]]) -> None:
        """Structure bootstrapped problem results."""
        super().__init__(problem_results.problem)
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
        self.fp_iterations = self.cumulative_fp_iterations = np.array(
            [[m[t] if m else 0 for m in iteration_mappings] for t in problem_results.problem.unique_market_ids]
        )
        self.contraction_evaluations = self.cumulative_contraction_evaluations = np.array(
            [[m[t] if m else 0 for m in evaluation_mappings] for t in problem_results.problem.unique_market_ids]
        )

    def __str__(self) -> str:
        """Format bootstrapped problem results as a string."""
        header = [("Computation", "Time"), ("Bootstrap", "Draws")]
        values = [format_seconds(self.computation_time), self.draws]
        if self.fp_iterations.sum() > 0 or self.contraction_evaluations.sum() > 0:
            header.extend([("Total Fixed Point", "Iterations"), ("Total Contraction", "Evaluations")])
            values.extend([self.fp_iterations.sum(), self.contraction_evaluations.sum()])
        widths = [max(len(k1), len(k2)) for k1, k2 in header]
        formatter = TableFormatter(widths)
        return "\n".join([
            "Bootstrapped Problem Results Summary:",
            formatter.line(),
            formatter([k[0] for k in header]),
            formatter([k[1] for k in header], underline=True),
            formatter(values),
            formatter.line()
        ])

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

    def _combine_arrays(self, compute_market_results: Callable, fixed_args: Sequence, market_args: Sequence) -> Array:
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
                self.problem, s, self.bootstrapped_sigma[c], self.bootstrapped_pi[c], self.bootstrapped_rho[c],
                self.bootstrapped_beta[c], self.bootstrapped_delta[c], data_override_cs
            )
            args_cs = [None if a is None else a[c, self.problem._product_market_indices[s][0]] for a in market_args]
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


class OptimalInstrumentResults(StringRepresentation):
    r"""Results of optimal instrument computation.

    The :meth:`OptimalInstrumentResults.to_problem` method can be used to update the original :class:`Problem` with
    the computed optimal instruments. If a supply side was estimated, some columns of optimal instruments may need to
    be dropped because of collinearity issues. Refer to :meth:`OptimalInstrumentResults.to_problem` for more information
    about how to drop these collinear instruments.

    Attributes
    ----------
    problem_results : `ProblemResults`
        :class:`ProblemResults` that was used to compute these optimal instrument results.
    demand_instruments: `ndarray`
        Estimated optimal demand-side instruments, :math:`\mathscr{Z}_D`.
    supply_instruments: `ndarray`
        Estimated optimal supply-side instruments, :math:`\mathscr{Z}_S`.
    inverse_covariance_matrix: `ndarray`
        Inverse of the sample covariance matrix of the estimated :math:`\xi` and :math:`\omega`, which is used to
        normalize the expected Jacobians. If a supply side was not estimated, this is simply the sample estimate of
        :math:`1 / \text{Var}(\xi)`.
    expected_xi_by_theta_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\xi / \partial\theta \mid Z]`.
    expected_xi_by_beta_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\xi / \partial\beta \mid Z]`.
    expected_omega_by_theta_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\omega / \partial\theta \mid Z]`.
    expected_omega_by_beta_jacobian: `ndarray`
        Estimated :math:`\operatorname{\mathbb{E}}[\partial\omega / \partial\beta \mid Z]`.
    computation_time : `float`
        Number of seconds it took to compute optimal instruments.
    draws : `int`
        Number of draws used to approximate the integral over the error term density.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute equilibrium prices in each market
        for each error term draw. Rows are in the same order as :attr:`Problem.unique_market_ids` and column indices
        correspond to draws.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute equilibrium prices was evaluated in each market for each error
        term draw. Rows are in the same order as :attr:`Problem.unique_market_ids` and column indices correspond to
        draws.
    MD : `int`
        Number of computed optimal demand-side instruments, :math:`M_D`.
    MS : `int`
        Number of computed optimal supply-side instruments, :math:`M_S`.

    Examples
    --------
    For an example of how to use this class, refer to the :doc:`Examples </examples>` section.

    """

    problem_results: ProblemResults
    demand_instruments: Array
    supply_instruments: Array
    inverse_covariance_matrix: Array
    expected_xi_by_theta_jacobian: Array
    expected_xi_by_beta_jacobian: Array
    expected_omega_by_theta_jacobian: Array
    expected_omega_by_beta_jacobian: Array
    computation_time: float
    draws: int
    fp_iterations: Array
    contraction_evaluations: Array
    MD: int
    MS: int

    def __init__(
            self, problem_results: ProblemResults, demand_instruments: Array, supply_instruments: Array,
            inverse_covariance_matrix: Array, expected_xi_by_theta_jacobian: Array,
            expected_xi_by_beta_jacobian: Array, expected_omega_by_theta_jacobian: Array,
            expected_omega_by_beta_jacobian: Array, start_time: float, end_time: float, draws: int,
            iteration_mappings: Sequence[Mapping[Hashable, int]],
            evaluation_mappings: Sequence[Mapping[Hashable, int]]) -> None:
        """Structure optimal instrument computation results."""
        self.problem_results = problem_results
        self.demand_instruments = demand_instruments
        self.supply_instruments = supply_instruments
        self.inverse_covariance_matrix = inverse_covariance_matrix
        self.expected_xi_by_theta_jacobian = expected_xi_by_theta_jacobian
        self.expected_xi_by_beta_jacobian = expected_xi_by_beta_jacobian
        self.expected_omega_by_theta_jacobian = expected_omega_by_theta_jacobian
        self.expected_omega_by_beta_jacobian = expected_omega_by_beta_jacobian
        self.computation_time = end_time - start_time
        self.draws = draws
        self.fp_iterations = self.cumulative_fp_iterations = np.array(
            [[m[t] if m else 0 for m in iteration_mappings] for t in problem_results.problem.unique_market_ids]
        )
        self.contraction_evaluations = self.cumulative_contraction_evaluations = np.array(
            [[m[t] if m else 0 for m in evaluation_mappings] for t in problem_results.problem.unique_market_ids]
        )
        self.MD = demand_instruments.shape[1]
        self.MS = supply_instruments.shape[1]

    def __str__(self) -> str:
        """Format optimal instrument computation results as a string."""

        # construct a section containing summary information
        summary_header = [
            ("Computation", "Time"), ("Error Term", "Draws"), ("MD:", "Demand Instruments"),
            ("MS:", "Supply Instruments")
        ]
        summary_values = [format_seconds(self.computation_time), self.draws, self.MD, self.MS]
        if self.fp_iterations.sum() > 0 or self.contraction_evaluations.sum() > 0:
            summary_header.extend([("Total Fixed Point", "Iterations"), ("Total Contraction", "Evaluations")])
            summary_values.extend([self.fp_iterations.sum(), self.contraction_evaluations.sum()])
        summary_widths = [max(len(k1), len(k2)) for k1, k2 in summary_header]
        summary_formatter = TableFormatter(summary_widths)
        summary_section = [
            "Optimal Instrument Computation Results Summary:",
            summary_formatter.line(),
            summary_formatter([k[0] for k in summary_header]),
            summary_formatter([k[1] for k in summary_header], underline=True),
            summary_formatter(summary_values),
            summary_formatter.line()
        ]

        # collect information about formulations associated with instruments
        problem = self.problem_results.problem
        formulation_mappings: List[Dict[str, ColumnFormulation]] = []  # noqa
        for parameter in self.problem_results._nonlinear_parameters.unfixed:
            if isinstance(parameter, SigmaParameter):
                formulation_mappings.append({
                    "Sigma Row": problem._X2_formulations[parameter.location[0]],
                    "Sigma Column": problem._X2_formulations[parameter.location[1]],
                })
            elif isinstance(parameter, PiParameter):
                formulation_mappings.append({
                    "Pi Row": problem._X2_formulations[parameter.location[0]],
                    "Pi Column": problem._X2_formulations[parameter.location[1]],
                })
            else:
                assert isinstance(parameter, RhoParameter)
                group_label = "All Groups" if parameter.single else problem.unique_nesting_ids[parameter.location[0]]
                formulation_mappings.append({"Rho Element": group_label})
        formulation_mappings.extend([{"Beta Element": f} for f in problem._X1_formulations])
        formulation_mappings.extend([{"Gamma Element": f} for f in problem._X3_formulations])

        # construct a section containing formulation information
        formulation_header = ["Column Indices:"] + list(map(str, range(len(formulation_mappings))))
        formulation_widths = [max(len(formulation_header[0]), max(map(len, formulation_mappings)))]
        for mapping in formulation_mappings:
            formulation_widths.append(max(5, max(map(len, map(str, mapping.values())))))
        formulation_formatter = TableFormatter(formulation_widths)
        formulation_section = [
            "Instruments:",
            formulation_formatter.line(),
            formulation_formatter(formulation_header, underline=True)
        ]
        for key in ["Sigma Row", "Sigma Column", "Pi Row", "Pi Column", "Rho Element", "Beta Element", "Gamma Element"]:
            formulations = [str(m.get(key, "")) for m in formulation_mappings]
            if any(formulations):
                formulation_section.append(formulation_formatter([key] + formulations))
        formulation_section.append(formulation_formatter.line())

        # combine the sections into one string
        return "\n\n".join("\n".join(s) for s in [summary_section, formulation_section])

    def to_problem(
            self, delete_demand_instruments: Sequence[int] = (),
            delete_supply_instruments: Sequence[int] = ()) -> '_Problem':
        """Re-create the problem with estimated optimal instruments.

        The re-created problem will be exactly the same, except that instruments will be replaced with the estimated
        optimal instruments.

        With a supply side, dropping one or more columns of instruments is often necessary because :math:`X_1` and
        :math:`X_3` are often formulated to include identical or similar exogenous product characteristics. Optimal
        instruments for these characteristics (the re-scaled characteristics themselves) will be collinear. The
        `delete_demand_instruments` and `delete_supply_instruments` arguments can be used to delete instruments
        when re-creating the problem. Outputted optimal instrument results indicate which instrument column indices
        correspond to which product characteristics.

        For example, if :math:`X_1` contains some exogenous characteristic ``x`` and :math:`X_3` contains ``log(x)``,
        both the demand- and supply-side optimal instruments will contain scaled versions of these almost collinear
        characteristics. One way to deal with this is to include the column index of the instrument for ``log(x)`` in
        `delete_demand_instruments` and to include the column index of the instrument for ``x`` in
        `delete_supply_instruments`.

        .. note::

           Any fixed effects that were absorbed in the original problem will be be absorbed here too. However, compared
           to a problem updated with optimal instruments when fixed effects are included as indicators, results may be
           slightly different.

        Parameters
        ----------
        delete_demand_instruments : `tuple of int, optional`
            Column indices of :attr:`OptimalInstrumentResults.demand_instruments` to drop when re-creating the problem.
        delete_supply_instruments : `tuple of int, optional`
            Column indices of :attr:`OptimalInstrumentResults.supply_instruments` to drop when re-creating the problem.
            This is only relevant if a supply side was estimated.

        Returns
        -------
        `Problem`
            :class:`Problem` updated to use the estimated optimal instruments.

        Examples
        --------
        For an example of turning these results into a :class:`Problem`, refer to the :doc:`Examples </examples>`
        section.

        """

        # keep track of long it takes to re-create the problem
        output("Re-creating the problem ...")
        start_time = time.time()

        # validate the indices
        if not isinstance(delete_demand_instruments, collections.Sequence):
            raise TypeError("delete_demand_instruments must be a tuple.")
        if not isinstance(delete_supply_instruments, collections.Sequence):
            raise TypeError("delete_supply_instruments must be a tuple.")
        if not all(i in range(self.MD) for i in delete_demand_instruments):
            raise ValueError(f"delete_demand_instruments must contain column indices between 0 and {self.MD}.")
        if not all(i in range(self.MS) for i in delete_supply_instruments):
            raise ValueError(f"delete_supply_instruments must contain column indices between 0 and {self.MS}.")
        if self.MS == 0 and delete_supply_instruments:
            raise ValueError("A supply side was not estimated, so delete_supply_instruments should not be specified.")

        # update the products array
        updated_products = update_matrices(self.problem_results.problem.products, {
            'ZD': (np.delete(self.demand_instruments, delete_demand_instruments, axis=1), options.dtype),
            'ZS': (np.delete(self.supply_instruments, delete_supply_instruments, axis=1), options.dtype)
        })

        # re-create the problem
        from .problem import _Problem  # noqa
        problem = _Problem(
            self.problem_results.problem.product_formulations, self.problem_results.problem.agent_formulation,
            updated_products, self.problem_results.problem.agents, updating_instruments=True
        )
        output(f"Re-created the problem after {format_seconds(time.time() - start_time)}.")
        output("")
        output(problem)
        return problem


class SimulationResults(StringRepresentation):
    """Results of a solved simulation of synthetic BLP data.

    The :meth:`SimulationResults.to_problem` method can be used to convert the full set of simulated data and configured
    information into a :class:`Problem`.

    Attributes
    ----------
    simulation: `Simulation`
        :class:`Simulation` that created these results.
    firms_index : `int`
        Column index of the firm IDs in the `firm_ids` field of `product_data` in :class:`Simulation` that defined which
        firms produce which products during computation of synthetic prices and shares.
    product_data : `recarray`
        Simulated :attr:`Simulation.product_data` that are updated with synthetic prices and shares.
    computation_time : `float`
        Number of seconds it took to compute synthetic prices and shares.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute synthetic prices in each market.
        Rows are in the same order as :attr:`Simulation.unique_market_ids`.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute synthetic prices was evaluated in each market. Rows are in the
        same order as :attr:`Simulation.unique_market_ids`.

    Examples
    --------
    For an example of turning these results into a :class:`Problem` and then solving the problem, refer to the
    :doc:`Examples </examples>` section.

    """

    simulation: 'Simulation'
    firms_index: int
    product_data: RecArray
    computation_time: float
    fp_iterations: Array
    contraction_evaluations: Array

    def __init__(
            self, simulation: 'Simulation', firms_index: int, prices: Array, shares: Array, start_time: float,
            end_time: float, iteration_mapping: Dict[Hashable, int], evaluation_mapping: Dict[Hashable, int]) -> None:
        """Structure simulation results."""
        self.simulation = simulation
        self.firms_index = firms_index
        self.product_data = simulation.product_data.copy()
        self.product_data.prices = prices
        self.product_data.shares = shares
        self.computation_time = end_time - start_time
        self.fp_iterations = np.array([iteration_mapping[t] for t in simulation.unique_market_ids])
        self.contraction_evaluations = np.array([evaluation_mapping[t] for t in simulation.unique_market_ids])

    def __str__(self) -> str:
        """Format simulation results as a string."""
        header = [("Computation", "Time"), ("Fixed Point", "Iterations"), ("Contraction", "Evaluations")]
        widths = [max(len(k1), len(k2)) for k1, k2 in header]
        formatter = TableFormatter(widths)
        return "\n".join([
            "Simulation Results Summary:",
            formatter.line(),
            formatter([k[0] for k in header]),
            formatter([k[1] for k in header], underline=True),
            formatter([
                format_seconds(self.computation_time),
                self.fp_iterations.sum(),
                self.contraction_evaluations.sum()
            ]),
            formatter.line()
        ])

    def to_problem(
            self, product_formulations: Optional[Union[Formulation, Sequence[Optional[Formulation]]]] = None,
            product_data: Optional[Mapping] = None, agent_formulation: Optional[Formulation] = None,
            agent_data: Optional[Mapping] = None, integration: Optional[Integration] = None) -> 'Problem':
        """Convert the solved simulation into a problem.

        Parameters are the same as those of :class:`Problem`. By default, the structure of the problem will be the same
        as that of the solved simulation.

        Parameters
        ----------
        product_formulations : `Formulation or tuple of Formulation, optional`
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

        """
        from .problem import Problem  # noqa
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


class ResultsMarket(Market):
    """A single market of a solved problem."""

    def solve_equilibrium(
            self, costs: Array, prices: Optional[Array], iteration: Optional[Iteration]) -> (
            Tuple[Array, Array, Array, List[Error], int, int]):
        """If not already estimated, compute equilibrium prices along with associated delta and shares."""
        errors: List[Error] = []

        # configure NumPy to identify floating point errors
        with np.errstate(divide='call', over='call', under='ignore', invalid='call'):
            np.seterrcall(lambda *_: errors.append(exceptions.EquilibriumPricesFloatingPointError()))

            # solve the fixed point problem if prices haven't already been estimated
            iterations = evaluations = 0
            if iteration is None:
                assert prices is not None
            else:
                prices, converged, iterations, evaluations = self.compute_equilibrium_prices(costs, iteration)
                if not converged:
                    errors.append(exceptions.EquilibriumPricesConvergenceError())

            # switch to identifying floating point errors with equilibrium share computation
            np.seterrcall(lambda *_: errors.append(exceptions.EquilibriumSharesFloatingPointError()))

            # compute the associated shares
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
            shares = self.compute_probabilities(delta, mu) @ self.agents.weights
            return prices, shares, delta, errors, iterations, evaluations

    def compute_omega_jacobians(
            self, tilde_costs: Array, xi_jacobian: Array, beta_jacobian: Array,
            nonlinear_parameters: NonlinearParameters, costs_type: str) -> Tuple[Array, Array, List[Error]]:
        """Compute the Jacobians of omega (equivalently, of transformed marginal costs) with respect to theta and
        beta (but first check if the latter is all zeros).
        """
        omega_by_theta_jacobian, theta_errors = self.compute_omega_by_theta_jacobian(
            tilde_costs, xi_jacobian, beta_jacobian, nonlinear_parameters, costs_type
        )
        omega_by_beta_jacobian, beta_errors = self.compute_omega_by_beta_jacobian(tilde_costs, costs_type)
        return omega_by_theta_jacobian, omega_by_beta_jacobian, theta_errors + beta_errors

    def compute_aggregate_elasticity(self, factor: float, name: str) -> Tuple[Array, List[Error]]:
        """Estimate the aggregate elasticity of demand with respect to a variable."""
        scaled_variable = (1 + factor) * self.products[name]
        delta = self.update_delta_with_variable(name, scaled_variable)
        mu = self.update_mu_with_variable(name, scaled_variable)
        shares = self.compute_probabilities(delta, mu) @ self.agents.weights
        aggregate_elasticities = (shares - self.products.shares).sum() / factor
        return aggregate_elasticities, []

    def compute_elasticities(self, name: str) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of elasticities of demand with respect to a variable."""
        derivatives = self.compute_utility_derivatives(name)
        jacobian = self.compute_shares_by_variable_jacobian(derivatives)
        elasticities = jacobian * self.products[name].T / self.products.shares
        return elasticities, []

    def compute_diversion_ratios(self, name: str) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of diversion ratios with respect to a variable."""
        derivatives = self.compute_utility_derivatives(name)
        jacobian = self.compute_shares_by_variable_jacobian(derivatives)

        # replace the diagonal with derivatives with respect to the outside option
        jacobian_diagonal = np.c_[jacobian.diagonal()]
        jacobian[np.diag_indices_from(jacobian)] = -jacobian.sum(axis=1)

        # compute the ratios
        ratios = -jacobian / np.tile(jacobian_diagonal, self.J)
        return ratios, []

    def compute_long_run_diversion_ratios(self) -> Tuple[Array, List[Error]]:
        """Estimate a matrix of long-run diversion ratios."""

        # compute share differences when products are excluded and store outside share differences on the diagonal
        changes = np.zeros((self.J, self.J), options.dtype)
        for j in range(self.J):
            shares_without_j = self.compute_probabilities(eliminate_product=j) @ self.agents.weights
            changes[j] = (shares_without_j - self.products.shares).flat
            changes[j, j] = -changes[j].sum()

        # compute the ratios
        ratios = changes / np.tile(self.products.shares, self.J)
        return ratios, []

    def extract_diagonal(self, matrix: Array) -> Tuple[Array, List[Error]]:
        """Extract the diagonal from a matrix."""
        diagonal = matrix[:, :self.J].diagonal()
        return diagonal, []

    def extract_diagonal_mean(self, matrix: Array) -> Tuple[Array, List[Error]]:
        """Extract the mean of the diagonal from a matrix."""
        diagonal_mean = matrix[:, :self.J].diagonal().mean()
        return diagonal_mean, []

    def compute_costs(self) -> Tuple[Array, List[Error]]:
        """Estimate marginal costs."""
        eta, errors = self.compute_eta()
        costs = self.products.prices - eta
        return costs, errors

    def compute_approximate_prices(
            self, firms_index: int = 0, costs: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Estimate approximate equilibrium prices under the assumption that shares and their price derivatives are
        unaffected by firm ID changes. By default, use unchanged firm IDs and compute marginal costs.
        """
        errors: List[Error] = []
        if costs is None:
            costs, errors = self.compute_costs()
        ownership_matrix = self.get_ownership_matrix(firms_index)
        eta, eta_errors = self.compute_eta(ownership_matrix)
        errors.extend(eta_errors)
        prices = costs + eta
        return prices, errors

    def compute_prices(
            self, iteration: Iteration, firms_index: int = 0, prices: Optional[Array] = None,
            costs: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Estimate equilibrium prices. By default, use unchanged firm IDs, use unchanged prices as starting values,
        and compute marginal costs.
        """
        errors: List[Error] = []
        if costs is None:
            costs, errors = self.compute_costs()

        # configure NumPy to identify floating point errors
        with np.errstate(divide='call', over='call', under='ignore', invalid='call'):
            np.seterrcall(lambda *_: errors.append(exceptions.EquilibriumPricesFloatingPointError()))

            # compute equilibrium prices
            prices, converged = self.compute_equilibrium_prices(costs, iteration, firms_index, prices)[:2]
            if not converged:
                errors.append(exceptions.EquilibriumPricesConvergenceError())
            return prices, errors

    def compute_shares(self, prices: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Estimate shares evaluated at specified prices. By default, use unchanged prices."""
        if prices is None:
            prices = self.products.prices
        delta = self.update_delta_with_variable('prices', prices)
        mu = self.update_mu_with_variable('prices', prices)
        shares = self.compute_probabilities(delta, mu) @ self.agents.weights
        return shares, []

    def compute_hhi(self, firms_index: int = 0, shares: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Estimate HHI. By default, use unchanged firm IDs and shares."""
        if shares is None:
            shares = self.products.shares
        firm_ids = self.products.firm_ids[:, [firms_index]]
        hhi = 1e4 * sum((shares[firm_ids == f].sum() / shares.sum())**2 for f in np.unique(firm_ids))
        return hhi, []

    def compute_markups(
            self, prices: Optional[Array] = None, costs: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Estimate markups. By default, use unchanged prices and compute marginal costs."""
        errors: List[Error] = []
        if prices is None:
            prices = self.products.prices
        if costs is None:
            costs, errors = self.compute_costs()
        markups = (prices - costs) / prices
        return markups, errors

    def compute_profits(
            self, prices: Optional[Array] = None, shares: Optional[Array] = None, costs: Optional[Array] = None) -> (
            Tuple[Array, List[Error]]):
        """Estimate population-normalized gross expected profits. By default, use unchanged prices, use unchanged
        shares, and compute marginal costs.
        """
        errors: List[Error] = []
        if prices is None:
            prices = self.products.prices
        if shares is None:
            shares = self.products.shares
        if costs is None:
            costs, errors = self.compute_costs()
        profits = (prices - costs) * shares
        return profits, errors

    def compute_consumer_surplus(self, prices: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Estimate population-normalized consumer surplus. By default, use unchanged prices."""
        if prices is None:
            delta = self.delta
            mu = self.mu
        else:
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
        if self.K2 == 0:
            mu = 0

        # compute the exponentiated utilities that will be summed in the expression for consume surplus
        exp_utilities = np.exp(delta + mu)
        if self.H > 0:
            exp_utilities = self.groups.sum(exp_utilities**(1 / (1 - self.rho)))**(1 - self.group_rho)

        # compute the derivatives of utility with respect to prices, which are assumed to be constant across products
        alpha = -self.compute_utility_derivatives('prices')[0]

        # compute consumer surplus
        consumer_surplus = (np.log1p(exp_utilities.sum(axis=0)) / alpha) @ self.agents.weights
        return consumer_surplus, []
