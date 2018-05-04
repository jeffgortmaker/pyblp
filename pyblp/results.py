"""Structuring of BLP problem results and computation of post-estimation outputs."""

import time

import numpy as np
import scipy.linalg

from .primitives import Market
from . import options, exceptions
from .utilities import output, ParallelItems, Iteration


class Results(object):
    r"""Results of a solved BLP problem.

    Many results are class attributes. Other post-estimation outputs be computed by calling class methods.

    Attributes
    ----------
    problem : `Problem`
        The BLP :class:`Problem` that created these results.
    last_results : `Results`
        The :class:`Results` from the last GMM step if the GMM step that created these results was not the first.
    step : `int`
        The GMM step that created these results.
    optimization_time : `float`
        The number of seconds it took the optimization routine to finish during the GMM step that created these results.
    total_time : `float`
        Sum of :attr:`Results.optimization_time` and the number of seconds it took to compute results after
        optimization had finished during the GMM step that created these results.
    objective_evaluations : `int`
        The number of times the GMM objective was evaluated during the GMM step that created these results.
    contraction_evaluations : `ndarray`
        For each objective evaluation during the GMM step that created these results, the total number of times across
        all markets the contraction used to compute :math:`\delta(\hat{\theta})` was evaluated.
    cumulative_optimization_time : `float`
        Sum of :attr:`Results.optimization_time` for this step and all prior steps.
    cumulative_total_time : `float`
        Sum of :attr:`Results.total_time` for this step and all prior steps.
    cumulative_objective_evaluations : `float`
        Sum of :attr:`Results.objective_evaluations` for this step and all prior steps.
    cumulative_contraction_evaluations : `float`
        Concatenation of :attr:`Results.contraction_evaluations` for this step and all prior steps.
    theta : `ndarray`
        Estimated unknown nonlinear parameters, :math:`\hat{\theta}`.
    sigma : `ndarray`
        Estimated Cholesky decomposition of the covariance matrix that measures agents' random taste distribution,
        :math:`\hat{\Sigma}`. If `nonlinear_prices` in :class:`Problem` initialization was ``True``, the first row and
        column correspond to prices, and if `product_data` contained a `nonlinear_characteristics` field, all other rows
        and columns correspond to its columns.
    pi : `ndarray`
        Estimated parameters that measures how agent tastes vary with demographics, :math:`\hat{\Pi}`. Rows correspond
        to the same product characteristics as in `sigma`. Columns correspond to the columns of the `demographics` field
        of `agent_data` in :class:`Problem` initialization.
    beta : `ndarray`
        Estimated demand-side linear parameters, :math:`\hat{\beta}`. The first element corresponds to prices and if
        `product_data` in :class:`Problem` initialization contained a `nonlinear_characteristics` field, all other
        elements correspond to its columns.
    gamma : `ndarray`
        Estimated supply-side linear parameters, :math:`\hat{\gamma}`, which are ``None`` if the problem that created
        these results was not initialized with supply-side data. Elements correspond to columns in the
        `cost_characteristics` field of `product_data` in :class:`Problem` initialization.
    sigma_se : `ndarray`
        Estimated standard errors for unknown :math:`\hat{\Sigma}` elements in :math:`\hat{\theta}`.
    pi_se : `ndarray`
        Estimated standard errors for unknown :math:`\hat{\Pi}` elements in :math:`\hat{\theta}`, which are ``None`` if
        the problem that created these results was not initialized with demographics.
    beta_se : `ndarray`
        Estimated standard errors for :math:`\hat{\beta}`.
    gamma_se : `ndarray`
        Estimated standard errors for :math:`\hat{\gamma}`.
    objective : `float`
        GMM objective value.
    jacobian : `ndarray`
        Estimated :math:`\partial\delta / \partial\theta`.
    gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to :math:`\theta`.
    sigma_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to unknown :math:`\Sigma` elements in :math:`\theta`.
    pi_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to unknown :math:`\Pi` elements in :math:`\theta`.
    delta : `ndarray`
        Estimated mean utility, :math:`\delta(\hat{\theta})`.
    xi : `ndarray`
        Estimated unobserved demand-side product characteristics, :math:`\xi(\hat{\theta})`.
    omega : `ndarray`
        Estimated unobserved supply-side product characteristics, :math:`\omega(\hat{\theta})`, which are ``None`` if
        the problem that created these results was not initialized with supply-side data.
    WD : `ndarray`
        Demand-side weighting matrix, :math:`W_D`, used to compute these results.
    WS : `ndarray`
        Supply-side weighting matrix, :math:`W_S`, used to compute these results, which is ``None`` if the problem that
        created these results was not initialized with supply-side data.
    updated_WD : `ndarray`
        Updated demand-side weighting matrix, :math:`(Z_D'\xi(\hat{\theta})\xi(\hat{\theta})'Z_D)^{-1}`.
    updated_WS : `ndarray`
        Updated supply-side weighting matrix, :math:`(Z_S'\omega(\hat{\theta})\omega(\hat{\theta})'Z_S)^{-1}`, which
        is ``None`` if the problem that created these results was not initialized with supply-side data.
    unique_market_ids : `ndarray`
        Unique market IDs, which are in the same order as post-estimation outputs returned by methods that compute a
        single value for each market.

    """

    def __init__(self, objective_info, last_results, start_time, end_time, objective_evaluations, contraction_evaluations,
                 center_moments, se_type):
        """Compute estimated standard errors and update weighting matrices."""

        # initialize values from the objective information
        self._parameter_info = objective_info.parameter_info
        self.problem = objective_info.problem
        self.WD = objective_info.WD
        self.WS = objective_info.WS
        self.theta = objective_info.theta
        self.delta = objective_info.delta
        self.jacobian = objective_info.jacobian
        self.beta = objective_info.beta
        self.gamma = objective_info.gamma
        self.xi = objective_info.xi
        self.omega = objective_info.omega
        self.objective = objective_info.objective
        self.gradient = objective_info.gradient

        # initialize counts and times (the total time will be added to at the end of initialization)
        self.step = 1
        self.total_time = self.cumulative_total_time = 0
        self.optimization_time = self.cumulative_optimization_time = end_time - start_time
        self.objective_evaluations = self.cumulative_objective_evaluations = objective_evaluations
        self.contraction_evaluations = self.cumulative_contraction_evaluations = np.asarray(contraction_evaluations)

        # initialize last results and add to cumulative values
        self.last_results = last_results
        if last_results is not None:
            self.step += 1
            self.cumulative_total_time += last_results.cumulative_total_time
            self.cumulative_optimization_time += last_results.cumulative_optimization_time
            self.cumulative_objective_evaluations += last_results.cumulative_objective_evaluations
            self.cumulative_contraction_evaluations = np.r_[
                last_results.cumulative_contraction_evaluations,
                self.cumulative_contraction_evaluations
            ]

        # construct an array of unique and sorted market IDs
        self.unique_market_ids = np.unique(self.problem.products.market_ids).flatten()

        # expand the nonlinear parameters and their gradient
        self.sigma, self.pi = self._parameter_info.expand(self.theta, fill_fixed=True)
        self.sigma_gradient, self.pi_gradient = self._parameter_info.expand(self.gradient)

        # compute demand-side standard errors and update the demand-side weighting matrix
        GD = self.problem.products.ZD.T @ np.c_[self.problem.products.X1, self.jacobian]
        demand_se = self._compute_se(self.xi, self.problem.products.ZD, self.WD, GD, se_type, "demand")
        self.beta_se = demand_se[:self.beta.size]
        self.sigma_se, self.pi_se = self._parameter_info.expand(demand_se[self.beta.size:])
        self.updated_WD = self._compute_W(self.xi, self.problem.products.ZD, center_moments, "demand")

        # compute supply-side standard errors and update the supply-side weighting matrix
        self.gamma_se = self.updated_WS = None
        if self.omega is not None:
            GS = self.problem.products.ZS.T @ self.problem.products.X3
            self.gamma_se = self._compute_se(self.omega, self.problem.products.ZS, self.WS, GS, se_type, "supply")
            self.updated_WS = self._compute_W(self.omega, self.problem.products.ZS, center_moments, "supply")

        # add the time it took to compute results for this step to total times
        self.total_time = time.time() - start_time
        self.cumulative_total_time += self.total_time

    def __str__(self):
        """Format full results as a string."""
        sections = []

        # construct a table of values
        header1 = ["GMM", "Objective", "Total Contraction", "Objective", "Largest Gradient"]
        header2 = ["Step", "Evaluations", "Evaluations", "Value", "Magnitude"]
        widths = [max(len(k1), len(k2)) for k1, k2 in list(zip(header1, header2))[:3]]
        widths.extend([max(len(k1), len(k2), options.digits + 6) for k1, k2 in list(zip(header1, header2))[3:]])
        formatter = output.table_formatter(widths)
        sections.append([
            formatter(header1),
            formatter(header2),
            formatter.lines(),
            formatter([
                self.step,
                self.objective_evaluations,
                self.contraction_evaluations.sum(),
                output.format_number(self.objective),
                output.format_number(np.abs(self.gradient).max())
            ])
        ])

        # construct a table of linear estimates
        linear_header = ["Beta:", "Price"] + [f"Linear #{i}" for i in range(self.beta.size - 1)]
        linear_formatter = output.table_formatter([14] + [max(len(k), options.digits + 8) for k in linear_header[1:]])
        sections.append([
            "Linear Parameter Estimates (SEs in Parentheses)",
            linear_formatter.border(),
            linear_formatter(linear_header),
            linear_formatter.lines(),
            linear_formatter([""] + [output.format_number(x) for x in self.beta]),
            linear_formatter([""] + [output.format_se(x) for x in self.beta_se]),
            linear_formatter.border()
        ])

        # construct a table of cost estimates
        if self.gamma is not None:
            cost_header = ["Gamma:"] + [f"Cost #{i}" for i in range(self.gamma.size)]
            cost_formatter = output.table_formatter([14] + [max(len(k), options.digits + 8) for k in cost_header[1:]])
            sections.append([
                "Cost Parameter Estimates (SEs in Parentheses)",
                cost_formatter.border(),
                cost_formatter(cost_header),
                cost_formatter.lines(),
                cost_formatter([""] + [output.format_number(x) for x in self.gamma]),
                cost_formatter([""] + [output.format_se(x) for x in self.gamma_se]),
                cost_formatter.border()
            ])

        # construct a section containing nonlinear estimates
        sections.append([
            "Nonlinear Parameter Estimates (SEs in Parentheses)",
            self._parameter_info.format_matrices(self.sigma, self.pi, self.sigma_se, self.pi_se)
        ])

        # combine the sections into one string
        return "\n\n".join("\n".join(s) for s in sections)

    def __repr__(self):
        """The representation of a class instance is also its string representation."""
        return str(self)

    def _compute_W(self, u, Z, center_moments, side):
        """Update a GMM weighting matrix."""
        g = u * Z

        # center the moments
        if center_moments:
            g -= g.mean(axis=0)

        # attempt to compute the weighting matrix and output information about the method used to compute it
        attempt, W = self._invert(g.T @ g)
        if attempt > 1:
            output("")
            output(f"The estimated covariance matrix of {side}-side GMM moments is singular.")
            if attempt == 2:
                output(f"Used the Moore-Penrose pseudo inverse to compute the {side}-side GMM weighting matrix.")
            else:
                output("Failed to compute the Moore-Penrose pseudo-inverse.")
                output(f"Inverted only the variance terms to compute the {side}-side GMM weighting matrix.")
            output("")

        # output any information about computation failure
        if np.isnan(W).any():
            output("")
            output(f"Failed to compute the {side}-side GMM weighting matrix because of null values.")
            output("")
        return W

    def _compute_se(self, u, Z, W, G, se_type, side):
        """Compute estimated standard errors."""

        # attempt to compute the unadjusted covariance matrix and output information about the method used to compute it
        attempt, covariances = self._invert(G.T @ W @ G)
        if attempt > 1:
            output("")
            output(f"The estimated covariance matrix of {side}-side GMM moments is singular.")
            if attempt == 2:
                output(f"Used the Moore-Penrose pseudo inverse to compute the {side}-side GMM weighting matrix.")
            else:
                output("Failed to compute the Moore-Penrose pseudo-inverse.")
                output(f"Inverted only the variance terms to compute the {side}-side GMM weighting matrix.")
            output("")

        # compute the robust covariance matrix
        with np.errstate(invalid='ignore'):
            if se_type == 'robust':
                covariances = covariances @ G.T @ W @ Z.T @ (np.diagflat(u) ** 2) @ Z @ W @ G @ covariances
            se = np.sqrt(np.c_[covariances.diagonal()])

        # output any information about computation failure
        if np.isnan(se).any():
            output("")
            output(f"Failed to compute {side}-side standard errors because of null values.")
            output("")
        return se

    @staticmethod
    def _invert(matrix):
        """Attempt to invert a matrix with decreasingly precise inversion functions. The first attempt is with a
        standard inversion function; the second, with the Moore-Penrose pseudo inverse; the third, with simple diagonal
        inversion.
        """
        try:
            return 1, scipy.linalg.inv(matrix)
        except ValueError:
            return 1, np.full_like(matrix, np.nan)
        except scipy.linalg.LinAlgError:
            try:
                return 2, scipy.linalg.pinv(matrix)
            except scipy.linalg.LinAlgError:
                return 3, np.diag(1 / np.diag(matrix))

    def _convert_indices(self, linear_index=None, nonlinear_index=None):
        """Identify the column indices in X1 and X2 that are associated with column indices in the matrices of non-price
        characteristics used to construct X1 and X2.
        """
        if linear_index is None and nonlinear_index is None:
            raise ValueError("linear_index and nonlinear_index cannot both be None.")

        # validate the indices
        linear_indices = frozenset(range(self.problem.K1 - 1))
        nonlinear_indices = frozenset(range(self.problem.K2 - int(self.problem.nonlinear_prices)))
        if linear_index is not None:
            if not linear_indices:
                raise ValueError("Price is the only linear characteristic, so linear_index must be None.")
            if linear_index not in linear_indices:
                raise ValueError(f"linear_index must be None or one of {list(linear_indices)}.")
        if nonlinear_index is not None:
            if not nonlinear_indices:
                raise ValueError("There are no non-price nonlinear characteristics, so nonlinear_index must be None.")
            if nonlinear_index not in nonlinear_indices:
                raise ValueError(f"nonlinear_index must be None or one of {list(nonlinear_indices)}.")

        # determine the corresponding indices in X1 and X2
        X1_index = None if linear_index is None else linear_index + 1
        X2_index = None if nonlinear_index is None else nonlinear_index + int(self.problem.nonlinear_prices)

        # if the characteristic is in both X1 and X2, make sure that it is identical in both
        if X1_index is not None and X2_index is not None:
            if not np.array_equal(self.problem.products.X1[:, X1_index], self.problem.products.X2[:, X2_index]):
                raise ValueError("linear_index and nonlinear_index must correspond to the same characteristic.")
        return X1_index, X2_index

    def _combine_results(self, compute_market_results, fixed_args=(), market_args=(), processes=1):
        """Compute post-estimation outputs for each market and stack them into a single matrix. Multiprocessing can be
        used to compute outputs in parallel.

        An output for a single market is computed by passing fixed_args (identical for all markets) and market_args
        (matrices with as many rows as there are products that are restricted to the market) to compute_market_results,
        a ResultsMarket method that returns the output for the market and a set of any errors encountered during
        computation.
        """

        # keep track of how long it takes to compute results
        start_time = time.time()

        # construct a mapping from market IDs to market-specific arguments used to compute results
        args_mapping = {}
        for t in self.unique_market_ids:
            market_t = ResultsMarket(
                t, self.problem.nonlinear_prices, self.problem.products, self.problem.agents, self.delta, self.xi,
                self.beta, self.sigma, self.pi
            )
            args_t = [None if a is None else a[self.problem.products.market_ids.flat == t] for a in market_args]
            args_mapping[t] = [market_t] + list(fixed_args) + args_t

        # construct a mapping from market IDs to market-specific results and compute the full results matrix size
        errors = set()
        rows = columns = 0
        matrix_mapping = {}
        with ParallelItems(compute_market_results, args_mapping, processes) as items:
            for t, (array_t, errors_t) in items:
                errors |= errors_t
                matrix_mapping[t] = np.c_[array_t]
                rows += matrix_mapping[t].shape[0]
                columns = max(columns, matrix_mapping[t].shape[1])

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # preserve the original product order or the sorted market order when stacking the matrices
        combined = np.full((rows, columns), np.nan, dtype=options.dtype)
        market_ids = self.problem.products.market_ids.flat if rows == self.problem.N else self.unique_market_ids
        for t, matrix_t in matrix_mapping.items():
            combined[market_ids == t, :matrix_t.shape[1]] = matrix_t

        # output how long it took to compute results
        end_time = time.time()
        output(f"Finished after {output.format_seconds(end_time - start_time)}.")
        return combined

    def compute_aggregate_elasticities(self, linear_index=None, nonlinear_index=None, factor=0.1):
        r"""Estimate aggregate elasticities of demand, :math:`E`, with respect to a product characteristic, :math:`x`.

        In market :math:`t`, the aggregate elasticity of demand is

        .. math:: E = \sum_{j=1}^{J_t} \frac{s_{jt}(x + \Delta x) - s_{jt}}{\Delta},

        in which :math:`\Delta` is a scalar factor and :math:`s_{jt}(x + \Delta x)` is the share of product :math:`j` in
        market :math:`t`, evaluated at the scaled values of the characteristic.

        Parameters
        ----------
        linear_index : `int, optional`
            Column index of :math:`x` in the `linear_characteristics` field of `product_data` in :class:`Problem`
            initialization.
        nonlinear_index : `int, optional`
            Column index of :math:`x` in the `nonlinear_characteristics` field.
        factor : `float, optional`
            The scalar factor, :math:`\Delta`.

        Returns
        -------
        `ndarray`
            Estimates of aggregate elasticities of demand, :math:`E`, for all markets. Rows are in the same order as
            :attr:`Results.unique_market_ids`.

        """
        output("Computing aggregate elasticities ...")
        output(f"Factor: {output.format_number(factor)}.")
        X1_index, X2_index = self._convert_indices(linear_index, nonlinear_index)
        return self._combine_results(ResultsMarket.compute_aggregate_elasticity, [X1_index, X2_index, factor])

    def compute_aggregate_price_elasticities(self, factor=0.1):
        """Estimate aggregate elasticities of demand, :math:`E`, with respect to prices, :math:`p`.

        Identical to :meth:`Results.compute_aggregate_elasticities` for :math:`x = p`.

        """
        output("Computing aggregate price elasticities ....")
        return self._combine_results(ResultsMarket.compute_aggregate_price_elasticity, [factor])

    def compute_elasticities(self, linear_index=None, nonlinear_index=None):
        r"""Estimate matrices of elasticities of demand, :math:`\varepsilon`, with respect to a product characteristic,
        :math:`x`.

        For each market, the value in row :math:`j` and column :math:`k` of :math:`\varepsilon` is

        .. math:: \varepsilon_{jk} = \frac{x_k}{s_j}\frac{\partial s_j}{\partial x_k}.

        Parameters
        ----------
        linear_index : `int, optional`
            Column index of :math:`x` in the `linear_characteristics` field of `product_data` in :class:`Problem`
            initialization.
        nonlinear_index : `int, optional`
            Column index of :math:`x` in the `nonlinear_characteristics` field.

        Returns
        -------
        `ndarray`
            Stacked :math:`J_t \times J_t` estimated matrices of elasticities of demand, :math:`\varepsilon`, for each
            market :math:`t`. Columns for a market are in the same order as products for the market. If a market has
            fewer products than others, extra columns will contain ``numpy.nan``.

        """
        output("Computing elasticities ...")
        X1_index, X2_index = self._convert_indices(linear_index, nonlinear_index)
        return self._combine_results(ResultsMarket.compute_elasticities, [X1_index, X2_index])

    def compute_price_elasticities(self):
        r"""Estimate matrices of elasticities of demand, :math:`\varepsilon`, with respect to prices, :math:`p`.

        Identical to :meth:`Results.compute_elasticities` for :math:`x = p`.

        """
        output("Computing price elasticities ....")
        return self._combine_results(ResultsMarket.compute_price_elasticities)

    def compute_diversion_ratios(self, linear_index=None, nonlinear_index=None):
        r"""Estimate matrices of diversion ratios, :math:`\mathscr{D}`, with respect to a product characteristic,
        :math:`x`.

        Diversion ratios to the outside good are reported on diagonals. For each market, the value in row :math:`j` and
        column :math:`k` is

        .. math:: \mathscr{D}_{jk} = -\frac{\partial s_{k(j)} / \partial x_j}{\partial s_j / \partial x_j},

        in which :math:`s_{k(j)}` is :math:`s_0 = 1 - \sum_j s_j` if :math:`j = k`, and is :math:`s_k` otherwise.

        Parameters
        ----------
        linear_index : `int, optional`
            Column index of :math:`x` in the `linear_characteristics` field of `product_data` in :class:`Problem`
            initialization.
        nonlinear_index : `int, optional`
            Column index of :math:`x` in the `nonlinear_characteristics` field.

        Returns
        -------
        `ndarray`
            Stacked :math:`J_t \times J_t` estimated matrices of diversion ratios, :math:`\mathscr{D}`, for all markets.
            Columns for a market are in the same order as products for the market. If a market has fewer products than
            others, extra columns will contain ``numpy.nan``.

        """
        output("Computing diversion ratios ...")
        X1_index, X2_index = self._convert_indices(linear_index, nonlinear_index)
        return self._combine_results(ResultsMarket.compute_diversion_ratios, [X1_index, X2_index])

    def compute_price_diversion_ratios(self):
        r"""Estimate matrices of diversion ratios, :math:`\mathscr{D}`, with respect to prices, :math:`p`.

        Identical to :meth:`Results.compute_diversion_ratios` for :math:`x = p`.

        """
        output("Computing aggregate price elasticities ....")
        return self._combine_results(ResultsMarket.compute_price_diversion_ratios)

    def compute_long_run_diversion_ratios(self):
        r"""Estimate matrices of long-run diversion ratios, :math:`\bar{\mathscr{D}}`.

        Long-run diversion ratios to the outside good are reported on diagonals. For each market, the value in row
        :math:`j` and column :math:`k` is

        .. math:: \bar{\mathscr{D}}_{jk} = \frac{s_{k(-j)} - s_k}{s_j},

        in which :math:`s_{k(-j)}` is the share of product :math:`k` computed with the outside option removed from the
        choice set if :math:`j = k`, and with product :math:`j` removed otherwise.

        Returns
        -------
        `ndarray`
            Stacked :math:`J_t \times J_t` estimated matrices of long-run diversion ratios, :math:`\bar{\mathscr{D}}`,
            for all markets. Columns for a market are in the same order as products for the market. If a market has
            fewer products than others, extra columns will contain ``numpy.nan``.

        """
        output("Computing long run mean diversion ratios ...")
        return self._combine_results(ResultsMarket.compute_long_run_diversion_ratios)

    def extract_diagonals(self, matrices):
        r"""Extract diagonals from stacked :math:`J_t \times J_t` matrices for each market :math:`t`.

        Parameters
        ----------
        matrices : `array-like`
            Stacked matrices, such as estimates of :math:`\varepsilon`, computed by :meth:`Results.compute_elasticities`
            or :meth:`Results.compute_price_elasticities`; :math:`\mathscr{D}`, computed by
            :meth:`Results.compute_diversion_ratios` or :meth:`Results.compute_price_diversion_ratios`; or
            :math:`\bar{\mathscr{D}}`, computed by :meth:`Results.compute_long_run_diversion_ratios`.

        Returns
        -------
        `ndarray`
            Stacked diagonals for all markets. If the matrices are estimates of :math:`\varepsilon`, a diagonal is a
            market's own elasticities of demand; if they are estimates of :math:`\mathscr{D}` or
            :math:`\bar{\mathscr{D}}`, a diagonal is a market's diversion ratios to the outside good.

        """
        output("Computing own elasticities ...")
        return self._combine_results(ResultsMarket.extract_diagonal, market_args=[matrices])

    def extract_diagonal_means(self, matrices):
        r"""Extract means of diagonals from stacked :math:`J_t \times J_t` matrices for each market :math:`t`.

        Parameters
        ----------
        matrices : `array-like`
            Stacked matrices, such as estimates of :math:`\varepsilon`, computed by :meth:`Results.compute_elasticities`
            or :meth:`Results.compute_price_elasticities`; :math:`\mathscr{D}`, computed by
            :meth:`Results.compute_diversion_ratios` or :meth:`Results.compute_price_diversion_ratios`; or
            :math:`\bar{\mathscr{D}}`, computed by :meth:`Results.compute_long_run_diversion_ratios`.

        Returns
        -------
        `ndarray`
            Stacked means of diagonals for all markets. If the matrices are estimates of :math:`\varepsilon`, the mean
            of a diagonal is a market's mean own elasticity of demand; if they are estimates of :math:`\mathscr{D}` or
            :math:`\bar{\mathscr{D}}`, the mean of a diagonal is a market's mean diversion ratio to the outside good.
            Rows are in the same order as :attr:`Results.unique_market_ids`.

        """
        output("Computing mean own elasticities ...")
        return self._combine_results(ResultsMarket.extract_diagonal_mean, market_args=[matrices])

    def compute_costs(self):
        r"""Estimate marginal costs, :math:`c`.

        Marginal costs are computed with the BLP-markup equation,

        .. math:: c = p - \eta.

        Returns
        -------
        `ndarray`
            Marginal costs, :math:`c`.

        """
        output("Computing marginal costs ...")
        return self._combine_results(ResultsMarket.compute_costs)

    def solve_approximate_merger(self, costs, firms_index=1):
        r"""Estimate approximate post-merger prices, :math:`p^a`, under the assumption that shares and their price
        derivatives are unaffected by the merger.

        This approximation is discussed in, for example, :ref:`Nevo (1997) <n97>`. Prices in each market are computed
        according to the approximate post-merger version of the BLP-markup equation,

        .. math:: p^a = c + \eta^a,

        in which the approximate markup term is

        .. math:: \eta^a = -\left(O^* \odot \frac{\partial s}{\partial p}\right)^{-1}s

        where :math:`O^*` is the post-merger ownership matrix.

        Parameters
        ----------
        costs : `array-like`
            Marginal costs, :math:`c`, computed by :meth:`Results.compute_costs`.
        firms_index : `int, optional`
            Column index of the changed firm IDs in the `firm_ids` field of `product_data` in :class:`Problem`
            initialization. If an `ownership` field was specified, the corresponding stack of ownership matrices will be
            used. Ownership changes need not reflect an actual merger.

        Returns
        -------
        `ndarray`
            Estimates of approximate post-merger prices, :math:`p^a`.

        """
        output("Solving for approximate post-merger prices ...")
        output(f"Firms index: {firms_index}.")
        return self._combine_results(ResultsMarket.solve_approximate_merger, [firms_index], [costs])

    def solve_merger(self, costs, firms_index=1, prices=None, iteration=None, processes=1):
        r"""Estimate post-merger prices, :math:`p^*`.

        Prices are computed in each market by iterating over the post-merger version of the :math:`\zeta`-markup
        equation from :ref:`Morrow and Skerlos (2011) <ms11>`,

        .. math:: p^* \leftarrow c + \zeta^*(p^*),

        in which the markup term is

        .. math:: \zeta^*(p^*) = \Lambda^{-1}(p^*)[O^* \odot \Gamma(p^*)]'(p^* - c) - \Lambda^{-1}(p^*)

        where :math:`O^*` is the post-merger ownership matrix and the other terms are the same as in the pre-merger
        :math:`\zeta`-markup equation but are evaluated at post-merger prices and shares.

        Parameters
        ----------
        costs : `array-like`
            Marginal costs, :math:`c`, computed by :meth:`Results.compute_costs`.
        firms_index : `int, optional`
            Column index of the changed firm IDs in the `firm_ids` field of `product_data` in :class:`Problem`
            initialization. If an `ownership` field was specified, the corresponding stack of ownership matrices will be
            used. Ownership changes need not reflect an actual merger.
        prices : `array-like, optional`
            Prices at which the fixed point iteration routine will start. By default, pre-merger prices, :math:`p`, are
            used as starting values. Other reasonable starting prices include :math:`p^a`, computed by
            :meth:`Results.solve_approximate_merger`.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem in each market. By default,
            ``Iteration('simple')`` is used.
        processes : `int, optional`
            The number of Python processes that will be used during computation. By default, multiprocessing will not be
            used. For values greater than one, a pool of that many Python processes will be created. Market-by-market
            computation of post-merger prices will be distributed among these processes. Using multiprocessing will only
            improve computation speed if gains from parallelization outweigh overhead from creating the process pool.

        Returns
        -------
        `ndarray`
            Estimates of post-merger prices, :math:`p^*`.

        """
        if iteration is None:
            iteration = Iteration('simple')
        elif not isinstance(iteration, Iteration):
            raise ValueError("iteration must an Iteration instance.")

        output("Solving for post-merger prices ...")
        output(f"Firms index: {firms_index}.")
        output("Starting with unchanged prices." if prices is None else "Starting with the specified prices.")
        output(iteration)
        output(f"Processes: {processes}.")
        return self._combine_results(ResultsMarket.solve_merger, [firms_index, iteration], [costs, prices], processes)

    def compute_shares(self, prices):
        r"""Estimate shares evaluated at specified prices.

        Parameters
        ----------
        prices : `array-like`
            Prices at which to evaluate shares, such as post-merger prices, :math:`p^*`, computed by
            :meth:`Results.solve_merger`, or approximate post-merger prices, :math:`p^a`, computed by
            :meth:`Results.solve_approximate_merger`.

        Returns
        -------
        `ndarray`
            Estimates of shares evaluated at the specified prices.

        """
        output("Computing shares ...")
        return self._combine_results(ResultsMarket.compute_shares, market_args=[prices])

    def compute_hhi(self, shares=None, firms_index=0):
        r"""Estimate Herfindahl-Hirschman Indices, :math:`\text{HHI}`.

        The index in market :math:`t` is

        .. math:: \text{HHI} = 10,000 \times \sum_{f=1}^{F_t} \left(\sum_{j \in \mathscr{J}_{ft}} s_j\right)^2,

        in which :math:`\mathscr{J}_{ft}` is the set of products produced by firm :math:`f` in market :math:`t`.

        Parameters
        ----------
        shares : `array-like, optional`
            Shares, :math:`s`, such as those computed by :meth:`Results.compute_shares`. By default, unchanged shares
            are used.
        firms_index : `int, optional`
            Column index of the firm IDs in the `firm_ids` field of `product_data` in :class:`Problem` initialization.
            By default, unchanged firm IDs are used.

        Returns
        -------
        `ndarray`
            Estimated Herfindahl-Hirschman Indices, :math:`\text{HHI}`, for all markets. Rows are in the same order as
            :attr:`Results.unique_market_ids`.

        """
        output("Computing HHI ...")
        output("Using unchanged shares" if shares is None else "Using the specified shares.")
        output(f"Firms index: {firms_index}.")
        return self._combine_results(ResultsMarket.compute_hhi, [firms_index], [shares])

    def compute_markups(self, costs, prices=None):
        r"""Estimate markups, :math:`\mathscr{M}`.

        The markup of product :math:`j` in market :math:`t` is

        .. math:: \mathscr{M}_{jt} = \frac{p_{jt} - c_{jt}}{p_{jt}}.

        Parameters
        ----------
        costs : `array-like`
            Marginal costs, :math:`c`, computed by :meth:`Results.compute_costs`.
        prices : `array-like, optional`
            Prices, :math:`p`, such as post-merger prices, :math:`p^*`, computed by :meth:`Results.solve_merger`, or
            approximate post-merger prices, :math:`p^a`, computed by :meth:`Results.solve_approximate_merger`. By
            default, unchanged prices are used.

        Returns
        -------
        `ndarray`
            Estimated markups, :math:`\mathscr{M}`.

        """
        output("Computing markups ...")
        output("Using unchanged prices." if prices is None else "Using the specified changed prices.")
        return self._combine_results(ResultsMarket.compute_markups, market_args=[costs, prices])

    def compute_profits(self, costs, prices=None, shares=None):
        r"""Estimate population-normalized gross expected profits, :math:`\pi`.

        The profit of product :math:`j` in market :math:`t` is

        .. math:: \pi_{jt} = p_{jt} - c_{jt}s_{jt}.

        Parameters
        ----------
        costs : `array-like`
            Marginal costs, :math:`c`, computed by `compute_costs`.
        prices : `array-like, optional`
            Prices, :math:`p`, such as post-merger prices, :math:`p^*`, computed by :meth:`Results.solve_merger`, or
            approximate post-merger prices, :math:`p^a`, computed by :meth:`Results.solve_approximate_merger`. By
            default, unchanged prices are used.
        shares : `array-like, optional`
            Shares, :math:`s`, such as those computed by :meth:`Results.compute_shares`. By default, unchanged shares
            are used.

        Returns
        -------
        `ndarray`
            Estimated population-normalized gross expected profits, :math:`\pi`.

        """
        output("Computing profits ...")
        output("Using unchanged prices." if prices is None else "Using the specified changed prices.")
        output("Using unchanged shares." if shares is None else "Using the specified changed shares.")
        return self._combine_results(ResultsMarket.compute_profits, market_args=[costs, prices, shares])

    def compute_consumer_surpluses(self, prices=None):
        r"""Estimate population-normalized consumer surpluses, :math:`\text{CS}`.

        The surplus in market :math:`t` is

        .. math:: \text{CS} = \sum_{i=1}^{I_t} w_i\text{CS}_i,

        in which the consumer surplus for individual :math:`i` is

        .. math:: \text{CS}_i = \frac{\log(1 + \sum_{j=1}^{J_t} \exp u_{jti})}{\alpha + \alpha_i}.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices at which utilities, :math:`u`, and price derivatives, :math:`\alpha` and :math:`\alpha_i`, will be
            evaluated, such as post-merger prices, :math:`p^*`, computed by :meth:`Results.solve_merger`, or
            approximate post-merger prices, :math:`p^a`, computed by :meth:`Results.solve_approximate_merger`. By
            default, unchanged prices are used.

        Returns
        -------
        `ndarray`
            Estimated population-normalized consumer surpluses, :math:`\text{CS}`, for all markets. Rows are in the same
            order as :attr:`Results.unique_market_ids`.

        """
        output("Computing consumer surpluses ...")
        output("Using unchanged prices." if prices is None else "Using the specified changed prices.")
        return self._combine_results(ResultsMarket.compute_consumer_surplus, market_args=[prices])


class ResultsMarket(Market):
    """Results for a single market of a solved BLP problem, which can be used to compute post-estimation outputs. Each
    method returns a matrix and a set of any errors that were encountered.
    """

    def compute_aggregate_elasticity(self, X1_index, X2_index, factor):
        """Market-specific computation for Results.compute_aggregate_elasticities."""
        characteristic = (1 + factor) * self.get_characteristic(X1_index, X2_index)
        delta = self.update_delta_with_characteristic(characteristic, X1_index)
        mu = self.update_mu_with_characteristic(characteristic, X2_index)
        shares = self.compute_probabilities(delta, mu) @ self.agents.weights
        aggregate_elasticities = (shares - self.products.shares).sum() / factor
        return aggregate_elasticities, set()

    def compute_aggregate_price_elasticity(self, factor):
        """Market-specific computation for Results.compute_aggregate_price_elasticities."""
        X1_index, X2_index = self.get_price_indices()
        return self.compute_aggregate_elasticity(X1_index, X2_index, factor)

    def compute_elasticities(self, X1_index, X2_index):
        """Market-specific computation for Results.compute_elasticities."""
        utilities_jacobian = self.compute_utilities_by_characteristic_jacobian(X1_index, X2_index)
        shares_jacobian = self.compute_shares_by_characteristic_jacobian(utilities_jacobian)
        characteristic = self.get_characteristic(X1_index, X2_index)
        elasticities = shares_jacobian * np.tile(characteristic, self.J).T / np.tile(self.products.shares, self.J)
        return elasticities, set()

    def compute_price_elasticities(self):
        """Market-specific computation for Results.compute_price_elasticities."""
        X1_index, X2_index = self.get_price_indices()
        return self.compute_elasticities(X1_index, X2_index)

    def compute_diversion_ratios(self, X1_index, X2_index):
        """Market-specific computation for Results.compute_diversion_ratios."""
        utilities_jacobian = self.compute_utilities_by_characteristic_jacobian(X1_index, X2_index)
        shares_jacobian = self.compute_shares_by_characteristic_jacobian(utilities_jacobian)

        # replace the diagonal with derivatives with respect to the outside option
        shares_jacobian_diagonal = np.c_[shares_jacobian.diagonal()]
        shares_jacobian[np.diag_indices_from(shares_jacobian)] = -shares_jacobian.sum(axis=1)

        # compute the ratios
        ratios = -shares_jacobian / np.tile(shares_jacobian_diagonal, self.J)
        return ratios, set()

    def compute_price_diversion_ratios(self):
        """Market-specific computation for Results.compute_price_diversion_ratios."""
        X1_index, X2_index = self.get_price_indices()
        return self.compute_diversion_ratios(X1_index, X2_index)

    def compute_long_run_diversion_ratios(self):
        """Market-specific computation for Results.compute_long_run_diversion_ratios."""

        # compute share differences when products are excluded and store outside share differences on the diagonal
        changes = np.zeros((self.J, self.J), dtype=options.dtype)
        for j in range(self.J):
            shares_without_j = self.compute_probabilities(eliminate_product=j) @ self.agents.weights
            changes[j] = (shares_without_j - self.products.shares).flat
            changes[j, j] = -changes[j].sum()

        # compute the ratios
        ratios = changes / np.tile(self.products.shares, self.J)
        return ratios, set()

    def extract_diagonal(self, matrix):
        """Market-specific computation for Results.extract_diagonals."""
        diagonal = matrix[:, :self.J].diagonal()
        return diagonal, set()

    def extract_diagonal_mean(self, matrix):
        """Market-specific computation for Results.extract_diagonal_means."""
        diagonal_mean = matrix[:, :self.J].diagonal().mean()
        return diagonal_mean, set()

    def compute_costs(self):
        """Market-specific computation for Results.compute_costs."""
        errors = set()
        try:
            costs = super().compute_costs()
        except scipy.linalg.LinAlgError:
            errors.add(exceptions.CostsSingularityError)
            costs = np.full((self.J, 1), np.nan)
        return costs, errors

    def solve_approximate_merger(self, firms_index, costs):
        """Market-specific computation for Results.solve_approximate_merger."""
        jacobian = self.compute_utilities_by_prices_jacobian()
        ownership = self.get_ownership_matrix(firms_index)
        prices = costs + self.compute_eta(ownership, jacobian)
        return prices, set()

    def solve_merger(self, firms_index, iteration, costs, prices=None):
        """Market-specific computation for Results.solve_merger."""

        # configure numpy to identify floating point errors
        errors = set()
        with np.errstate(all='call'):
            np.seterrcall(lambda *_: errors.add(exceptions.ChangedPricesFloatingPointError))

            # solve the fixed point problem
            ownership = self.get_ownership_matrix(firms_index)
            jacobian = self.compute_utilities_by_prices_jacobian()
            contraction = lambda p: costs + self.compute_zeta(ownership, jacobian, costs, p)
            prices, converged = iteration._iterate(contraction, self.products.prices if prices is None else prices)[:2]

        # store whether the fixed point converged
        if not converged:
            errors.add(exceptions.ChangedPricesConvergenceError)
        return prices, errors

    def compute_shares(self, prices):
        """Market-specific computation for Results.compute_shares."""
        delta = self.update_delta_with_prices(prices)
        mu = self.update_mu_with_prices(prices)
        shares = self.compute_probabilities(delta, mu) @ self.agents.weights
        return shares, set()

    def compute_hhi(self, firms_index, shares=None):
        """Market-specific computation for Results.compute_hhi."""
        firm_ids = self.products.firm_ids[:, [firms_index]]
        if shares is None:
            shares = self.products.shares
        hhi = 1e4 * sum((shares[firm_ids == f].sum() / shares.sum()) ** 2 for f in np.unique(firm_ids))
        return hhi, set()

    def compute_markups(self, costs, prices=None):
        """Market-specific computation for Results.compute_markups."""
        if prices is None:
            prices = self.products.prices
        markups = (prices - costs) / prices
        return markups, set()

    def compute_profits(self, costs, prices=None, shares=None):
        """Market-specific computation for Results.compute_profits."""
        if prices is None:
            prices = self.products.prices
        if shares is None:
            shares = self.products.shares
        profits = (prices - costs) * shares
        return profits, set()

    def compute_consumer_surplus(self, prices=None):
        """Market-specific computation for Results.compute_consumer_surpluses."""
        if prices is None:
            prices = self.products.prices
        utilities = np.tile(self.update_delta_with_prices(prices), self.I) + self.update_mu_with_prices(prices)
        alpha = -self.compute_utilities_by_prices_jacobian()[0]
        consumer_surplus = (np.log1p(np.exp(utilities).sum(axis=0)) / alpha) @ self.agents.weights
        return consumer_surplus, set()
