"""Structuring of BLP problem results and computation of post-estimation outputs."""

import time

import numpy as np
import scipy.linalg

from . import options, exceptions
from .primitives import Market, LinearParameters
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
        Number of seconds it took the optimization routine to finish.
    cumulative_optimization_time : `float`
        Sum of :attr:`Results.optimization_time` for this step and all prior steps.
    total_time : `float`
        Sum of :attr:`Results.optimization_time` and the number of seconds it took to compute results after
        optimization had finished.
    cumulative_total_time : `float`
        Sum of :attr:`Results.total_time` for this step and all prior steps.
    optimization_iterations : `int`
        Number of major iterations completed by the optimization routine.
    cumulative_optimization_iterations : `int`
        Sum of :attr:`Results.optimization_iterations` for this step and all prior steps.
    objective_evaluations : `int`
        Number of times the GMM objective was evaluated.
    cumulative_objective_evaluations : `int`
        Sum of :attr:`Results.objective_evaluations` for this step and all prior steps.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute :math:`\delta(\hat{\theta})` in
        each market during each objective evaluation. Rows are in the same order as :attr:`Results.unique_market_ids`
        and column indices correspond to objective evaluations.
    cumulative_fp_iterations : `ndarray`
        Concatenation of :attr:`Results.fp_iterations` for this step and all prior steps.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute :math:`\delta(\hat{\theta})` was evaluated in each market during
        each objective evaluation. Rows are in the same order as :attr:`Results.unique_market_ids` and column indices
        correspond to objective evaluations.
    cumulative_contraction_evaluations : `ndarray`
        Concatenation of :attr:`Results.contraction_evaluations` for this step and all prior steps.
    theta : `ndarray`
        Estimated unknown nonlinear parameters, :math:`\hat{\theta}`.
    sigma : `ndarray`
        Estimated Cholesky decomposition of the covariance matrix that measures agents' random taste distribution,
        :math:`\hat{\Sigma}`.
    pi : `ndarray`
        Estimated parameters that measures how agent tastes vary with demographics, :math:`\hat{\Pi}`.
    beta : `ndarray`
        Estimated demand-side linear parameters, :math:`\hat{\beta}`.
    gamma : `ndarray`
        Estimated supply-side linear parameters, :math:`\hat{\gamma}`, which are ``None`` if the problem that created
        these results was not initialized with supply-side data.
    sigma_se : `ndarray`
        Estimated standard errors for unknown :math:`\hat{\Sigma}` elements in :math:`\hat{\theta}`.
    pi_se : `ndarray`
        Estimated standard errors for unknown :math:`\hat{\Pi}` elements in :math:`\hat{\theta}`.
    beta_se : `ndarray`
        Estimated standard errors for :math:`\hat{\beta}`.
    gamma_se : `ndarray`
        Estimated standard errors for :math:`\hat{\gamma}`.
    delta : `ndarray`
        Estimated mean utility, :math:`\delta(\hat{\theta})`.
    tilde_costs : `ndarray`
        Estimated transformed marginal costs, :math:`\tilde{c}(\hat{\theta})`, which are ``None`` if the problem that
        created these results was not initialized with supply-side data. Transformed marginal costs are simply
        :math:`\tilde{c} = c`, marginal costs, under a linear cost specification, and are :math:`\tilde{c} = \log c`
        under a log-linear specification.
    xi : `ndarray`
        Estimated unobserved demand-side product characteristics, :math:`\xi(\hat{\theta})`, or equivalently, the
        demand-side structural error term.
    omega : `ndarray`
        Estimated unobserved supply-side product characteristics, :math:`\omega(\hat{\theta})`, or equivalently, the
        supply-side structural error term, which is ``None`` if the problem that created these results was not
        initialized with supply-side data.
    objective : `float`
        GMM objective value.
    xi_jacobian : `ndarray`
        Estimated :math:`\partial\xi / \partial\theta = \partial\delta / \partial\theta`.
    omega_jacobian : `ndarray`
        Estimated :math:`\partial\omega / \partial\theta = \partial\tilde{c} / \partial\theta`, which is ``None`` if the
        problem that created these results was not initialized with supply-side data.
    gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to :math:`\theta`.
    gradient_norm : `ndarray`
        Infinity norm of :attr:`Results.gradient`.
    sigma_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to unknown :math:`\Sigma` elements in :math:`\theta`.
    pi_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to unknown :math:`\Pi` elements in :math:`\theta`.
    WD : `ndarray`
        Demand-side weighting matrix, :math:`W_D`, used to compute these results.
    WS : `ndarray`
        Supply-side weighting matrix, :math:`W_S`, used to compute these results, which is ``None`` if the problem that
        created these results was not initialized with supply-side data.
    updated_WD : `ndarray`
        Updated demand-side weighting matrix.
    updated_WS : `ndarray`
        Updated supply-side weighting matrix, which is ``None`` if the problem that created these results was not
        initialized with supply-side data.
    unique_market_ids : `ndarray`
        Unique market IDs, which are in the same order as post-estimation outputs returned by methods that compute a
        single value for each market.

    """

    def __init__(self, objective_info, last_results, start_time, end_time, iterations, evaluations, iteration_mappings,
                 evaluation_mappings, center_moments, se_type):
        """Compute estimated standard errors and update weighting matrices."""

        # initialize values from the objective information
        self.problem = objective_info.problem
        self.WD = objective_info.WD
        self.WS = objective_info.WS
        self.theta = objective_info.theta
        self.delta = objective_info.delta
        self.tilde_costs = objective_info.tilde_costs
        self.xi_jacobian = objective_info.xi_jacobian
        self.omega_jacobian = objective_info.omega_jacobian
        self.beta = objective_info.beta
        self.gamma = objective_info.gamma
        self.xi = objective_info.xi
        self.omega = objective_info.omega
        self.objective = objective_info.objective
        self.gradient = objective_info.gradient
        self.gradient_norm = objective_info.gradient_norm

        # store parameter information
        self._nonlinear_parameters = objective_info.nonlinear_parameters
        self._linear_parameters = LinearParameters(self.problem, self.beta, self.gamma)

        # expand the nonlinear parameters and their gradient
        self.sigma, self.pi = self._nonlinear_parameters.expand(self.theta, fill_fixed=True)
        self.sigma_gradient, self.pi_gradient = self._nonlinear_parameters.expand(self.gradient)

        # stack the error terms, weighting matrices, instruments, and Jacobian of the error terms with respect to all
        #   parameters
        if self.problem.K3 == 0:
            u = self.xi
            W = self.WD
            Z = self.problem.products.ZD
            jacobian = np.c_[self.xi_jacobian, self.problem.products.X1]
        else:
            u = np.r_[self.xi, self.omega]
            W = scipy.linalg.block_diag(self.WD, self.WS)
            Z = scipy.linalg.block_diag(self.problem.products.ZD, self.problem.products.ZS)
            jacobian = np.c_[
                np.r_[self.xi_jacobian, self.omega_jacobian],
                scipy.linalg.block_diag(self.problem.products.X1, self.problem.products.X2)
            ]

        # compute standard errors
        se = self._compute_se(u, Z, W, jacobian, se_type)
        self.sigma_se, self.pi_se = self._nonlinear_parameters.expand(se[:self._nonlinear_parameters.P])
        self.beta_se = se[self._nonlinear_parameters.P:self._nonlinear_parameters.P + self.problem.K1]
        self.gamma_se = se[-self.problem.K3:] if self.problem.K3 > 0 else None

        # update weighting matrices
        self.updated_WD = self._update_W(self.xi, self.problem.products.ZD, center_moments, "demand")
        self.updated_WS = None
        if self.problem.K3 > 0:
            self.updated_WS = self._update_W(self.omega, self.problem.products.ZS, center_moments, "supply")

        # construct an array of unique and sorted market IDs
        self.unique_market_ids = np.unique(self.problem.products.market_ids).flatten()

        # initialize counts and times
        self.step = 1
        self.total_time = self.cumulative_total_time = time.time() - start_time
        self.optimization_time = self.cumulative_optimization_time = end_time - start_time
        self.optimization_iterations = self.cumulative_optimization_iterations = iterations
        self.objective_evaluations = self.cumulative_objective_evaluations = evaluations

        # convert contraction mappings to a matrices with rows ordered by market
        contraction_iteration_lists = [[m[t] for m in iteration_mappings] for t in self.unique_market_ids]
        contraction_evaluation_lists = [[m[t] for m in evaluation_mappings] for t in self.unique_market_ids]
        self.fp_iterations = self.cumulative_fp_iterations = np.array(contraction_iteration_lists)
        self.contraction_evaluations = self.cumulative_contraction_evaluations = np.array(contraction_evaluation_lists)

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

    def __str__(self):
        """Format full results as a string."""
        sections = []

        # construct section containing summary information
        header = [
            ("GMM", "Step"), ("Optimization", "Iterations"), ("Objective", "Evaluations"),
            ("Total Fixed Point", "Iterations"), ("Total Contraction", "Evaluations"), ("Objective", "Value"),
            ("Gradient", "Infinity Norm")
        ]
        widths = [max(len(k1), len(k2), options.digits + 6 if i > 4 else 0) for i, (k1, k2) in enumerate(header)]
        formatter = output.table_formatter(widths)
        sections.append([
            "Results Summary:",
            formatter.border(),
            formatter([k[0] for k in header]),
            formatter([k[1] for k in header]),
            formatter.lines(),
            formatter([
                self.step,
                self.optimization_iterations,
                self.objective_evaluations,
                self.fp_iterations.sum(),
                self.contraction_evaluations.sum(),
                output.format_number(self.objective),
                output.format_number(self.gradient_norm)
            ]),
            formatter.border()
        ])

        # construct a section containing linear estimates
        sections.append([
            "Linear Estimates (SEs in Parentheses):",
            self._linear_parameters.format(self.beta, self.gamma, self.beta_se, self.gamma_se)
        ])

        # construct a section containing nonlinear estimates
        sections.append([
            "Nonlinear Estimates (SEs in Parentheses):",
            self._nonlinear_parameters.format(self.sigma, self.pi, self.sigma_se, self.pi_se)
        ])

        # combine the sections into one string
        return "\n\n".join("\n".join(s) for s in sections)

    def __repr__(self):
        """Defer to the string representation."""
        return str(self)

    def _compute_se(self, u, Z, W, jacobian, se_type):
        """Use an error term, instruments, a weighting matrix, and the Jacobian of the error term with respect to
        parameters to estimate standard errors.
        """

        # compute the Jacobian of the sample moments with respect to all parameters
        G = Z.T @ jacobian

        # attempt to compute the unadjusted covariance matrix and output information about the method used to compute it
        attempt, covariances = self._invert(G.T @ W @ G)
        if attempt > 1:
            output("")
            output(f"The estimated covariance matrix of parameters is singular.")
            if attempt == 2:
                output(f"Used the Moore-Penrose pseudo inverse to estimate the covariance matrix of parameters.")
            else:
                output("Failed to compute the Moore-Penrose pseudo-inverse.")
                output(f"Inverted only the variance terms to estimate the covariance matrix of parameters.")
            output("")

        # compute the robust covariance matrix
        with np.errstate(invalid='ignore'):
            if se_type == 'robust':
                covariances = covariances @ G.T @ W @ Z.T @ (np.diagflat(u) ** 2) @ Z @ W @ G @ covariances
            se = np.sqrt(np.c_[covariances.diagonal()])

        # output any information about computation failure
        if np.isnan(se).any():
            output("")
            output(f"Failed to compute standard errors because of null values.")
            output("")
        return se

    def _update_W(self, u, Z, center_moments, side):
        """Use an error term and instruments to update a GMM weighting matrix."""

        # compute and center the sample moments
        g = u * Z
        if center_moments:
            g -= g.mean(axis=0)

        # attempt to compute the weighting matrix and output information about the method used to compute it
        attempt, W = self._invert(g.T @ g)
        if attempt > 1:
            output("")
            output(f"The estimated covariance matrix of {side}-side GMM moments is singular.")
            if attempt == 2:
                output(f"Used the Moore-Penrose pseudo inverse to update the {side}-side GMM weighting matrix.")
            else:
                output("Failed to compute the Moore-Penrose pseudo-inverse.")
                output(f"Inverted only the variance terms to update the {side}-side GMM weighting matrix.")
            output("")

        # output any information about computation failure
        if np.isnan(W).any():
            output("")
            output(f"Failed to compute the {side}-side GMM weighting matrix because of null values.")
            output("")
        return W

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

    def _validate_name(self, name):
        """Validate that a name corresponds to a variable in X1 or X2 (or both)."""
        names = {n for f in self.problem._X1_formulations + self.problem._X2_formulations for n in f.names}
        if name not in names:
            raise NameError(f"The name '{name}' is not one of the variables in X1 or X2, {list(sorted(names))}.")

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
            market_t = ResultsMarket(self.problem, t, self.delta, self.xi, self.beta, self.sigma, self.pi)
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
        combined = np.full((rows, columns), np.nan, options.dtype)
        market_ids = self.problem.products.market_ids.flat if rows == self.problem.N else self.unique_market_ids
        for t, matrix_t in matrix_mapping.items():
            combined[market_ids == t, :matrix_t.shape[1]] = matrix_t

        # output how long it took to compute results
        end_time = time.time()
        output(f"Finished after {output.format_seconds(end_time - start_time)}.")
        return combined

    def compute_aggregate_elasticities(self, factor=0.1, name='prices'):
        r"""Estimate aggregate elasticities of demand, :math:`E`, with respect to a variale, :math:`x`.

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
            :attr:`Results.unique_market_ids`.

        """
        self._validate_name(name)
        output(f"Computing aggregate elasticities with respect to {name} ...")
        return self._combine_results(ResultsMarket.compute_aggregate_elasticity, [factor, name])

    def compute_elasticities(self, name='prices'):
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
        self._validate_name(name)
        output(f"Computing elasticities with respect to {name} ...")
        return self._combine_results(ResultsMarket.compute_elasticities, [name])

    def compute_diversion_ratios(self, name='prices'):
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
        self._validate_name(name)
        output(f"Computing diversion ratios with respect to {name} ...")
        return self._combine_results(ResultsMarket.compute_diversion_ratios, [name])

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
            Stacked matrices, such as estimates of :math:`\varepsilon`, computed by
            :meth:`Results.compute_elasticities`; :math:`\mathscr{D}`, computed by
            :meth:`Results.compute_diversion_ratios`; or :math:`\bar{\mathscr{D}}`, computed by
            :meth:`Results.compute_long_run_diversion_ratios`.

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
            Stacked matrices, such as estimates of :math:`\varepsilon`, computed by
            :meth:`Results.compute_elasticities`; :math:`\mathscr{D}`, computed by
            :meth:`Results.compute_diversion_ratios`; or :math:`\bar{\mathscr{D}}`, computed by
            :meth:`Results.compute_long_run_diversion_ratios`.

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

    def solve_approximate_merger(self, firms_index=1, costs=None):
        r"""Estimate approximate post-merger prices, :math:`p^a`, under the assumption that shares and their price
        derivatives are unaffected by the merger.

        This approximation is discussed in, for example, :ref:`Nevo (1997) <n97>`. Prices in each market are computed
        according to the approximate post-merger version of the BLP-markup equation,

        .. math:: p^a = c + \eta^a,

        in which the approximate markup term is

        .. math:: \eta^a = -\left(O^* \circ \frac{\partial s}{\partial p}\right)^{-1}s

        where :math:`O^*` is the post-merger ownership matrix.

        Parameters
        ----------
        firms_index : `int, optional`
            Column index of the changed firm IDs in the `firm_ids` field of `product_data` in :class:`Problem`
            initialization. If an `ownership` field was specified, the corresponding stack of ownership matrices will be
            used. Ownership changes need not reflect an actual merger.
        costs : `array-like, optional`
            Marginal costs, :math:`c`, computed by :meth:`Results.compute_costs`. By default, marginal costs are
            computed.

        Returns
        -------
        `ndarray`
            Estimates of approximate post-merger prices, :math:`p^a`.

        """
        output("Solving for approximate post-merger prices ...")
        return self._combine_results(ResultsMarket.solve_approximate_merger, [firms_index], [costs])

    def solve_merger(self, iteration=None, firms_index=1, prices=None, costs=None, processes=1):
        r"""Estimate post-merger prices, :math:`p^*`.

        Prices are computed in each market by iterating over the post-merger version of the :math:`\zeta`-markup
        equation from :ref:`Morrow and Skerlos (2011) <ms11>`,

        .. math:: p^* \leftarrow c + \zeta^*(p^*),

        in which the markup term is

        .. math:: \zeta^*(p^*) = \Lambda^{-1}(p^*)[O^* \circ \Gamma(p^*)]'(p^* - c) - \Lambda^{-1}(p^*)

        where :math:`O^*` is the post-merger ownership matrix and the other terms are the same as in the pre-merger
        :math:`\zeta`-markup equation but are evaluated at post-merger prices and shares.

        Parameters
        ----------
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem in each market. By default,
            ``Iteration('simple', {'tol': 1e-12})`` is used.
        firms_index : `int, optional`
            Column index of the changed firm IDs in the `firm_ids` field of `product_data` in :class:`Problem`
            initialization. If an `ownership` field was specified, the corresponding stack of ownership matrices will be
            used. Ownership changes need not reflect an actual merger.
        prices : `array-like, optional`
            Prices at which the fixed point iteration routine will start. By default, pre-merger prices, :math:`p`, are
            used as starting values. Other reasonable starting prices include :math:`p^a`, computed by
            :meth:`Results.solve_approximate_merger`.
        costs : `array-like`
            Marginal costs, :math:`c`, computed by :meth:`Results.compute_costs`. By default, marginal costs are
            computed.
        processes : `int, optional`
            Number of Python processes that will be used during computation. By default, multiprocessing will not be
            used. For values greater than one, a pool of that many Python processes will be created. Market-by-market
            computation of post-merger prices will be distributed among these processes. Using multiprocessing will only
            improve computation speed if gains from parallelization outweigh overhead from creating the process pool.

        Returns
        -------
        `ndarray`
            Estimates of post-merger prices, :math:`p^*`.

        """
        if iteration is None:
            iteration = Iteration('simple', {'tol': 1e-12})
        elif not isinstance(iteration, Iteration):
            raise ValueError("iteration must an Iteration instance.")
        output("Solving for post-merger prices ...")
        return self._combine_results(ResultsMarket.solve_merger, [iteration, firms_index], [prices, costs], processes)

    def compute_shares(self, prices=None):
        r"""Estimate shares evaluated at specified prices.

        Parameters
        ----------
        prices : `array-like`
            Prices at which to evaluate shares, such as post-merger prices, :math:`p^*`, computed by
            :meth:`Results.solve_merger`, or approximate post-merger prices, :math:`p^a`, computed by
            :meth:`Results.solve_approximate_merger`. By default, unchanged prices are used.

        Returns
        -------
        `ndarray`
            Estimates of shares evaluated at the specified prices.

        """
        output("Computing shares ...")
        return self._combine_results(ResultsMarket.compute_shares, market_args=[prices])

    def compute_hhi(self, firms_index=0, shares=None):
        r"""Estimate Herfindahl-Hirschman Indices, :math:`\text{HHI}`.

        The index in market :math:`t` is

        .. math:: \text{HHI} = 10,000 \times \sum_{f=1}^{F_t} \left(\sum_{j \in \mathscr{J}_{ft}} s_j\right)^2,

        in which :math:`\mathscr{J}_{ft}` is the set of products produced by firm :math:`f` in market :math:`t`.

        Parameters
        ----------
        firms_index : `int, optional`
            Column index of the firm IDs in the `firm_ids` field of `product_data` in :class:`Problem` initialization.
            By default, unchanged firm IDs are used.
        shares : `array-like, optional`
            Shares, :math:`s`, such as those computed by :meth:`Results.compute_shares`. By default, unchanged shares
            are used.

        Returns
        -------
        `ndarray`
            Estimated Herfindahl-Hirschman Indices, :math:`\text{HHI}`, for all markets. Rows are in the same order as
            :attr:`Results.unique_market_ids`.

        """
        output("Computing HHI ...")
        return self._combine_results(ResultsMarket.compute_hhi, [firms_index], [shares])

    def compute_markups(self, prices=None, costs=None):
        r"""Estimate markups, :math:`\mathscr{M}`.

        The markup of product :math:`j` in market :math:`t` is

        .. math:: \mathscr{M}_{jt} = \frac{p_{jt} - c_{jt}}{p_{jt}}.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices, :math:`p`, such as post-merger prices, :math:`p^*`, computed by :meth:`Results.solve_merger`, or
            approximate post-merger prices, :math:`p^a`, computed by :meth:`Results.solve_approximate_merger`. By
            default, unchanged prices are used.
        costs : `array-like`
            Marginal costs, :math:`c`, computed by :meth:`Results.compute_costs`. By default, marginal costs are
            computed.

        Returns
        -------
        `ndarray`
            Estimated markups, :math:`\mathscr{M}`.

        """
        output("Computing markups ...")
        return self._combine_results(ResultsMarket.compute_markups, market_args=[prices, costs])

    def compute_profits(self, prices=None, shares=None, costs=None):
        r"""Estimate population-normalized gross expected profits, :math:`\pi`.

        The profit of product :math:`j` in market :math:`t` is

        .. math:: \pi_{jt} = p_{jt} - c_{jt}s_{jt}.

        Parameters
        ----------
        prices : `array-like, optional`
            Prices, :math:`p`, such as post-merger prices, :math:`p^*`, computed by :meth:`Results.solve_merger`, or
            approximate post-merger prices, :math:`p^a`, computed by :meth:`Results.solve_approximate_merger`. By
            default, unchanged prices are used.
        shares : `array-like, optional`
            Shares, :math:`s`, such as those computed by :meth:`Results.compute_shares`. By default, unchanged shares
            are used.
        costs : `array-like`
            Marginal costs, :math:`c`, computed by :meth:`Results.compute_costs`. By default, marginal costs are
            computed.

        Returns
        -------
        `ndarray`
            Estimated population-normalized gross expected profits, :math:`\pi`.

        """
        output("Computing profits ...")
        return self._combine_results(ResultsMarket.compute_profits, market_args=[prices, shares, costs])

    def compute_consumer_surpluses(self, prices=None):
        r"""Estimate population-normalized consumer surpluses, :math:`\text{CS}`.

        Assuming away nonlinear income effects, the surplus in market :math:`t` is

        .. math:: \text{CS} = \sum_{i=1}^{I_t} w_i\text{CS}_i,

        in which the consumer surplus for individual :math:`i` is

        .. math:: \text{CS}_i = \frac{\log(1 + \sum_{j=1}^{J_t} \exp u_{jti})}{\alpha + \alpha_i}.

        .. warning::

           The consumer surpluses computed by this method are only correct when there are not nonlinear income effects.
           For example, computed consumer surpluses will be incorrect if a formulation contains ``log(prices)``.

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
        output("Computing consumer surpluses with the equation that assumes away nonlinear income effects ...")
        return self._combine_results(ResultsMarket.compute_consumer_surplus, market_args=[prices])


class ResultsMarket(Market):
    """Results for a single market of a solved BLP problem, which can be used to compute post-estimation outputs. Each
    method returns a matrix and a set of any errors that were encountered.
    """

    def compute_aggregate_elasticity(self, factor, name):
        """Market-specific computation for Results.compute_aggregate_elasticities."""
        scaled_variable = (1 + factor) * self.products[name]
        delta = self.update_delta_with_variable(name, scaled_variable)
        mu = self.update_mu_with_variable(name, scaled_variable)
        shares = self.compute_probabilities(delta, mu) @ self.agents.weights
        aggregate_elasticities = (shares - self.products.shares).sum() / factor
        return aggregate_elasticities, set()

    def compute_elasticities(self, name):
        """Market-specific computation for Results.compute_elasticities."""
        derivatives = self.compute_utility_by_variable_derivatives(name)
        jacobian = self.compute_shares_by_variable_jacobian(derivatives)
        elasticities = jacobian * self.products[name].T / self.products.shares
        return elasticities, set()

    def compute_diversion_ratios(self, name):
        """Market-specific computation for Results.compute_diversion_ratios."""
        derivatives = self.compute_utility_by_variable_derivatives(name)
        jacobian = self.compute_shares_by_variable_jacobian(derivatives)

        # replace the diagonal with derivatives with respect to the outside option
        jacobian_diagonal = np.c_[jacobian.diagonal()]
        jacobian[np.diag_indices_from(jacobian)] = -jacobian.sum(axis=1)

        # compute the ratios
        ratios = -jacobian / np.tile(jacobian_diagonal, self.J)
        return ratios, set()

    def compute_long_run_diversion_ratios(self):
        """Market-specific computation for Results.compute_long_run_diversion_ratios."""

        # compute share differences when products are excluded and store outside share differences on the diagonal
        changes = np.zeros((self.J, self.J), options.dtype)
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
            costs = self.products.prices - self.compute_eta()
        except scipy.linalg.LinAlgError:
            errors.add(exceptions.CostsSingularityError)
            costs = np.full((self.J, 1), np.nan, options.dtype)
        return costs, errors

    def solve_approximate_merger(self, firms_index, costs):
        """Market-specific computation for Results.solve_approximate_merger."""
        errors = set()
        if costs is None:
            costs, errors = self.compute_costs()
        ownership_matrix = self.get_ownership_matrix(firms_index)
        try:
            prices = costs + self.compute_eta(ownership_matrix)
        except scipy.linalg.LinAlgError:
            errors.add(exceptions.CostsSingularityError)
            prices = np.full((self.J, 1), np.nan, options.dtype)
        return prices, errors

    def solve_merger(self, iteration, firms_index, prices, costs):
        """Market-specific computation for Results.solve_merger."""
        errors = set()

        # configure NumPy to identify floating point errors
        with np.errstate(divide='call', over='call', under='ignore', invalid='call'):
            np.seterrcall(lambda *_: errors.add(exceptions.ChangedPricesFloatingPointError))
            prices, converged = self.compute_bertrand_nash_prices(iteration, firms_index, prices, costs)[:2]

        # determine whether the fixed point converged
        if not converged:
            errors.add(exceptions.ChangedPricesConvergenceError)
        return prices, errors

    def compute_shares(self, prices):
        """Market-specific computation for Results.compute_shares."""
        if prices is None:
            prices = self.products.prices
        delta = self.update_delta_with_variable('prices', prices)
        mu = self.update_mu_with_variable('prices', prices)
        shares = self.compute_probabilities(delta, mu) @ self.agents.weights
        return shares, set()

    def compute_hhi(self, firms_index, shares):
        """Market-specific computation for Results.compute_hhi."""
        if shares is None:
            shares = self.products.shares
        firm_ids = self.products.firm_ids[:, [firms_index]]
        hhi = 1e4 * sum((shares[firm_ids == f].sum() / shares.sum()) ** 2 for f in np.unique(firm_ids))
        return hhi, set()

    def compute_markups(self, prices, costs):
        """Market-specific computation for Results.compute_markups."""
        errors = set()
        if prices is None:
            prices = self.products.prices
        if costs is None:
            costs, errors = self.compute_costs()
        markups = (prices - costs) / prices
        return markups, errors

    def compute_profits(self, prices, shares, costs):
        """Market-specific computation for Results.compute_profits."""
        errors = set()
        if prices is None:
            prices = self.products.prices
        if shares is None:
            shares = self.products.shares
        if costs is None:
            costs, errors = self.compute_costs()
        profits = (prices - costs) * shares
        return profits, errors

    def compute_consumer_surplus(self, prices):
        """Market-specific computation for Results.compute_consumer_surpluses."""
        if prices is None:
            delta = self.delta
            mu = self.mu
        else:
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
        alpha = -self.compute_utility_by_variable_derivatives('prices')[0]
        consumer_surplus = (np.log1p(np.exp(delta + mu).sum(axis=0)) / alpha) @ self.agents.weights
        return consumer_surplus, set()
