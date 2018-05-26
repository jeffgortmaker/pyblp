"""Configuration of the BLP problem and routines used to solve it."""

import time
import functools

import numpy as np
import scipy.linalg

from . import options, exceptions
from .primitives import Products, Agents, Market
from .utilities import output, ParallelItems, Iteration, Optimization


class Problem(object):
    r"""A BLP problem

    This class is initialized with relevant data and solved with :meth:`Problem.solve`.

    In both `product_data` and `agent_data`, fields with multiple columns can be either matrices or can be broken up
    into multiple one-dimensional fields with column index suffixes that start at zero. For example, if there are two
    columns of non-price linear product characteristics, the `linear_characteristics` field in `product_data`, which in
    this case should be a matrix with two columns, can be replaced with two one-dimensional fields:
    `linear_characteristics0` and `linear_characteristics1`.

    Parameters
    ----------
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. Fields:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object, optional`) - IDs that associate products with firms. This field is required if
              `cost_characteristics` and `supply_instruments` are specified, since they will be used to estimate the
              supply side of the problem. These are also needed to compute some post-estimation outputs. Any columns
              after the first can be used to compute post-estimation outputs for changes, such as mergers.

            - **ownership** : (`numeric, optional') - Custom stacked :math:`J_t \times J_t` ownership matrices,
              :math:`O`, for each market :math:`t`, which can be built with :func:`build_ownership`. Each stack is
              associated with a `firm_ids` column and must have as many columns as there are products in the market with
              the most products. If a market has fewer products than others, extra columns will be ignored and may be
              filled with any value, such as ``numpy.nan``. If an ownership matrix stack is unspecified, its
              corresponding column in `firm_ids` is used by :func:`build_ownership` to build a stack of standard
              ownership matrices.

            - **shares** : (`numeric`) - Shares, :math:`s`.

            - **prices** : (`numeric`) - Prices, :math:`p`, which will be included as the first column in :math:`X_1`
              and :math:`X_2` if `linear_prices` and `nonlinear_prices` are ``True``, respectively.

            - **linear_characteristics** : (`numeric, optional`) - Non-price product characteristics that constitute the
              remaining columns in :math:`X_1` if `linear_prices` is ``True``, and the entirety of the matrix if prices
              are not included in :math:`X_1`.

            - **nonlinear_characteristics** : (`numeric, optional`) - Non-price product characteristics that constitute
              the remaining columns in :math:`X_2` if `nonlinear_prices` is ``True``, and the entirety of the matrix if
              prices are not included in :math:`X_2`.

            - **cost_characteristics** : (`numeric, optional`) - Cost product characteristics, :math:`X_3`. This field
              is required if `supply_instruments` is specified, since they will be used to estimate the supply side of
              the problem. If unspecified, only the demand side will be estimated.

            - **demand_instruments** : (`numeric`) - Demand-side instruments, :math:`Z_D`, which should contain the sets
              of columns in `linear_characteristics` and `nonlinear_characteristics`.

            - **supply_instruments** : (`numeric, optional`) - Supply-side instruments, :math:`Z_S`, which should
              contain `cost_characteristics`. This field is required if `cost_characteristics` is specified, since they
              will be used to estimate the supply side of the problem. If unspecified, only the demand side will be
              estimated.

    agent_data : `structured array-like, optional`
        Required if `integration` is unspecified. Each row corresponds to an agent. Markets can have differing numbers
        of agents. Fields:

            - **market_ids** : (`object`) - IDs that associate agents with markets. The set of distinct IDs should be
              the same as the set of distinct IDs in the `market_ids` field of `product_data`. If `integration` is
              specified, there must be at least as many rows in each market as the number of nodes and weights that are
              built for each market.

            - **weights** : (`numeric, optional`) - Integration weights, :math:`w`. This field is required if
              `integration` is unspecified.

            - **nodes** : (`numeric, optional`) - Unobserved agent characteristics called integration nodes,
              :math:`\nu`. This field is required if `integration` is unspecified. If there are more than :math:`K_2`
              columns, only the first :math:`K_2` will be used.

            - **demographics** : (`numeric, optional`) - Observed agent characteristics called demographics, :math:`d`.
              If `integration` is specified and there are more rows of demographics in a market :math:`t` than
              :math:`I_t`, the number node and weight rows built for that market, only the first :math:`I_t` rows of
              demographics will be used.

    integration : `Integration, optional`
        :class:`Integration` configuration for how to build nodes and weights for integration over agent utilities,
        which is required if the `nodes` and `weights` fields of `agent_data` are unspecified. If they are specified,
        the nodes and weights will be build according to this configuration for each market, and they will replace those
        in `agent_data`.
    linear_prices : `bool, optional`
        Whether prices will be included in :math:`X_1` as the first column. By default, prices are included in
        :math:`X_1`.
    nonlinear_prices : `bool, optional`
        Whether prices will be included in :math:`X_2` as the first column. By default, prices are included in
        :math:`X_2`.

    Attributes
    ----------
    products : `Products`
        Restructured `product_data` from :class:`Problem` initialization, which is an instance of
        :class:`primitives.Products`.
    agents : `Agents`
        Restructured `agent_data` from :class:`Problem` initialization with nodes and weights built according to
        `integration` if it is specified, which is an instance of :class:`primitives.Agents`.
    N : `int`
        Number of products across all markets, :math:`N`.
    T : `int`
        Number of markets, :math:`T`.
    K1 : `int`
        Number of linear product characteristics, :math:`K_1`.
    K2 : `int`
        Number of nonlinear product characteristics, :math:`K_2`.
    K3 : `int`
        Number of cost product characteristics, :math:`K_3`.
    D : `int`
        Number of demographic variables, :math:`D`.
    MD : `int`
        Number of demand-side instruments, :math:`M_D`.
    MS : `int`
        Number of supply-side instruments, :math:`M_S`.
    linear_prices : `bool`
        Whether prices are included in :math:`X_1` as the first column.
    nonlinear_prices : `bool`
        Whether prices are included in :math:`X_2` as the first column.

    Example
    -------
    The following code initializes a very simple problem with some of the automobile product data from
    :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`:

    .. ipython:: python

       data = np.recfromcsv(pyblp.data.BLP_PRODUCTS_LOCATION)
       linear = np.c_[np.ones(data.size), data['hpwt'], data['air'], data['mpg'], data['space']]
       characteristic_data = {
           'market_ids': data['market_ids'],
           'firm_ids': data['firm_ids'],
           'characteristics': linear
       }
       instruments = np.c_[linear, pyblp.build_blp_instruments(characteristic_data)]
       product_data = {
           'market_ids': data['market_ids'],
           'shares': data['shares'],
           'prices': data['prices'],
           'linear_characteristics': linear,
           'nonlinear_characteristics': linear[:, :-1],
           'demand_instruments': instruments
       }
       problem = pyblp.Problem(product_data, integration=pyblp.Integration('monte_carlo', 50, seed=0))

    After choosing to optimize over the diagonal variance elements in :math:`\Sigma` and choosing starting values, the
    initialized problem can be solved. For simplicity, the following code halts estimation after one GMM step:

    .. ipython:: python

       initial_sigma = np.diag(np.ones(5))
       results = problem.solve(initial_sigma, steps=1)
       results

    """

    def __init__(self, product_data, agent_data=None, integration=None, linear_prices=True, nonlinear_prices=True):
        """Structure and validate data before computing matrix dimensions."""
        output("Structuring product data ...")
        self.products = Products(product_data, linear_prices, nonlinear_prices)

        output("Structuring agent data ...")
        self.agents = Agents(self.products, agent_data, integration)

        # store problem configuration information
        self.linear_prices = linear_prices
        self.nonlinear_prices = nonlinear_prices
        self.N = self.products.shape[0]
        self.T = np.unique(self.products.market_ids).size
        self.K1 = self.products.X1.shape[1]
        self.K2 = self.products.X2.shape[1]
        try:
            self.K3 = self.products.X3.shape[1]
        except AttributeError:
            self.K3 = 0
        try:
            self.D = self.agents.demographics.shape[1]
        except AttributeError:
            self.D = 0
        self.MD = self.products.ZD.shape[1]
        try:
            self.MS = self.products.ZS.shape[1]
        except AttributeError:
            self.MS = 0

        # output configuration information
        output("")
        output(self)

    def __str__(self):
        """Format problem information as a string."""
        header = ["N", "T", "K1", "K2", "K3", "D", "MD", "MS", "Linear Prices", "Nonlinear Prices"]
        widths = [max(len(k), 10) for k in header]
        formatter = output.table_formatter(widths)
        return "\n".join([
            "Problem Configuration:",
            formatter.border(),
            formatter(header),
            formatter.lines(),
            formatter([
                self.N, self.T, self.K1, self.K2, self.K3, self.D, self.MD, self.MS, self.linear_prices,
                self.nonlinear_prices
            ]),
            formatter.border()
        ])

    def __repr__(self):
        """Defer to the string representation."""
        return str(self)

    def solve(self, sigma, pi=None, sigma_bounds=None, pi_bounds=None, delta=None, WD=None, WS=None, steps=2,
              optimization=None, error_behavior='revert', error_punishment=1, iteration=None, linear_fp=True,
              linear_costs=True, center_moments=True, se_type='robust', processes=1):
        r"""Solve the problem.

        Parameters
        ----------
        sigma : `array-like`
            Configuration for which elements in the Cholesky decomposition of the covariance matrix that measures
            agents' random taste distribution, :math:`\Sigma`, are fixed at zero and starting values for the other
            elements, which, if not fixed by `sigma_bounds`, are in the vector of unknown elements, :math:`\theta`. If
            `nonlinear_prices` in :class:`Problem` initialization was `True`, the first row and column correspond to
            prices, and if `product_data` contained a `nonlinear_characteristics` field, all other rows and columns
            correspond to its columns.

            Values below the diagonal are ignored. Zeros are assumed to be zero throughout estimation and nonzeros are,
            if not fixed by `sigma_bounds`, starting values for unknown elements in :math:`\theta`.

        pi : `array-like, optional`
            Configuration for which elements in the matrix of parameters that measures how agent tastes vary with
            demographics, :math:`\Pi`, are fixed at zero and starting values for the other elements, which, if not fixed
            by `pi_bounds`, are in the vector of unknown elements, :math:`\theta`. Rows correspond to the same product
            characteristics as in `sigma`. Columns correspond to the columns of the `demographics` field of `agent_data`
            in :class:`Problem` initialization.

            Zeros are assumed to be zero throughout estimation and nonzeros are, if not fixed by `pi_bounds`, starting
            values for unknown elements in :math:`\theta`.

        sigma_bounds : `tuple, optional`
            Configuration for :math:`\Sigma` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as `sigma`. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in `sigma`. If `optimization` does not support bounds, these will be ignored. By default, if
            bounds are supported, the only unfixed elements that are bounded are those on the diagonal of `sigma`, which
            are bounded below by zero. That is, the diagonal of ``lb`` is all zeros by default.

            Values below the diagonal are ignored. Lower and upper bounds corresponding to zeros in `sigma` are set to
            zero. Setting a lower bound equal to an upper bound fixes the corresponding element. Both ``None`` and
            ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        pi_bounds : `tuple, optional`
            Configuration for :math:`\Pi` bounds of the form ``(lb, ub)``, in which both ``lb`` are ``ub`` are of the
            same size as `pi`. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in `pi`. If `optimization` does not support bounds, these will be ignored. By default, if bounds
            are supported, all unfixed elements are unbounded.

            Lower and upper bounds corresponding to zeros in `pi` are set to zero. Setting a lower bound equal to an
            upper bound fixes the corresponding element. Both ``None`` and ``numpy.nan`` are converted to ``-numpy.inf``
            in ``lb`` and to ``numpy.inf`` in ``ub``.

        delta : `array-like, optional`
            Starting values for the mean utility, :math:`\delta`. By default,
            :math:`\delta_{jt} = \log s_{jt} - \log s_{0t}` is used.
        WD : `array-like, optional`
            Starting values for the demand-side weighting matrix, :math:`W_D`. By default, the 2SLS weighting matrix,
            :math:`W_D = (Z_D'Z_D)^{-1}`, is used.
        WS : `array-like, optional`
            Starting values for the supply-side weighting matrix, :math:`W_S`, which is only used if the problem was
            initialized with supply-side data. By default, the 2SLS weighting matrix, :math:`W_S = (Z_S'Z_S)^{-1}`, is
            used.
        steps : `int, optional`
            Number of GMM steps. By default, two-step GMM is used.
        optimization : `Optimization, optional`
            :class:`Optimization` configuration for how to solve the optimization problem in each GMM step. By default,
            ``Optimization('l-bfgs-b')`` is used. Routines that do not support bounds will ignore `sigma_bounds` and
            `pi_bounds`. Choosing a routine that does not use analytic gradients will slow down estimation.
        error_behavior : `str, optional`
            How to handle errors when computing the objective. For example, it is common to encounter overflow when
            computing :math:`\delta(\hat{\theta})` at a large :math:`\hat{\theta}`. The following behaviors are
            supported:

                - ``'raise'`` - Raises an exception.

                - ``'revert'`` (default) - Reverts problematic :math:`\delta(\hat{\theta})` elements to their last
                  computed values and uses reverted values to compute :math:`\partial\delta / \partial\theta` and, if
                  the problem was configured with supply-side data, to compute :math:`\tilde{c}(\hat{\theta})` as well.
                  If there are problematic elements in :math:`\partial\delta / \partial\theta` or
                  :math:`\tilde{c}(\hat{\theta})`, these are also reverted to their last computed values. If there are
                  problematic elements in the first iteration, values in :math:`\delta(\hat{\theta})` are reverted to
                  their starting values; in :math:`\partial\delta / \partial\theta`, to zeros; and in
                  :math:`\tilde{c}(\hat{\theta})`, to prices.

                - ``'punish'`` - Sets the objective to ``1`` and its gradient to all zeros. This option along with a
                  large `error_punishment` can be helpful for routines that do not use analytic gradients.

        error_punishment : `float, optional`
            How much to scale the GMM objective value when computing it gives rise to an error. By default, the value
            is not scaled.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem used to compute
            :math:`\delta(\hat{\theta})` in each market. By default, ``Iteration('squarem', {'tol': 1e-14})`` is used.
        linear_fp : `bool, optional`
            Whether to compute :math:`\delta(\hat{\theta})` with the standard linear contraction mapping,

            .. math:: \delta \leftarrow \delta + \log s - \log s(\delta, \hat{\theta}),

            or with its nonlinear formulation,

            .. math:: \exp(\delta) \leftarrow \exp(\delta)s / s(\delta, \hat{\theta}).

            The default linear contraction is more popular; however, its nonlinear formulation can be faster because
            fewer logarithms need to be calculated, which can also help mitigate problems stemming from negative
            integration weights.

        linear_costs : `bool, optional`
            Whether to compute :math:`\tilde{c}(\hat{\theta})` according to a linear or a log-linear marginal cost
            specification. This is only relevant if the problem was initialized with supply-side data. By default, a
            linear specification is assumed. That is, :math:`\tilde{c} = c` instead of :math:`\tilde{c} = \log c`.
        center_moments : `bool, optional`
            Whether to center the sample moments before using them to update weighting matrices. By default, sample
            moments are centered.
        se_type : `str, optional`
            How to compute standard errors. The following types are supported:

                - ``'robust'`` (default) - Robust standard errors.

                - ``'unadjusted'`` - Unadjusted standard errors computed under the assumption that weighting matrices
                  are optimal.

        processes : `int, optional`
            Number of Python processes that will be used during estimation. By default, multiprocessing will not be
            used. For values greater than one, a pool of that many Python processes will be created during each
            iteration of the optimization routine. Market-by-market computation of :math:`\delta(\hat{\theta})` and
            its Jacobian will be distributed among these processes. Using multiprocessing will only improve estimation
            speed if gains from parallelization outweigh overhead from creating the process pool.

        Returns
        -------
        `Results`
            :class:`Results` of the solved problem.

        """
        output("")

        # configure or validate optimization and integration
        if optimization is None:
            optimization = Optimization('l-bfgs-b')
        if iteration is None:
            iteration = Iteration('squarem', {'tol': 1e-14})
        if not isinstance(optimization, Optimization):
            raise ValueError("optimization must be an Optimization instance.")
        if not isinstance(iteration, Iteration):
            raise ValueError("iteration must be an Iteration instance.")

        # validate error behavior and the standard error type
        if error_behavior not in {'raise', 'revert', 'punish'}:
            raise ValueError("error_behavior must be 'raise', 'revert', or 'punish'.")
        if se_type not in {'robust', 'unadjusted'}:
            raise ValueError("se_type must be 'robust' or 'unadjusted'.")

        # output configuration information
        output(f"GMM steps: {steps}.")
        output(optimization)
        output(f"Error behavior: {error_behavior}.")
        output(f"Error punishment: {output.format_number(error_punishment)}.")
        output(iteration)
        output(f"Linear fixed point formulation: {linear_fp}.")
        if self.K3 > 0:
            output(f"Linear marginal cost specification: {linear_costs}.")
        output(f"Centering sample moments before updating weighting matrices: {center_moments}.")
        output(f"Standard error type: {se_type}.")
        output(f"Processes: {processes}.")

        # compress sigma and pi into theta but retain information about the original matrices
        theta_info = ThetaInfo(self, sigma, pi, sigma_bounds, pi_bounds, optimization._supports_bounds)
        output(f"Number of unfixed nonlinear parameters in theta: {theta_info.P}.")
        output("")
        output(theta_info)
        output("")
        theta = theta_info.compress(theta_info.sigma, theta_info.pi)

        # construct or validate the demand-side weighting matrix
        if WD is None:
            output("Starting with the 2SLS demand-side weighting matrix.")
            WD = scipy.linalg.inv(self.products.ZD.T @ self.products.ZD)
        else:
            output("Starting with the specified demand-side weighting matrix.")
            WD = np.asarray(WD, options.dtype)
            if WD.shape != (self.MD, self.MD):
                raise ValueError(f"WD must have {self.MD} rows and columns.")

        # construct or validate the supply-side weighting matrix
        if self.MS == 0:
            WS = None
        elif WS is None:
            output("Starting with the 2SLS supply-side weighting matrix.")
            WS = scipy.linalg.inv(self.products.ZS.T @ self.products.ZS)
        else:
            output("Starting with the specified supply-side weighting matrix.")
            WS = np.asarray(WS, options.dtype)
            if WS.shape != (self.MS, self.MS):
                raise ValueError(f"WS must have {self.MS} rows and columns.")

        # construct or validate delta
        if delta is None:
            output("Starting with the delta that solves the logit model.")
            delta = np.log(self.products.shares)
            for t in np.unique(self.products.market_ids):
                shares_t = self.products.shares[self.products.market_ids.flat == t]
                delta[self.products.market_ids.flat == t] -= np.log(shares_t.sum())
        else:
            output("Starting with the specified delta.")
            delta = np.c_[np.asarray(delta, options.dtype)]
            if delta.shape != (self.N, 1):
                raise ValueError(f"delta must be a vector with {self.N} elements.")

        # initialize the Jacobian of xi with respect to theta as all zeros
        xi_jacobian = np.zeros((self.N, theta.size), options.dtype)

        # initialize marginal costs as prices and the Jacobian of omega with respect to theta as all zeros
        tilde_costs = omega_jacobian = None
        if self.K3 > 0:
            tilde_costs = self.products.prices if linear_costs else np.log(self.products.prices)
            omega_jacobian = xi_jacobian.copy()

        # iterate through each step
        last_results = None
        for step in range(1, steps + 1):
            # wrap computation of objective information with step-specific information
            compute_step_info = functools.partial(
                self._compute_objective_info, theta_info, WD, WS, error_behavior, error_punishment, iteration,
                linear_costs, linear_fp, processes
            )

            # define the objective function for the optimization routine, which also outputs progress updates
            def wrapper(new_theta, current_iterations, current_evaluations):
                info = wrapper.cache = compute_step_info(new_theta, wrapper.cache, optimization._compute_gradient)
                wrapper.iteration_mappings.append(info.iteration_mapping)
                wrapper.evaluation_mappings.append(info.evaluation_mapping)
                if optimization._universal_display:
                    output(info.format_progress(
                        step, current_iterations, current_evaluations, wrapper.smallest_objective,
                        wrapper.smallest_gradient_norm
                    ))
                wrapper.smallest_objective = min(wrapper.smallest_objective, info.objective)
                wrapper.smallest_gradient_norm = min(wrapper.smallest_gradient_norm, info.gradient_norm)
                return (info.objective, info.gradient) if optimization._compute_gradient else info.objective

            # initialize optimization progress
            wrapper.iteration_mappings = []
            wrapper.evaluation_mappings = []
            wrapper.smallest_objective = wrapper.smallest_gradient_norm = np.inf
            wrapper.cache = ObjectiveInfo(
                self, theta_info, WD, WS, theta, delta, tilde_costs, xi_jacobian, omega_jacobian
            )

            # optimize theta
            output(f"Starting optimization for step {step} out of {steps} ...")
            output("")
            start_time = time.time()
            bounds = [(p.lb, p.ub) for p in theta_info.unfixed]
            theta, converged, iterations, evaluations = optimization._optimize(theta, bounds, wrapper)
            status = "completed" if converged else "failed"
            end_time = time.time()
            output("")
            output(f"Optimization for step {step} {status} after {output.format_seconds(end_time - start_time)}.")

            # handle convergence problems
            if not converged:
                if error_behavior == 'raise':
                    raise exceptions.ThetaConvergenceError
                output("")
                output(exceptions.ThetaConvergenceError())
                output("")

            # use objective information computed at the optimal theta to compute results for the step
            output(f"Computing results for step {step} ...")
            step_info = compute_step_info(theta, wrapper.cache, compute_gradient=True)
            results = step_info.to_results(
                last_results, start_time, end_time, iterations, evaluations, wrapper.iteration_mappings,
                wrapper.evaluation_mappings, center_moments, se_type
            )
            delta = step_info.delta
            tilde_costs = step_info.tilde_costs
            xi_jacobian = step_info.xi_jacobian
            omega_jacobian = step_info.omega_jacobian
            WD = results.updated_WD
            WS = results.updated_WS
            output("")
            output(results)

            # store the last results and return results from the last step
            last_results = results
            if step == steps:
                return results

    def _compute_objective_info(self, theta_info, WD, WS, error_behavior, error_punishment, iteration, linear_costs,
                                linear_fp, processes, theta, last_objective_info, compute_gradient):
        """Compute demand- and supply-side contributions. Then, form the GMM objective value and its gradient. Finally,
        handle any errors that were encountered before structuring relevant objective information.
        """
        sigma, pi = theta_info.expand(theta, fill_fixed=True)

        # compute demand-side contributions
        demand_contributions = self._compute_demand_contributions(
            theta_info, WD, iteration, linear_fp, processes, sigma, pi, last_objective_info, compute_gradient
        )
        delta, xi_jacobian, PD, beta, xi, demand_errors, iteration_mapping, evaluation_mapping = demand_contributions

        # compute supply-side contributions
        supply_errors = set()
        tilde_costs = omega_jacobian = PS = gamma = omega = None
        if self.K3 > 0:
            supply_contributions = self._compute_supply_contributions(
                theta_info, WD, WS, linear_costs, processes, delta, xi_jacobian, beta, sigma, pi, last_objective_info,
                compute_gradient
            )
            tilde_costs, omega_jacobian, PS, gamma, omega, supply_errors = supply_contributions

        # stack the error terms, projection matrices, and Jacobian of the error terms with respect to theta
        if self.K3 == 0:
            u = xi
            P = PD
            jacobian = xi_jacobian
        else:
            u = np.r_[xi, omega]
            P = scipy.linalg.block_diag(PD, PS)
            jacobian = np.r_[xi_jacobian, omega_jacobian]

        # compute the objective value and its gradient
        objective = u.T @ P @ u
        gradient = 2 * (jacobian.T @ P @ u) if compute_gradient else None

        # handle any errors
        errors = demand_errors | supply_errors
        if errors:
            exception = exceptions.MultipleErrors(errors)
            if error_behavior == 'raise':
                raise exception
            output(exception)
            if error_behavior == 'revert':
                if error_punishment != 1:
                    objective *= error_punishment
                    output(f"Multiplied the objective by {output.format_number(error_punishment)}.")
            elif error_behavior == 'punish':
                objective = np.array(error_punishment)
                gradient = np.zeros_like(theta)
                output(f"Set the objective to {output.format_number(error_punishment)} and its gradient to all zeros.")

        # structure objective information
        return ObjectiveInfo(
            self, theta_info, WD, WS, theta, delta, tilde_costs, xi_jacobian, omega_jacobian, beta, gamma, xi, omega,
            objective, gradient, iteration_mapping, evaluation_mapping
        )

    def _compute_demand_contributions(self, theta_info, WD, iteration, linear_fp, processes, sigma, pi,
                                      last_objective_info,  compute_gradient):
        """Compute delta and the Jacobian of xi (equivalently, of delta) with respect to theta market-by-market. If
        necessary, revert problematic elements to their last values. Lastly, recover beta and compute xi.
        """
        errors = set()

        # construct a mapping from market IDs to market-specific arguments used to compute delta and its Jacobian
        mapping = {}
        for t in np.unique(self.products.market_ids):
            market_t = DemandProblemMarket(
                t, self.linear_prices, self.nonlinear_prices, self.products, self.agents, sigma=sigma, pi=pi
            )
            last_delta_t = last_objective_info.delta[self.products.market_ids.flat == t]
            mapping[t] = [market_t, last_delta_t, theta_info, iteration, linear_fp, compute_gradient]

        # fill delta and its Jacobian market-by-market (the Jacobian will be null if the gradient isn't being computed)
        iteration_mapping = {}
        evaluation_mapping = {}
        delta = np.zeros((self.N, 1), options.dtype)
        xi_jacobian = np.zeros((self.N, theta_info.P), options.dtype)
        with ParallelItems(DemandProblemMarket.solve, mapping, processes) as items:
            for t, (delta_t, xi_jacobian_t, errors_t, iteration_mapping[t], evaluation_mapping[t]) in items:
                delta[self.products.market_ids.flat == t] = delta_t
                xi_jacobian[self.products.market_ids.flat == t] = xi_jacobian_t
                errors |= errors_t

        # replace invalid elements in delta with their last values
        bad_delta_indices = ~np.isfinite(delta)
        if np.any(bad_delta_indices):
            delta[bad_delta_indices] = last_objective_info.delta[bad_delta_indices]
            output(f"Number of problematic elements in delta that were reverted: {bad_delta_indices.sum()}.")

        # replace invalid elements in its Jacobian with their last values
        if compute_gradient:
            bad_jacobian_indices = ~np.isfinite(xi_jacobian)
            if np.any(bad_jacobian_indices):
                xi_jacobian[bad_jacobian_indices] = last_objective_info.xi_jacobian[bad_jacobian_indices]
                output(
                    f"Number of problematic elements in the Jacobian of xi (equivalently, of delta) with respect to "
                    f"theta that were reverted: {bad_jacobian_indices.sum()}."
                )

        # recover beta and compute xi
        PD = self.products.ZD @ WD @ self.products.ZD.T
        X1PD = self.products.X1.T @ PD
        beta = scipy.linalg.solve(X1PD @ self.products.X1, X1PD @ delta)
        xi = delta - self.products.X1 @ beta
        return delta, xi_jacobian, PD, beta, xi, errors, iteration_mapping, evaluation_mapping

    def _compute_supply_contributions(self, theta_info, WD, WS, linear_costs, processes, delta, xi_jacobian, beta,
                                      sigma, pi, last_objective_info, compute_gradient):
        """Compute transformed marginal costs and the Jacobian of omega (equivalently, of transformed marginal costs)
        with respect to theta market-by-market. If necessary, revert problematic elements to their last values. Lastly,
        recover gamma and compute omega.
        """
        errors = set()

        # compute the Jacobian of beta with respect to theta, which is needed to compute other Jacobians
        beta_jacobian = None
        if compute_gradient:
            PD = self.products.ZD @ WD @ self.products.ZD.T
            X1PD = self.products.X1.T @ PD
            beta_jacobian = scipy.linalg.solve(X1PD @ self.products.X1, X1PD @ xi_jacobian)

        # construct a mapping from market IDs to market-specific arguments used to compute transformed marginal costs
        #   and their Jacobian
        mapping = {}
        for t in np.unique(self.products.market_ids):
            market_t = SupplyProblemMarket(
                t, self.linear_prices, self.nonlinear_prices, self.products, self.agents, delta, beta=beta, sigma=sigma,
                pi=pi
            )
            last_tilde_costs_t = last_objective_info.tilde_costs[self.products.market_ids.flat == t]
            xi_jacobian_t = xi_jacobian[self.products.market_ids.flat == t]
            mapping[t] = [
                market_t, last_tilde_costs_t, xi_jacobian_t, beta_jacobian, theta_info, linear_costs, compute_gradient
            ]

        # fill transformed marginal costs and their Jacobian market-by-market (the Jacobian will be null if the gradient
        #   isn't being computed)
        tilde_costs = np.zeros((self.N, 1), options.dtype)
        omega_jacobian = np.zeros((self.N, theta_info.P), options.dtype)
        with ParallelItems(SupplyProblemMarket.solve, mapping, processes) as items:
            for t, (tilde_costs_t, omega_jacobian_t, errors_t) in items:
                tilde_costs[self.products.market_ids.flat == t] = tilde_costs_t
                omega_jacobian[self.products.market_ids.flat == t] = omega_jacobian_t
                errors |= errors_t

        # replace invalid transformed marginal costs with their last values
        bad_tilde_costs_indices = ~np.isfinite(tilde_costs)
        if np.any(bad_tilde_costs_indices):
            tilde_costs[bad_tilde_costs_indices] = last_objective_info.tilde_costs[bad_tilde_costs_indices]
            output(f"Number of problematic marginal costs that were reverted: {bad_tilde_costs_indices.sum()}.")

        # replace invalid elements in their Jacobian with their last values
        if compute_gradient:
            bad_jacobian_indices = ~np.isfinite(omega_jacobian)
            if np.any(bad_jacobian_indices):
                omega_jacobian[bad_jacobian_indices] = last_objective_info.omega_jacobian[bad_jacobian_indices]
                output(
                    f"Number of problematic elements in the Jacobian of omega (equivalently, of transformed marginal "
                    f"costs) with respect to theta that were reverted: {bad_jacobian_indices.sum()}."
                )

        # recover gamma and compute omega
        PS = self.products.ZS @ WS @ self.products.ZS.T
        X3PS = self.products.X3.T @ PS
        gamma = scipy.linalg.solve(X3PS @ self.products.X3, X3PS @ tilde_costs)
        omega = tilde_costs - self.products.X3 @ gamma
        return tilde_costs, omega_jacobian, PS, gamma, omega, errors


class NonlinearParameter(object):
    """Information about a single nonlinear parameter."""

    def __init__(self, location, bounds):
        """Store the information and determine whether the parameter is fixed or unfixed."""
        self.location = location
        self.lb = bounds[0][location]
        self.ub = bounds[1][location]
        self.value = self.lb if self.lb == self.ub else None

    def get_characteristics(self, products, agents):
        """Get the product and agents characteristics associated with the parameter."""
        raise NotImplementedError


class SigmaParameter(NonlinearParameter):
    """Information about a single parameter in sigma."""

    def get_characteristics(self, products, agents):
        """Get the product and agents characteristics associated with the parameter."""
        return products.X2[:, [self.location[0]]], agents.nodes[:, [self.location[1]]]


class PiParameter(NonlinearParameter):
    """Information about a single parameter in pi."""

    def get_characteristics(self, products, agents):
        """Get the product and agents characteristics associated with the parameter."""
        return products.X2[:, [self.location[0]]], agents.demographics[:, [self.location[1]]]


class ThetaInfo(object):
    """Information about nonlinear parameters, which relates sigma and pi to theta."""

    def __init__(self, problem, sigma, pi, sigma_bounds, pi_bounds, supports_bounds):
        """Validate initial parameter matrices and their bounds. Then, construct lists of information about fixed (equal
        bounds) and unfixed (unequal bounds) elements of sigma and pi.
        """
        self.problem = problem

        # validate and clean up sigma
        sigma = np.asarray(sigma, options.dtype)
        if sigma.shape != (problem.K2, problem.K2):
            raise ValueError(f"sigma must have {problem.K2} rows and columns.")
        sigma[np.tril_indices(problem.K2, -1)] = 0

        # validate and clean up pi
        if (pi is None) != (problem.D == 0):
            raise ValueError("pi should be None only when there are no demographics.")
        if pi is not None:
            pi = np.asarray(pi, options.dtype)
            if pi.shape != (problem.K2, problem.D):
                raise ValueError(f"pi must have {problem.K2} rows and {problem.D} columns.")

        # construct default sigma bounds or validate specified bounds
        if sigma_bounds is None or not supports_bounds:
            sigma_bounds = (
                np.full_like(sigma, -np.inf, options.dtype),
                np.full_like(sigma, +np.inf, options.dtype)
            )
            if supports_bounds:
                np.fill_diagonal(sigma_bounds[0], 0)
        else:
            if len(sigma_bounds) != 2:
                raise ValueError("sigma_bounds must be a tuple of the form (lb, ub).")
            sigma_bounds = [np.asarray(b, options.dtype).copy() for b in sigma_bounds]
            for bounds_index, bounds in enumerate(sigma_bounds):
                bounds[np.isnan(bounds)] = -np.inf if bounds_index == 0 else +np.inf
                if bounds.shape != sigma.shape:
                    raise ValueError(f"sigma_bounds[{bounds_index}] must have the same shape as sigma.")
            if ((sigma < sigma_bounds[0]) | (sigma > sigma_bounds[1])).any():
                raise ValueError("sigma must be within its bounds.")

        # construct default pi bounds or validate specified bounds
        if pi is None:
            pi_bounds = None
        elif pi_bounds is None or not supports_bounds:
            pi_bounds = (np.full_like(pi, -np.inf, options.dtype), np.full_like(pi, +np.inf, options.dtype))
        else:
            if len(pi_bounds) != 2:
                raise ValueError("pi_bounds must be a tuple of the form (lb, ub).")
            pi_bounds = [np.asarray(b, options.dtype).copy() for b in pi_bounds]
            for bounds_index, bounds in enumerate(pi_bounds):
                bounds[np.isnan(bounds)] = -np.inf if bounds_index == 0 else +np.inf
                if bounds.shape != pi.shape:
                    raise ValueError(f"pi_bounds[{bounds_index}] must have the same shape as pi.")
            if ((pi < pi_bounds[0]) | (pi > pi_bounds[1])).any():
                raise ValueError("pi must be within its bounds.")

        # set upper and lower bounds to zero for parameters that are fixed at zero
        sigma_bounds[0][np.where(sigma == 0)] = sigma_bounds[1][np.where(sigma == 0)] = 0
        if pi is not None:
            pi_bounds[0][np.where(pi == 0)] = pi_bounds[1][np.where(pi == 0)] = 0

        # store the initial parameter matrices and their bounds
        self.sigma = sigma
        self.pi = pi
        self.sigma_bounds = sigma_bounds
        self.pi_bounds = pi_bounds

        # store information about individual sigma and pi elements in lists
        self.fixed = []
        self.unfixed = []

        # store information for the upper triangle of sigma
        for location in zip(*np.triu_indices_from(sigma)):
            parameter = SigmaParameter(location, sigma_bounds)
            if parameter.value is None:
                self.unfixed.append(parameter)
            else:
                self.fixed.append(parameter)

        # store information for pi
        if pi_bounds is not None:
            for location in np.ndindex(pi.shape):
                parameter = PiParameter(location, pi_bounds)
                if parameter.value is None:
                    self.unfixed.append(parameter)
                else:
                    self.fixed.append(parameter)

        # store the number of unfixed parameters and make sure that there is at least one of them
        self.P = len(self.unfixed)
        if self.P == 0:
            raise ValueError("There must be at least one unfixed nonlinear parameter.")

    def __str__(self):
        """Format the initial nonlinear parameters and their bounds as a string."""
        return "\n".join([
            "Initial Nonlinear Parameters:",
            self.format_matrices(self.sigma, self.pi),
            "",
            "Lower Bounds on Nonlinear Parameters:",
            self.format_matrices(self.sigma_bounds[0], None if self.pi_bounds is None else self.pi_bounds[0]),
            "",
            "Upper Bounds on Nonlinear Parameters:",
            self.format_matrices(self.sigma_bounds[1], None if self.pi_bounds is None else self.pi_bounds[1])
        ])

    def format_matrices(self, sigma_like, pi_like, sigma_se=None, pi_se=None):
        """Format matrices that are of the same size as the matrices of nonlinear parameters as a string. If matrices
        of standard errors are given, they will be formatted as numbers surrounded by parentheses underneath the
        elements of sigma and pi.
        """
        lines = []

        # construct the parameter table formatter
        sigma_widths = [14] + [max(14, options.digits + 8)] * self.problem.K2
        pi_widths = [] if self.pi is None else [14] + [max(14, options.digits + 8)] * self.problem.D
        line_indices = {} if self.pi is None else {len(sigma_widths) - 1}
        formatter = output.table_formatter(sigma_widths + pi_widths, line_indices)

        # construct the table header
        header = ["Sigma:"]
        if self.problem.nonlinear_prices:
            header.append("Price")
        for nonlinear_index in range(self.problem.K2 - int(self.problem.nonlinear_prices)):
            header.append(f"Nonlinear #{nonlinear_index}")
        if self.pi is not None:
            header.append("Pi:")
            for demographic_index in range(self.problem.D):
                header.append(f"Demographic #{demographic_index}")

        # build the top of the table
        lines.extend([formatter.border(), formatter(header), formatter.lines()])

        # construct the rows containing parameter information
        for row_index in range(self.problem.K2):
            # determine the label of the row
            row_label = f"Nonlinear #{row_index - int(self.problem.nonlinear_prices)}"
            if row_index == 0 and self.problem.nonlinear_prices:
                row_label = "Price"

            # the row of values consists of the label, blanks for the lower triangle of sigma, sigma values, the label
            #   again, and finally pi values
            values_row = [row_label] + [""] * row_index
            for column_index in range(row_index, self.problem.K2):
                values_row.append(output.format_number(sigma_like[row_index, column_index]))
            if self.pi is not None:
                values_row.append(row_label)
                for column_index in range(self.problem.D):
                    values_row.append(output.format_number(pi_like[row_index, column_index]))
            lines.append(formatter(values_row))

            # construct a row of standard errors for unfixed parameters
            if sigma_se is not None:
                # determine which columns in this row correspond to unfixed parameters
                sigma_indices = set()
                pi_indices = set()
                for parameter in self.unfixed:
                    if parameter.location[0] == row_index:
                        if isinstance(parameter, SigmaParameter):
                            sigma_indices.add(parameter.location[1])
                        else:
                            pi_indices.add(parameter.location[1])

                # construct a row similar to the values row without row labels and with standard error formatting
                se_row = [""] * (1 + row_index)
                for column_index in range(row_index, self.problem.K2):
                    se = sigma_se[row_index, column_index]
                    se_row.append(output.format_se(se) if column_index in sigma_indices else "")
                if self.pi is not None:
                    se_row.append("")
                    for column_index in range(self.problem.D):
                        se = pi_se[row_index, column_index]
                        se_row.append(output.format_se(se) if column_index in pi_indices else "")

                # format the row of values and add an additional blank line if there is another row of values
                lines.append(formatter(se_row))
                if row_index < self.problem.K2 - 1:
                    lines.append(formatter.blank())

        # construct the bottom border and combine the lines into one string
        lines.append(formatter.border())
        return "\n".join(lines)

    def compress(self, sigma, pi):
        """Compress nonlinear parameter matrices into theta."""
        sigma_locations = list(zip(*[p.location for p in self.unfixed if isinstance(p, SigmaParameter)]))
        theta = sigma[sigma_locations].ravel()
        if pi is not None:
            pi_locations = list(zip(*[p.location for p in self.unfixed if isinstance(p, PiParameter)]))
            theta = np.r_[theta, pi[pi_locations].ravel()]
        return theta

    def expand(self, theta_like, fill_fixed=False):
        """Recover nonlinear parameter-sized matrices from a vector of the same size as theta. If fill_fixed is True,
        elements corresponding to fixed parameters will be set to their fixed values instead of to None.
        """
        sigma_like = np.full_like(self.sigma, np.nan)
        pi_like = None if self.pi is None else np.full_like(self.pi, np.nan)

        # set values for elements that correspond to unfixed parameters
        for parameter, value in zip(self.unfixed, theta_like):
            if isinstance(parameter, SigmaParameter):
                sigma_like[parameter.location] = value
            else:
                pi_like[parameter.location] = value

        # set values for elements that correspond to fixed parameters
        if fill_fixed:
            sigma_like[np.tril_indices_from(sigma_like, -1)] = 0
            for parameter in self.fixed:
                if isinstance(parameter, SigmaParameter):
                    sigma_like[parameter.location] = parameter.value
                else:
                    pi_like[parameter.location] = parameter.value

        # return the expanded matrices
        return sigma_like, pi_like


class ObjectiveInfo(object):
    """Structured information about a completed iteration of the optimization routine."""

    def __init__(self, problem, theta_info, WD, WS, theta, delta, tilde_costs, xi_jacobian, omega_jacobian, beta=None,
                 gamma=None, xi=None, omega=None, objective=None, gradient=None, iteration_mapping=None,
                 evaluation_mapping=None):
        """Initialize objective information. Optional parameters will not be specified when preparing for the first
        objective evaluation.
        """
        self.problem = problem
        self.theta_info = theta_info
        self.WD = WD
        self.WS = WS
        self.theta = theta
        self.delta = delta
        self.tilde_costs = tilde_costs
        self.xi_jacobian = xi_jacobian
        self.omega_jacobian = omega_jacobian
        self.beta = beta
        self.gamma = gamma
        self.xi = xi
        self.omega = omega
        self.objective = objective
        self.gradient = gradient
        self.iteration_mapping = iteration_mapping
        self.evaluation_mapping = evaluation_mapping
        self.gradient_norm = np.nan if gradient is None else np.abs(gradient).max()

    def format_progress(self, step, current_iterations, current_evaluations, smallest_objective,
                        smallest_gradient_norm):
        """Format optimization progress as a string. The first iteration will include the progress table header. The
        smallest_objective is the smallest objective value encountered so far during optimization.
        """
        lines = []
        header = [
            ("GMM", "Step"), ("Optimization", "Iterations"), ("Objective", "Evaluations"),
            ("Fixed Point", "Iterations"), ("Contraction", "Evaluations"), ("Objective", "Value"),
            ("Objective", "Improvement"), ("Gradient", "Infinity Norm"), ("Gradient", "Improvement")
        ]
        widths = [max(len(k1), len(k2), options.digits + 6 if i > 4 else 0) for i, (k1, k2) in enumerate(header)]
        formatter = output.table_formatter(widths)

        # if this is the first iteration, include the header
        if current_evaluations == 1:
            lines.extend([formatter([k[0] for k in header]), formatter([k[1] for k in header]), formatter.lines()])

        # include the progress update
        objective_improved = np.isfinite(smallest_objective) and self.objective < smallest_objective
        gradient_improved = np.isfinite(smallest_gradient_norm) and self.gradient_norm < smallest_gradient_norm
        lines.append(formatter([
            step,
            current_iterations,
            current_evaluations,
            sum(self.iteration_mapping.values()),
            sum(self.evaluation_mapping.values()),
            output.format_number(self.objective),
            output.format_number(smallest_objective - self.objective) if objective_improved else "",
            output.format_number(self.gradient_norm),
            output.format_number(smallest_gradient_norm - self.gradient_norm) if gradient_improved else "",
        ]))

        # combine the lines into one string
        return "\n".join(lines)

    def to_results(self, *args):
        """Convert this information about an iteration of the optimization routine into full results."""
        from .results import Results
        return Results(self, *args)


class DemandProblemMarket(Market):
    """A single market in the BLP problem, which can be solved to compute delta--related information."""

    def solve(self, initial_delta, theta_info, iteration, linear_fp, compute_gradient):
        """Compute the mean utility for this market that equates market shares to observed values by solving a fixed
        point problem. Then, if compute_gradient is True, compute the Jacobian of xi (equivalently, of delta) with
        respect to theta. If necessary, replace null elements in delta with their last values before computing its
        Jacobian.
        """

        # configure numpy to identify floating point errors
        errors = set()
        with np.errstate(divide='call', over='call', under='ignore', invalid='call'):
            np.seterrcall(lambda *_: errors.add(exceptions.DeltaFloatingPointError))

            # define a custom log wrapper that identifies issues with taking logs
            def custom_log(x):
                with np.errstate(all='ignore'):
                    if np.any(x <= 0):
                        errors.add(exceptions.NonpositiveSharesError)
                    return np.log(x)

            # solve the fixed point problem
            if linear_fp:
                log_shares = np.log(self.products.shares)
                contraction = lambda d: d + log_shares - custom_log(self.compute_probabilities(d) @ self.agents.weights)
                delta, converged, iterations, evaluations = iteration._iterate(initial_delta, contraction)
            else:
                compute_probabilities = functools.partial(self.compute_probabilities, mu=np.exp(self.mu), linear=False)
                contraction = lambda d: d * self.products.shares / (compute_probabilities(d) @ self.agents.weights)
                exp_delta, converged, iterations, evaluations = iteration._iterate(np.exp(initial_delta), contraction)
                delta = custom_log(exp_delta)

            # identify whether the fixed point converged
            if not converged:
                errors.add(exceptions.DeltaConvergenceError)

            # if the gradient is to be computed, replace invalid values in delta with the last computed values before
            #   computing its Jacobian
            xi_jacobian = np.full((self.J, theta_info.P), np.nan, options.dtype)
            if compute_gradient:
                valid_delta = delta.copy()
                bad_delta_indices = ~np.isfinite(delta)
                valid_delta[bad_delta_indices] = initial_delta[bad_delta_indices]
                xi_jacobian = self.compute_xi_by_theta_jacobian(valid_delta, theta_info)
            return delta, xi_jacobian, errors, iterations, evaluations

    def compute_xi_by_theta_jacobian(self, delta, theta_info):
        """Use the Implicit Function Theorem to compute the Jacobian of xi (equivalently, of delta) with respect to
        theta.
        """
        probabilities = self.compute_probabilities(delta)
        shares_by_xi_jacobian = self.compute_shares_by_xi_jacobian(probabilities)
        shares_by_theta_jacobian = self.compute_shares_by_theta_jacobian(probabilities, theta_info)
        try:
            return scipy.linalg.solve(-shares_by_xi_jacobian, shares_by_theta_jacobian)
        except (ValueError, scipy.linalg.LinAlgError):
            return np.full_like(shares_by_theta_jacobian, np.nan)

    def compute_shares_by_xi_jacobian(self, probabilities):
        """Compute the Jacobian of shares with respect to xi (equivalently, to delta)."""
        square_shares = np.diagflat(self.products.shares)
        square_weights = np.diagflat(self.agents.weights)
        return square_shares - probabilities @ square_weights @ probabilities.T

    def compute_shares_by_theta_jacobian(self, probabilities, theta_info):
        """Compute the Jacobian of shares with respect to theta."""
        jacobian = np.zeros((self.J, theta_info.P), options.dtype)
        for p, parameter in enumerate(theta_info.unfixed):
            x, v = parameter.get_characteristics(self.products, self.agents)
            jacobian[:, [p]] = probabilities * v.T * (x - x.T @ probabilities) @ self.agents.weights
        return jacobian


class SupplyProblemMarket(Market):
    """A single market in the BLP problem, which can be solved to compute costs-related information."""

    def solve(self, initial_tilde_costs, xi_jacobian, beta_jacobian, theta_info, linear_costs, compute_gradient):
        """Compute transformed marginal costs for this market. Then, if compute_gradient is True, compute the Jacobian
        of omega (equivalently, of transformed marginal costs) with respect to theta. If necessary, replace null
        elements in transformed marginal costs with their last values before computing their Jacobian.
        """

        # configure numpy to identify floating point errors
        errors = set()
        with np.errstate(divide='call', over='call', under='ignore', invalid='call'):
            np.seterrcall(lambda *_: errors.add(exceptions.CostsFloatingPointError))

            # compute marginal costs
            try:
                costs = self.compute_costs()
            except scipy.linalg.LinAlgError:
                errors.add(exceptions.CostsSingularityError)
                costs = np.full((self.J, 1), np.nan, options.dtype)

            # take the log of marginal costs under a log-linear specification
            tilde_costs = costs
            if not linear_costs:
                with np.errstate(all='ignore'):
                    if np.any(costs <= 0):
                        errors.add(exceptions.NonpositiveCostsError)
                    tilde_costs = np.log(costs)

            # if the gradient is to be computed, replace invalid transformed marginal costs with their last computed
            #   values before computing their Jacobian
            omega_jacobian = np.full((self.J, theta_info.P), np.nan, options.dtype)
            if compute_gradient:
                valid_tilde_costs = tilde_costs.copy()
                bad_costs_indices = ~np.isfinite(tilde_costs)
                valid_tilde_costs[bad_costs_indices] = initial_tilde_costs[bad_costs_indices]
                omega_jacobian = self.compute_omega_by_theta_jacobian(
                    valid_tilde_costs, xi_jacobian, beta_jacobian, theta_info, linear_costs
                )
            return tilde_costs, omega_jacobian, errors

    def compute_omega_by_theta_jacobian(self, tilde_costs, xi_jacobian, beta_jacobian, theta_info, linear_costs):
        """Compute the Jacobian of omega (equivalently, of transformed marginal costs) with respect to theta."""
        costs_jacobian = -self.compute_eta_by_theta_jacobian(xi_jacobian, beta_jacobian, theta_info)
        if linear_costs:
            return costs_jacobian
        return costs_jacobian / np.exp(tilde_costs)

    def compute_eta_by_theta_jacobian(self, xi_jacobian, beta_jacobian, theta_info):
        """Compute the Jacobian of the markup term in the BLP-markup equation with respect to theta."""

        # compute the intermediate matrix V that shows up in the decomposition of eta
        probabilities = self.compute_probabilities()
        derivatives = self.compute_utility_by_prices_derivatives()
        V = probabilities * derivatives

        # compute the matrix A, which, when inverted and multiplied by shares, gives eta (negative the intra-firm
        #   Jacobian of shares with respect to prices)
        ownership_matrix = self.get_ownership_matrix()
        square_weights = np.diagflat(self.agents.weights)
        capital_gamma = V @ square_weights @ probabilities.T
        capital_lambda = np.diagflat(V @ self.agents.weights)
        A = ownership_matrix * (capital_gamma - capital_lambda)

        # compute the inverse of A and use it to compute eta
        try:
            A_inverse = scipy.linalg.inv(A)
        except (ValueError, scipy.linalg.LinAlgError):
            A_inverse = np.full_like(A, np.nan)
        eta = A_inverse @ self.products.shares

        # compute the tensor derivative of V with respect to xi (equivalently, to delta), indexed with the first axis
        probabilities_by_xi_tensor = -probabilities[np.newaxis] * probabilities[np.newaxis].swapaxes(0, 1)
        probabilities_by_xi_tensor[np.diag_indices(self.J)] += probabilities
        V_by_xi_tensor = probabilities_by_xi_tensor * derivatives

        # compute the tensor derivative of A with respect to xi
        capital_gamma_by_xi_tensor = (
            V @ square_weights @ probabilities_by_xi_tensor.swapaxes(1, 2) +
            V_by_xi_tensor @ square_weights @ probabilities.T
        )
        capital_lambda_by_xi_tensor = np.zeros_like(capital_gamma_by_xi_tensor)
        V_by_xi_tensor_times_weights = np.squeeze(V_by_xi_tensor @ self.agents.weights)
        capital_lambda_by_xi_tensor[:, np.arange(self.J), np.arange(self.J)] = V_by_xi_tensor_times_weights
        A_by_xi_tensor = ownership_matrix[np.newaxis] * (capital_gamma_by_xi_tensor - capital_lambda_by_xi_tensor)

        # compute the product of the tensor and eta
        A_by_xi_tensor_times_eta = np.squeeze(A_by_xi_tensor @ eta)

        # fill the Jacobian of eta with respect to theta parameter-by-parameter
        eta_jacobian = np.zeros((self.J, theta_info.P), options.dtype)
        for p, parameter in enumerate(theta_info.unfixed):
            # compute the tangent of V with respect to the parameter
            X1_index, X2_index = self.get_price_indices()
            x, v = parameter.get_characteristics(self.products, self.agents)
            probabilities_tangent = probabilities * v.T * (x - x.T @ probabilities)
            V_tangent = probabilities_tangent * derivatives
            if X1_index is not None:
                V_tangent += probabilities * beta_jacobian[X1_index, p]
            if X2_index == parameter.location[0]:
                V_tangent += probabilities * v.T

            # compute the tangent of A with respect to the parameter
            capital_gamma_tangent = (
                V @ square_weights @ probabilities_tangent.T +
                V_tangent @ square_weights @ probabilities.T
            )
            capital_lambda_tangent = np.diagflat(V_tangent @ self.agents.weights)
            A_tangent = ownership_matrix * (capital_gamma_tangent - capital_lambda_tangent)

            # extract the tangent of xi with respect to the parameter and compute the associated tangent of eta
            eta_jacobian[:, [p]] = -A_inverse @ (A_tangent @ eta + A_by_xi_tensor_times_eta.T @ xi_jacobian[:, [p]])

        # return the filled Jacobian
        return eta_jacobian
