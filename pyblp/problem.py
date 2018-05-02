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

            - **prices** : (`numeric`) - Prices, :math:`p`, which will always be included in :math:`X_1`. If
              `nonlinear_prices` is `True`, they will be included in :math:`X_2` as well.

            - **linear_characteristics** : (`numeric, optional`) - Non-price product characteristics that constitute the
              remaining columns in :math:`X_1`.

            - **nonlinear_characteristics** : (`numeric, optional`) - Non-price product characteristics that constitute
              the remaining columns in :math:`X_2` if `nonlinear_prices` is `True`, and the entirety of the matrix if it
              is `False`.

            - **cost_characteristics** : (`numeric, optional`) - Cost product characteristics, :math:`X_3`. This field is
              required if `supply_instruments` is specified, since they will be used to estimate the supply side of the
              problem. If unspecified, only the demand side will be estimated.

            - **demand_instruments** : (`numeric`) - Demand-side instruments, :math:`Z_D`, which should contain the sets
              of columns in `linear_characteristics` and `nonlinear_characteristics`.

            - **supply_instruments** : (`numeric, optional`) - Supply-side instruments, :math:`Z_S`, which should contain
              `cost_characteristics`. This field is required if `cost_characteristics` is specified, since they will be
              used to estimate the supply side of the problem. If unspecified, only the demand side will be estimated.

    agent_data : `structured array-like, optional`
        Required if `integration` is unspecified. Each row corresponds to an agent. Markets can have differing numbers
        of agents. Fields:

            - **market_ids** : (`object`) - IDs that associate agents with markets. The set of distinct IDs should be the
              same as the set of distinct IDs in the `market_ids` field of `product_data`. If `integration` is
              specified, there must be at least as many rows in each market as the number of nodes and weights that are
              built for each market.

            - **weights** : (`numeric, optional`) - Integration weights, :math:`w`. This field is required if
              `integration` is unspecified.

            - **nodes** : (`numeric, optional`) - Unobserved agent characteristics called integration nodes, :math:`\nu`.
              This field is required if `integration` is unspecified. If there are more than :math:`K_2` columns, only
              the first :math:`K_2` will be used.

            - **demographics** : (`numeric, optional`) - Observed agent characteristics called demographics, :math:`d`. If
              `integration` is specified and there are more rows of demographics in a market :math:`t` than :math:`I_t`,
              the number node and weight rows built for that market, only the first :math:`I_t` rows of demographics
              will be used.

    integration : `Integration, optional`
        :class:`Integration` configuration for how to build nodes and weights for integration over agent utilities,
        which is required if the `nodes` and `weights` fields of `agent_data` are unspecified. If they are specified,
        the nodes and weights will be build according to this configuration for each market, and they will replace those
        in `agent_data`.
    nonlinear_prices : `bool, optional`
        Whether prices will be a nonlinear characteristic in :math:`X_2` in addition to being a linear characteristic in
        :math:`X_1`.

    Attributes
    ----------
    products : `Products`
        Restructured `product_data` from :class:`Problem` initialization, which is an instance of
        :class:`primitives.Products`.
    agents : `Agents`
        Restructured `agent_data` from :class:`Problem` initialization with nodes and weights built according to
        `integration` if it is specified, which is an instance of :class:`primitives.Agents`.
    nonlinear_prices : `bool`
        Whether prices are a nonlinear characteristic in :math:`X_2` in addition to being a linear characteristic in
        :math:`X_1`.
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

    Example
    -------
    The following code initializes a problem with the automobile product data from
    :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>` and agent data consisting of random draws used by
    :ref:`Knittel and Metaxoglou (2014) <km14>`. The configuration is the same as in
    :ref:`Knittel and Metaxoglou (2014) <km14>`:

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
           'firm_ids': data['firm_ids'],
           'shares': data['shares'],
           'prices': data['prices'],
           'linear_characteristics': linear,
           'nonlinear_characteristics': linear[:, :-1],
           'demand_instruments': instruments
       }
       agent_data = np.recfromcsv(pyblp.data.BLP_AGENTS_LOCATION)
       problem = pyblp.Problem(product_data, agent_data)

    After choosing to optimize over the diagonal variance elements in :math:`\Sigma` and choosing starting values, the
    initialized problem can be solved. For simplicity, the following code halts estimation after one GMM step:

    .. ipython:: python

       initial_sigma = np.diag(np.ones(5))
       results = problem.solve(initial_sigma, steps=1)
       results

    """

    def __init__(self, product_data, agent_data=None, integration=None, nonlinear_prices=True):
        """Structure and validate data before computing matrix dimensions."""
        self.products = Products(product_data, nonlinear_prices)
        self.agents = Agents(self.products, agent_data, integration)
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

    def solve(self, sigma, pi=None, sigma_bounds=None, pi_bounds=None, delta=None, WD=None, WS=None, steps=2,
              optimization=None, custom_display=True, error_behavior='revert', error_punishment=1, iteration=None,
              linear_fp=True, linear_costs=True, center_moments=True, se_type='robust', processes=1):
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
            Configuration for :math:`\Pi` bounds of the form ``(lb, ub)``, in which both ``lb`` are `ub` are of the same
            size as `pi`. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its counterpart in
            `pi`. If `optimization` does not support bounds, these will be ignored. By default, if bounds are supported,
            all unfixed elements are unbounded.

            Lower and upper bounds corresponding to zeros in `pi` are set to zero. Setting a lower bound equal to an
            upper bound fixes the corresponding element. Both ``None`` and ``numpy.nan`` are converted to ``-numpy.inf``
            in ``lb`` and to ``numpy.inf`` in ``ub``.

        delta : `array-like, optional`
            Starting values for the mean utility, :math:`\delta`. By default,
            :math:`\delta_{jt} = \log s_{jt} - \log s_{0t}` is used.
        WD : `array-like, optional`
            Starting values for the demand-side weighting matrix, :math:`W_D`. By default, the 2SLS weighting matrix
            :math:`W_D = (Z_D'Z_D)^{-1}`.
        WS : `array-like, optional`
            Starting values for the supply-side weighting matrix, :math:`W_S`, which are only used if the problem was
            initialized with supply-side data. By default, the 2SLS weighting matrix :math:`W_S = (Z_S'Z_S)^{-1}`.
        steps : `int, optional`
            Number of GMM steps. By default, two-step GMM is used.
        optimization : `Optimization, optional`
            :class:`Optimization` configuration for how to solve the optimization problem in each GMM step. By default,
            ``Optimization('l-bfgs-b')`` is used. Routines that do not support bounds will ignore `sigma_bounds` and
            `pi_bounds`. Choosing a routine that does not use analytic gradients will slow down estimation.
        custom_display : `bool, optional`
            Whether to output a custom display of optimization progress. If this is ``False``, the default display for
            the routine configured by `optimization` will be used. Default displays may contain some extra information,
            but compared to other output will look very different, so the default is ``True``.
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
            :math:`\delta(\hat{\theta})` in each market. By default, ``Iteration('squarem')`` is used.
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
            The number of Python processes that will be used during estimation. By default, multiprocessing will not be
            used. For values greater than one, a pool of that many Python processes will be created during each
            iteration of the optimization routine. Market-by-market computation of :math:`\delta(\hat{\theta})` and
            its Jacobian will be distributed among these processes. Using multiprocessing will only improve estimation
            speed if gains from parallelization outweigh overhead from creating the process pool.

        Returns
        -------
        `Results`
            :class:`Results` of the solved problem.

        """

        # configure or validate optimization and integration
        if optimization is None:
            optimization = Optimization('l-bfgs-b')
        if iteration is None:
            iteration = Iteration('squarem')
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
        output(optimization)
        output(f"Error behavior: {error_behavior}.")
        output(f"Error punishment: {output.format_number(error_punishment)}.")
        output(iteration)
        if self.K3 > 0:
            output(f"Linear marginal cost specification: {linear_costs}.")
        output(f"Linear fixed point formulation: {linear_fp}.")
        output(f"Centering sample moments before updating weighting matrices: {center_moments}.")
        output(f"Standard error type: {se_type}.")
        output(f"Processes: {processes}.")

        # compress sigma and pi into theta but retain information about the original matrices
        parameter_info = ParameterInfo(self, sigma, pi, sigma_bounds, pi_bounds, optimization._supports_bounds)
        theta = parameter_info.compress(parameter_info.sigma, parameter_info.pi)
        output(f"Unfixed nonlinear parameters: {theta.size}.")
        output("")
        output(parameter_info)
        output("")

        # construct or validate the demand-side weighting matrix
        if WD is None:
            output("Starting with the 2SLS demand-side weighting matrix.")
            WD = scipy.linalg.inv(self.products.ZD.T @ self.products.ZD)
        else:
            output("Starting with the specified demand-side weighting matrix.")
            WD = np.asarray(WD, dtype=options.dtype)
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
            WS = np.asarray(WS, dtype=options.dtype)
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
            delta = np.c_[np.asarray(delta, dtype=options.dtype)]
            if delta.shape != (self.N, 1):
                raise ValueError(f"delta must be a vector with {self.N} elements.")

        # initialize the Jacobian of delta as all zeros and initialize marginal costs as prices
        jacobian = np.zeros((self.N, theta.size), dtype=options.dtype)
        tilde_costs = self.products.prices if linear_costs else np.log(self.products.prices)

        # iterate through each step
        for step in range(1, steps + 1):
            # wrap computation of objective information with step-specific information
            compute_step_info = functools.partial(
                self._compute_objective_info, parameter_info, WD, WS, error_behavior, error_punishment, iteration,
                linear_costs, linear_fp, processes
            )

            # define the objective function for the optimization routine, which will also store optimization progress
            def objective_function(new_theta):
                info = compute_step_info(new_theta, objective_function.last_info, optimization._compute_gradient)
                if custom_display:
                    output(info.format_progress(objective_function.iterations, objective_function.smallest))
                objective_function.iterations += 1
                objective_function.smallest = min(objective_function.smallest, info.objective)
                objective_function.last_info = info
                return (info.objective, info.gradient) if optimization._compute_gradient else info.objective

            # initialize optimization progress
            objective_function.iterations = 1
            objective_function.smallest = np.inf
            objective_function.last_info = ObjectiveInfo(
                self, parameter_info, WD, WS, theta, delta, jacobian, tilde_costs
            )

            # optimize theta
            output(f"Starting step {step} out of {steps} ...")
            output("")
            start_time = time.time()
            bounds = [p.bounds for p in parameter_info.unfixed]
            verbose = options.verbose and not custom_display
            theta = optimization._optimize(objective_function, theta, bounds, verbose=verbose)
            end_time = time.time()
            output("")
            output(f"Completed step {step} after {output.format_seconds(end_time - start_time)}.")

            # use objective information computed at the optimal theta to compute results for the step
            output(f"Computing step {step} results ...")
            step_info = compute_step_info(theta, objective_function.last_info, compute_gradient=True)
            results = step_info.to_results(step, center_moments, se_type)
            delta = step_info.delta
            jacobian = step_info.jacobian
            tilde_costs = step_info.tilde_costs
            WD = results.updated_WD
            WS = results.updated_WS
            output("")
            output(results)

            # return results from the last step
            if step == steps:
                return results

    def _compute_objective_info(self, parameter_info, WD, WS, error_behavior, error_punishment, iteration, linear_costs,
                                linear_fp, processes, theta, last_objective_info, compute_gradient):
        """Compute delta and its Jacobian market-by-market. Then, recover beta, compute xi, and use xi to compute the
        demand-side contribution to the GMM objective value. The Jacobian and xi are both used to compute the gradient.
        If the problem was configured with supply-side data, marginal costs are then computed, these are used to recover
        gamma, omega is computed, and omega is used to compute the supply-side contribution to the GMM objective value.
        """

        # expand theta into sigma and pi
        sigma, pi = parameter_info.expand(theta, fill_fixed=True)

        # construct a mapping from market IDs to market-specific arguments used to compute delta and its Jacobian
        mapping = {}
        for t in np.unique(self.products.market_ids):
            market_t = ProblemMarket(t, self.nonlinear_prices, self.products, self.agents, sigma=sigma, pi=pi)
            last_delta_t = last_objective_info.delta[self.products.market_ids.flat == t]
            mapping[t] = [market_t, last_delta_t, parameter_info, iteration, linear_fp, compute_gradient]

        # store any error classes
        errors = set()

        # fill delta and its Jacobian market-by-market (the Jacobian will be null if the objective isn't being computed)
        delta = np.zeros((self.N, 1), dtype=options.dtype)
        jacobian = np.zeros((self.N, theta.size), dtype=options.dtype)
        with ParallelItems(ProblemMarket.solve, mapping, processes) as items:
            for t, (delta_t, jacobian_t, errors_t) in items:
                delta[self.products.market_ids.flat == t] = delta_t
                jacobian[self.products.market_ids.flat == t] = jacobian_t
                errors |= errors_t

        # replace invalid elements in delta and its Jacobian with their last values
        invalid_delta_indices = ~np.isfinite(delta)
        delta[invalid_delta_indices] = last_objective_info.delta[invalid_delta_indices]
        invalid_jacobian_indices = None
        if compute_gradient:
            invalid_jacobian_indices = ~np.isfinite(jacobian)
            jacobian[invalid_jacobian_indices] = last_objective_info.jacobian[invalid_jacobian_indices]

        # recover beta and compute xi
        PD = self.products.ZD @ WD @ self.products.ZD.T
        X1PD = self.products.X1.T @ PD
        beta = scipy.linalg.solve(X1PD @ self.products.X1, X1PD @ delta)
        xi = delta - self.products.X1 @ beta

        # compute the demand-side contribution to the GMM objective value and its gradient
        objective = xi.T @ PD @ xi
        gradient = 2 * (jacobian.T @ PD @ xi) if compute_gradient else None

        # move on to the supply side
        tilde_costs = invalid_tilde_cost_indices = gamma = omega = None
        if WS is not None:
            # fill marginal costs market-by-market
            costs = np.zeros((self.N, 1), dtype=options.dtype)
            for t in np.unique(self.products.market_ids):
                market_t = Market(
                    t, self.nonlinear_prices, self.products, self.agents, delta, beta=beta, sigma=sigma, pi=pi
                )
                try:
                    with np.errstate(all='call'):
                        np.seterrcall(lambda *_: errors.add(exceptions.CostsFloatingPointError))
                        costs_t = market_t.compute_costs()
                except scipy.linalg.LinAlgError:
                    errors.add(exceptions.CostsSingularityError)
                    costs_t = np.full((market_t.J, 1), np.nan)
                costs[self.products.market_ids.flat == t] = costs_t

            # take the log of costs under a log-linear specification
            tilde_costs = costs
            if not linear_costs:
                with np.errstate(invalid='call', divide='call'):
                    np.seterrcall(lambda *_: errors.add(exceptions.NonpositiveCostsError))
                    tilde_costs = np.log(costs)

            # replace invalid elements in marginal costs with their last values
            invalid_tilde_cost_indices = ~np.isfinite(tilde_costs)
            tilde_costs[invalid_tilde_cost_indices] = last_objective_info.tilde_costs[invalid_tilde_cost_indices]

            # recover gamma and compute omega
            PS = self.products.ZS @ WS @ self.products.ZS.T
            X3PS = self.products.X3.T @ PS
            gamma = scipy.linalg.solve(X3PS @ self.products.X3, X3PS @ tilde_costs)
            omega = tilde_costs - self.products.X3 @ gamma

            # add the supply-side contribution to the GMM objective value
            objective += omega.T @ PS @ omega

        # handle any errors that were encountered
        if errors:
            exception = exceptions.MultipleErrors(errors)
            if error_behavior == 'raise':
                raise exception
            output("")
            output(exception)
            if error_behavior == 'revert':
                objective *= error_punishment
                if invalid_delta_indices.any():
                    output(f"Problematic delta elements that were reverted: {invalid_delta_indices.sum()}.")
                if invalid_jacobian_indices is not None and invalid_jacobian_indices.any():
                    output(f"Problematic Jacobian elements that were reverted: {invalid_jacobian_indices.sum()}.")
                if invalid_tilde_cost_indices is not None and invalid_tilde_cost_indices.any():
                    output(f"Problematic marginal costs that were reverted: {invalid_tilde_cost_indices.sum()}.")
            elif error_behavior == 'punish':
                objective = np.array(error_punishment)
                gradient = np.zeros_like(theta)
                output(f"Set the objective to {output.format_number(error_punishment)} and its gradient to all zeros.")
            output("")

        # structure the objective information
        return ObjectiveInfo(
            self, parameter_info, WD, WS, theta, delta, jacobian, tilde_costs, beta, gamma, xi, omega, objective,
            gradient
        )


class Parameter(object):
    """Information about a single nonlinear parameter."""

    def __init__(self, location, bounds, in_sigma=False, in_pi=False):
        """Initialize the information."""
        self.location = location
        self.bounds = bounds
        self.value = bounds[0] if bounds[0] == bounds[1] else None
        self.in_sigma = in_sigma
        self.in_pi = in_pi

    @classmethod
    def from_sigma(cls, location, bounds):
        """Initialize a parameter in sigma."""
        return cls(location, bounds, in_sigma=True)

    @classmethod
    def from_pi(cls, location, bounds):
        """Initialize a parameter in pi."""
        return cls(location, bounds, in_pi=True)


class ParameterInfo(object):
    """Information about nonlinear parameters, which relates sigma and pi to theta."""

    def __init__(self, problem, sigma, pi, sigma_bounds, pi_bounds, supports_bounds):
        """Validate initial parameter matrices and their bounds. Then, construct lists of information about fixed (equal
        bounds) and unfixed (unequal bounds) elements of sigma and pi.
        """
        self.problem = problem

        # validate and clean up sigma
        sigma = np.asarray(sigma, dtype=options.dtype)
        if sigma.shape != (problem.K2, problem.K2):
            raise ValueError(f"sigma must have {problem.K2} rows and columns.")
        sigma[np.tril_indices(problem.K2, -1)] = 0

        # validate and clean up pi
        if (pi is None) != (problem.D == 0):
            raise ValueError("pi should be None only when there are no demographics.")
        if pi is not None:
            pi = np.asarray(pi, dtype=options.dtype)
            if pi.shape != (problem.K2, problem.D):
                raise ValueError(f"pi must have {problem.K2} rows and {problem.D} columns.")

        # construct default sigma bounds or validate specified bounds
        if sigma_bounds is None or not supports_bounds:
            sigma_bounds = (
                np.full_like(sigma, -np.inf, dtype=options.dtype),
                np.full_like(sigma, +np.inf, dtype=options.dtype)
            )
            if supports_bounds:
                np.fill_diagonal(sigma_bounds[0], 0)
        else:
            if len(sigma_bounds) != 2:
                raise ValueError("sigma_bounds must be a tuple of the form (lb, ub).")
            sigma_bounds = [np.asarray(b, dtype=options.dtype).copy() for b in sigma_bounds]
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
            pi_bounds = (np.full_like(pi, -np.inf, dtype=options.dtype), np.full_like(pi, +np.inf, dtype=options.dtype))
        else:
            if len(pi_bounds) != 2:
                raise ValueError("pi_bounds must be a tuple of the form (lb, ub).")
            pi_bounds = [np.asarray(b, dtype=options.dtype).copy() for b in pi_bounds]
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
            parameter = Parameter.from_sigma(location, (sigma_bounds[0][location], sigma_bounds[1][location]))
            parameters = self.unfixed if parameter.value is None else self.fixed
            parameters.append(parameter)

        # store information for pi
        if pi_bounds is not None:
            for location in np.ndindex(pi.shape):
                parameter = Parameter.from_pi(location, (pi_bounds[0][location], pi_bounds[1][location]))
                parameters = self.unfixed if parameter.value is None else self.fixed
                parameters.append(parameter)

        # store the number of unfixed parameters and make sure that there is at least one of them
        self.P = len(self.unfixed)
        if self.P == 0:
            raise ValueError("There must be at least one unfixed nonlinear parameter.")

    def __str__(self):
        """Format the initial nonlinear parameters and their bounds as a string."""
        return "\n".join([
            "Initial nonlinear parameters:",
            self.format_matrices(self.sigma, self.pi),
            "",
            "Lower bounds on nonlinear parameters:",
            self.format_matrices(self.sigma_bounds[0], None if self.pi_bounds is None else self.pi_bounds[0]),
            "",
            "Upper bounds on nonlinear parameters:",
            self.format_matrices(self.sigma_bounds[1], None if self.pi_bounds is None else self.pi_bounds[1])
        ])

    def format_matrices(self, sigma_like, pi_like, sigma_standard_errors=None, pi_standard_errors=None):
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
        header = ["Sigma:"] + (["Price"] if self.problem.nonlinear_prices else [])
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
            if sigma_standard_errors is not None:
                # determine which columns in this row correspond to unfixed parameters
                sigma_indices = set()
                pi_indices = set()
                for parameter in self.unfixed:
                    if parameter.location[0] == row_index:
                        if parameter.in_sigma:
                            sigma_indices.add(parameter.location[1])
                        elif parameter.in_pi:
                            pi_indices.add(parameter.location[1])

                # construct a row similar to the values row without row labels and with standard error formatting
                errors_row = [""] * (1 + row_index)
                for column_index in range(row_index, self.problem.K2):
                    error = sigma_standard_errors[row_index, column_index]
                    errors_row.append(output.format_se(error) if column_index in sigma_indices else "")
                if self.pi is not None:
                    errors_row.append("")
                    for column_index in range(self.problem.D):
                        error = pi_standard_errors[row_index, column_index]
                        errors_row.append(output.format_se(error) if column_index in pi_indices else "")

                # format the row of values and add an additional blank line if there is another row of values
                lines.append(formatter(errors_row))
                if row_index < self.problem.K2 - 1:
                    lines.append(formatter.blank())

        # construct the bottom border and combine the lines into one string
        lines.append(formatter.border())
        return "\n".join(lines)

    def compress(self, sigma, pi):
        """Compress nonlinear parameter matrices into theta."""
        sigma_locations = list(zip(*[p.location for p in self.unfixed if p.in_sigma]))
        if pi is None:
            return sigma[sigma_locations].ravel()
        pi_locations = list(zip(*[p.location for p in self.unfixed if p.in_pi]))
        return np.r_[sigma[sigma_locations].ravel(), pi[pi_locations].ravel()]

    def expand(self, theta_like, fill_fixed=False):
        """Recover nonlinear parameter-sized matrices from a vector of the same size as theta. If fill_fixed is True,
        elements corresponding to fixed parameters will be set to their fixed values instead of to None.
        """
        sigma_like = np.full_like(self.sigma, np.nan)
        pi_like = None if self.pi is None else np.full_like(self.pi, np.nan)

        # set values for elements that correspond to unfixed parameters
        for parameter, value in zip(self.unfixed, theta_like):
            if parameter.in_sigma:
                sigma_like[parameter.location] = value
            elif parameter.in_pi:
                pi_like[parameter.location] = value

        # set values for elements that correspond to fixed parameters
        if fill_fixed:
            sigma_like[np.tril_indices_from(sigma_like, -1)] = 0
            for parameter in self.fixed:
                if parameter.in_sigma:
                    sigma_like[parameter.location] = parameter.value
                elif parameter.in_pi:
                    pi_like[parameter.location] = parameter.value

        # return the expanded matrices
        return sigma_like, pi_like


class ObjectiveInfo(object):
    """Structured information about a completed iteration of the optimization routine."""

    def __init__(self, problem, parameter_info, WD, WS, theta, delta, jacobian, tilde_costs, beta=None, gamma=None,
                 xi=None, omega=None, objective=None, gradient=None):
        """Initialize objective information. Optional parameters will not be specified in the first iteration."""
        self.problem = problem
        self.parameter_info = parameter_info
        self.WD = WD
        self.WS = WS
        self.theta = theta
        self.delta = delta
        self.jacobian = jacobian
        self.tilde_costs = tilde_costs
        self.beta = beta
        self.gamma = gamma
        self.xi = xi
        self.omega = omega
        self.objective = objective
        self.gradient = gradient

    def format_progress(self, iterations, smallest_objective):
        """Format optimization progress as a string. The first iteration will include the progress table header. The
        smallest_objective is the smallest objective value encountered so far during optimization.
        """
        lines = []
        header1 = ["Objective", "Objective", "Objective", "Largest Gradient"]
        header2 = ["Evaluations", "Value", "Improvement", "Magnitude"]
        widths = [max(len(header1[0]), len(header2[0]))]
        widths.extend([max(len(k1), len(k2), options.digits + 6) for k1, k2 in list(zip(header1, header2))[1:]])
        formatter = output.table_formatter(widths)

        # if this is the first iteration, include the header
        if iterations == 1:
            lines.extend([formatter(header1), formatter(header2), formatter.lines()])

        # include the progress update
        improved = np.isfinite(smallest_objective) and self.objective < smallest_objective
        lines.append(formatter([
            iterations,
            output.format_number(self.objective),
            output.format_number(smallest_objective - self.objective) if improved else "",
            output.format_number(None if self.gradient is None else np.abs(self.gradient).max())
        ]))

        # combine the lines into one string
        return "\n".join(lines)

    def to_results(self, step, center_moments, se_type):
        """Convert this information about an iteration of the optimization routine into full results."""
        from .results import Results
        return Results(self, step, center_moments, se_type)


class ProblemMarket(Market):
    """A single market in the BLP problem, which can be solved to compute delta-related information."""

    def solve(self, initial_delta, parameter_info, iteration, linear_fp, compute_gradient):
        """Compute the mean utility for this market that equates market shares to observed values by solving a fixed
        point problem. Then, if compute_gradient is True, compute its Jacobian with respect to theta. Finally, return a
        set of any errors encountered during computation. If necessary, replace null elements in delta with their last
        values before computing its Jacobian.
        """

        # configure numpy to identify floating point errors
        errors = set()
        with np.errstate(all='call'):
            np.seterrcall(lambda *_: errors.add(exceptions.DeltaFloatingPointError))

            # define a custom log wrapper that identifies numerical issues with taking logs
            def log(x):
                with np.errstate(invalid='call', divide='call'):
                    old = np.seterrcall(lambda *_: errors.add(exceptions.NonpositiveSharesError))
                    try:
                        return np.log(x)
                    finally:
                        np.seterrcall(old)

            # solve the fixed point problem
            if linear_fp:
                log_shares = log(self.products.shares)
                contraction = lambda d: d + log_shares - log(self.compute_probabilities(d) @ self.agents.weights)
                delta, converged = iteration._iterate(contraction, initial_delta)
            else:
                compute_probabilities = functools.partial(self.compute_probabilities, mu=np.exp(self.mu), linear=False)
                contraction = lambda d: d * self.products.shares / (compute_probabilities(d) @ self.agents.weights)
                exp_delta, converged = iteration._iterate(contraction, np.exp(initial_delta))
                delta = log(exp_delta)

            # identify whether the fixed point converged
            if not converged:
                errors.add(exceptions.DeltaConvergenceError)

            # return a null Jacobian if the gradient isn't being computed
            if not compute_gradient:
                return delta, np.full((self.J, parameter_info.P), np.nan, dtype=options.dtype), errors

            # replace invalid values in delta with the last computed values before computing the Jacobian
            valid_delta = delta.copy()
            delta_indices = ~np.isfinite(delta)
            valid_delta[delta_indices] = initial_delta[delta_indices]

            # compute the Jacobian of delta with respect to theta
            jacobian = self.compute_delta_by_theta_jacobian(valid_delta, parameter_info)
            return delta, jacobian, errors

    def compute_delta_by_theta_jacobian(self, delta, parameter_info):
        """Compute the Jacobian of delta with respect to theta."""

        # compute the Jacobian of shares with respect to delta
        probabilities = self.compute_probabilities(delta)
        diagonal_shares = np.diagflat(self.products.shares)
        diagonal_weights = np.diagflat(self.agents.weights)
        by_delta = diagonal_shares - probabilities @ diagonal_weights @ probabilities.T

        # compute the Jacobian of shares with respect to theta by iterating over each parameter and checking to see
        #   which characteristic, nodes, and demographics contribute to each partial
        by_theta = np.zeros((self.J, parameter_info.P), dtype=options.dtype)
        for p, parameter in enumerate(parameter_info.unfixed):
            row, column = parameter.location
            x = self.products.X2[:, [row]]
            n = self.agents.nodes[:, [column]] if parameter.in_sigma else self.agents.demographics[:, [column]]
            by_theta[:, [p]] = probabilities * n.T * (x - x.T @ probabilities) @ self.agents.weights

        # use the Implicit Function Theorem to compute the Jacobian of delta with respect to theta
        try:
            return scipy.linalg.solve(-by_delta, by_theta)
        except ValueError:
            return np.full_like(by_theta, np.nan)
