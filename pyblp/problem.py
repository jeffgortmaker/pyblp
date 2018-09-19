"""Configuration of the BLP problem and routines used to solve it."""

import collections
import functools
import time
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import exceptions, options
from .configurations.formulation import Formulation
from .configurations.integration import Integration
from .configurations.iteration import Iteration
from .configurations.optimization import Optimization
from .economy import Economy, Market
from .parameters import NonlinearParameters
from .primitives import Agents, Products
from .results import ProblemResults
from .utilities.basics import (
    Array, Bounds, Error, Groups, TableFormatter, format_number, format_seconds, generate_items, output
)
from .utilities.statistics import IV, compute_2sls_weights


class PrimitiveProblem(Economy):
    """A BLP problem initialized with structured product and agent data."""

    def solve(
            self, sigma: Optional[Any] = None, pi: Optional[Any] = None, rho: Optional[Any] = None,
            sigma_bounds: Optional[Tuple[Any, Any]] = None, pi_bounds: Optional[Tuple[Any, Any]] = None,
            rho_bounds: Optional[Tuple[Any, Any]] = None, delta: Optional[Any] = None, WD: Optional[Any] = None,
            WS: Optional[Any] = None, method: str = '2s', optimization: Optional[Optimization] = None,
            error_behavior: str = 'revert', error_punishment: float = 1, delta_behavior: str = 'last',
            iteration: Optional[Iteration] = None, fp_type: str = 'linear', costs_type: str = 'linear',
            costs_bounds: Optional[Tuple[Any, Any]] = None, center_moments: bool = True, W_type: str = 'robust',
            se_type: str = 'robust') -> ProblemResults:
        r"""Solve the problem.

        The problem is solved in one or more GMM steps. During each step, any unfixed nonlinear parameters in
        :math:`\hat{\theta}` are optimized to minimize the GMM objective value. If all nonlinear parameters are fixed or
        if there are no nonlinear parameters (as in the Logit model), the objective is evaluated once during the step.

        If there are nonlinear parameters, the mean utility, :math:`\delta(\hat{\theta})` is computed market-by-market
        with fixed point iteration. Otherwise, it is computed analytically according to the solution of the Logit model.
        If a supply side is to be estimated, marginal costs, :math:`c(\hat{\theta})`, are also computed
        market-by-market. Linear parameters are then estimated, which are used to recover structural error terms, which
        in turn are used to form the objective value. By default, the objective gradient is computed as well.

        .. note::

           This method supports :func:`parallel` processing. If multiprocessing is used, market-by-market computation of
           :math:`\delta(\hat{\theta})` and, if :math:`X_3` was formulated by `product_formulations` in
           :class:`Problem`, of :math:`\tilde{c}(\hat{\theta})`, along with associated Jacobians, will be distributed
           among the processes.

        Parameters
        ----------
        sigma : `array-like, optional`
            Configuration for which elements in the Cholesky decomposition of the covariance matrix that measures
            agents' random taste distribution, :math:`\Sigma`, are fixed at zero and starting values for the other
            elements, which, if not fixed by `sigma_bounds`, are in the vector of unknown elements, :math:`\theta`. Rows
            and columns correspond to columns in :math:`X_2`, which is formulated according `product_formulations` in
            :class:`Problem`. If :math:`X_2` was not formulated, this should not be specified, since the Logit model
            will be estimated.

            Values below the diagonal are ignored. Zeros are assumed to be zero throughout estimation and nonzeros are,
            if not fixed by `sigma_bounds`, starting values for unknown elements in :math:`\theta`.

        pi : `array-like, optional`
            Configuration for which elements in the matrix of parameters that measures how agent tastes vary with
            demographics, :math:`\Pi`, are fixed at zero and starting values for the other elements, which, if not fixed
            by `pi_bounds`, are in the vector of unknown elements, :math:`\theta`. Rows correspond to the same product
            characteristics as in `sigma`. Columns correspond to columns in :math:`d`, which is formulated according to
            `agent_formulation` in :class:`Problem`. If :math:`d` was not formulated, this should not be specified.

            Zeros are assumed to be zero throughout estimation and nonzeros are, if not fixed by `pi_bounds`, starting
            values for unknown elements in :math:`\theta`.

        rho : `array-like, optional`
            Configuration for which elements in the vector of parameters that measure within nesting group correlation,
            :math:`\rho`, are fixed at zero and starting values for the other elements, which, if not fixed by
            `rho_bounds`, are in the vector of unknown elements, :math:`\theta`. If there is only one element, it
            corresponds to all groups defined by the `nesting_ids` field of `product_data` in :class:`Problem`. If there
            is more than one element, there must be as many elements as :math:`H`, the number of distinct nesting
            groups, and elements correspond to group IDs in the sorted order given by
            :attr:`Problem.unique_nesting_ids`. If nesting IDs were not specified, this should not be specified either.

            Zeros are assumed to be zero throughout estimation and nonzeros are, if not fixed by `rho_bounds`, starting
            values for unknown elements in :math:`\theta`.

        sigma_bounds : `tuple, optional`
            Configuration for :math:`\Sigma` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as `sigma`. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in `sigma`. If `optimization` does not support bounds, these will be ignored.

            By default, if bounds are supported, the diagonal of `sigma` is bounded from below by zero. Conditional on
            :math:`X_2`, :math:`\nu`, and an initial estimate of :math:`\mu`, default bounds for off-diagonal parameters
            are chosen to reduce the chance of overflow.

            Values below the diagonal are ignored. Lower and upper bounds corresponding to zeros in `sigma` are set to
            zero. Setting a lower bound equal to an upper bound fixes the corresponding element. Both ``None`` and
            ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        pi_bounds : `tuple, optional`
            Configuration for :math:`\Pi` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as `pi`. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in `pi`. If `optimization` does not support bounds, these will be ignored.

            By default, if bounds are supported, conditional on :math:`X_2`, :math:`d`, and an initial estimate of
            :math:`\mu`, default bounds are chosen to reduce the chance of overflow.

            Lower and upper bounds corresponding to zeros in `pi` are set to zero. Setting a lower bound equal to an
            upper bound fixes the corresponding element. Both ``None`` and ``numpy.nan`` are converted to ``-numpy.inf``
            in ``lb`` and to ``numpy.inf`` in ``ub``.

        rho_bounds : `tuple, optional`
            Configuration for :math:`\rho` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as `rho`. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in `rho`. If `optimization` does not support bounds, these will be ignored.

            By default, if bounds are supported, all elements are bounded from below by ``0``, which corresponds to the
            simple Logit model. Conditional on an initial estimate of :math:`\mu`, upper bounds are chosen to reduce the
            chance of overflow and are less than ``1`` because larger values are inconsistent with utility maximization.

            Lower and upper bounds corresponding to zeros in `rho` are set to zero. Setting a lower bound equal to an
            upper bound fixes the corresponding element. Both ``None`` and ``numpy.nan`` are converted to ``-numpy.inf``
            in ``lb`` and to ``numpy.inf`` in ``ub``.

        delta : `array-like, optional`
            Initial values for the mean utility, :math:`\delta`. If there are any nonlinear parameters, these are the
            values at which the fixed point iteration routine will start during the first objective evaluation. By
            default, the solution to the sample Logit model is used:

            .. math:: \delta_{jt} = \log s_{jt} - \log s_{0t}.

            If there is nesting, the solution to the nested Logit model under the initial `rho` is used instead:

            .. math:: \delta_{jt} = \log s_{jt} - \log s_{0t} - \rho_{h(j)}\log\frac{s_{jt}}{s_{h(j)t}}

            where

            .. math:: s_{h(j)t} = \sum_{k\in\mathscr{J}_{h(j)t}} s_{kt}.

        WD : `array-like, optional`
            Starting values for the demand-side weighting matrix, :math:`W_D`. By default, the 2SLS weighting matrix,
            :math:`W_D = (Z_D'Z_D)^{-1}`, is used.
        WS : `array-like, optional`
            Starting values for the supply-side weighting matrix, :math:`W_S`, which is only used if :math:`X_3` was
            formulated by `product_formulations` in :class:`Problem`. By default, the 2SLS weighting matrix,
            :math:`W_S = (Z_S'Z_S)^{-1}`, is used.
        method : `str, optional`
            The estimation routine that will be used. The following methods are supported:

                - ``'1s'`` - One-step GMM.

                - ``'2s'`` (default) - Two-step GMM.

            Iterated GMM can be manually implemented by executing single GMM steps in a loop, in which after the first
            iteration, nonlinear parameters and weighting matrices from the last :class:`ProblemResults` are passed as
            arguments.

        optimization : `Optimization, optional`
            :class:`Optimization` configuration for how to solve the optimization problem in each GMM step, which is
            only used if there are unfixed nonlinear parameters over which to optimize. By default,
            ``Optimization('l-bfgs-b')`` is used. Routines that do not support bounds will ignore `sigma_bounds` and
            `pi_bounds`. Choosing a routine that does not use analytic gradients will slow down estimation.
        error_behavior : `str, optional`
            How to handle any errors. For example, it is common to encounter overflow when computing
            :math:`\delta(\hat{\theta})` at a large :math:`\hat{\theta}`. The following behaviors are supported:

                - ``'revert'`` (default) - Revert problematic :math:`\delta(\hat{\theta})` elements to their last
                  computed values and use reverted values to compute :math:`\partial\xi / \partial\theta`, and, if the
                  supply side is considered, to compute both :math:`\tilde{c}(\hat{\theta})` and
                  :math:`\partial\omega / \partial\theta` as well. If there are problematic elements in
                  :math:`\partial\xi / \partial\theta`, :math:`\tilde{c}(\hat{\theta})`, or
                  :math:`\partial\omega / \partial\theta`, revert these to their last computed values as well. If there
                  are problematic elements in the first objective evaluation, revert values in
                  :math:`\delta(\hat{\theta})` to their starting values; in :math:`\tilde{c}(\hat{\theta})`, to prices;
                  and in Jacobians, to zeros. In the unlikely event that the gradient or its objective have problematic
                  elements, revert them as well, and if this happens during the first objective evaluation, revert the
                  objective to ``1e10`` and its gradient to zeros.

                - ``'punish'`` - Set the objective to ``1`` and its gradient to all zeros. This option along with a
                  large `error_punishment` can be helpful for routines that do not use analytic gradients.

                - ``'raise'`` - Raise an exception.

        error_punishment : `float, optional`
            How to scale the GMM objective value after an error. By default, the objective value is not scaled.
        delta_behavior : `str, optional`
            Configuration for the values at which the fixed point computation of :math:`\delta(\hat{\theta})` in each
            market will start. This configuration is only relevant if there are unfixed nonlinear parameters over which
            to optimize. The following behaviors are supported:

                - ``'last'`` (default) - Start at the values of :math:`\delta(\hat{\theta})` computed during the last
                  objective evaluation, or, if this is the first evaluation, at the values configured by `delta`. This
                  behavior tends to speed up computation but may introduce noise into gradient computation, especially
                  when the fixed point iteration tolerance is low.

                - ``'first'`` - Start at the values configured by `delta` during the first GMM step, and at the values
                  computed by the last GMM step for each subsequent step. This behavior is more conservative and will
                  often be slower.

        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem used to compute
            :math:`\delta(\hat{\theta})` in each market. This configuration is only relevant if there are nonlinear
            parameters, since :math:`\delta` can be estimated analytically in the Logit model. By default,
            ``Iteration('squarem', {'tol': 1e-14})`` is used.
        fp_type : `str, optional`
            Configuration for the type of contraction mapping used to compute :math:`\delta(\hat{\theta})`. The
            following types of contraction mappings are supported:

                - ``'linear'`` (default) - Linear contraction mapping,

                  .. math:: \delta_{jt} \leftarrow \delta_{jt} + \log s_{jt} - \log s_{jt}(\delta, \hat{\theta}),

                  which is the standard formulation.

                - ``'nonlinear'`` - Exponentiated version,

                  .. math:: \exp(\delta_{jt}) \leftarrow \exp(\delta_{jt})s_{jt} / s_{jt}(\delta, \hat{\theta}),

                  which can be faster because fewer logarithms need to be calculated. Additionally, when there are no
                  nesting parameters, as in :ref:`Brunner, Heiss, Romahn, and Weiser (2017) <bhrw17>`,
                  :math:`\exp(\delta)` is cancelled out of the numerator in the expression for
                  :math:`s(\delta, \hat{\theta})`, which slightly reduces the computational burden. This formulation can
                  also help mitigate problems stemming from any negative integration weights; however, without
                  conservative parameter bounds, using this formulation may increase the chance of overflow errors.

            This option is only relevant if there are nonlinear parameters, since :math:`\delta` can be estimated
            analytically in the Logit model.

        costs_type : `str, optional`
            Marginal cost specification. The following specifications are supported:

                - ``'linear'`` (default) - Linear specification: :math:`\tilde{c} = c`.

                - ``'log'`` - Log-linear specification: :math:`\tilde{c} = \log c`.

            This specification is only relevant if :math:`X_3` was formulated by `product_formulations` in
            :class:`Problem`.

        costs_bounds : `tuple, optional`
            Configuration for :math:`c` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are floats.
            This is only relevant if :math:`X_3` was formulated by `product_formulations` in :class:`Problem`. By
            default, marginal costs are unbounded.

            When `costs_type` is ``'log'``, nonpositive :math:`c(\hat{\theta})` values can create problems when
            computing :math:`\tilde{c}(\hat{\theta}) = \log c(\hat{\theta})`. One solution is to set ``lb`` to a small
            number. Rows in Jacobians associated with clipped marginal costs will be zero.

            Both ``None`` and ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        center_moments : `bool, optional`
            Whether to center the sample moments before using them to update weighting matrices. By default, sample
            moments are centered. This has no effect if `W_type` is ``'unadjusted'``.
        W_type : `str, optional`
            How to update weighting matrices. This has no effect if `method` is ``'1s'``. Often, `se_type` should be the
            same. The following types are supported:

                - ``'robust'`` (default) - Heteroscedasticity robust weighting matrices.

                - ``'unadjusted'`` - Homoskedastic weighting matrices. Errors are always centered when computing this
                  type of weighting matrix, so `center_moments` has no effect.

                - ``'clustered'`` - Clustered weighting matrices, which account for arbitrary within-group correlation.
                  Clusters must be defined by the `clustering_ids` field of `product_data` in :class:`Problem`.

        se_type : `str, optional`
            How to compute standard errors. Often, `W_type` should be the same. The following types are supported:

                - ``'robust'`` (default) - Heteroscedasticity robust standard errors.

                - ``'unadjusted'`` - Homoskedastic standard errors. Unadjusted standard errors are computed under the
                  assumption that weighting matrices are optimal.

                - ``'clustered'`` - Clustered standard errors, which account for arbitrary within-group correlation.
                  Clusters must be defined by the `clustering_ids` field of `product_data` in :class:`Problem`.

        Returns
        -------
        `ProblemResults`
            :class:`ProblemResults` of the solved problem.

        Examples
        --------
        In this example, we'll first set up the fake cereal problem used in the example for :class:`Problem` from
        :ref:`Nevo (2000) <n00>`.

        .. ipython:: python

           product_data = np.recfromcsv(pyblp.data.NEVO_PRODUCTS_LOCATION, encoding='utf-8')
           agent_data = np.recfromcsv(pyblp.data.NEVO_AGENTS_LOCATION, encoding='utf-8')
           product_formulations = (
               pyblp.Formulation('0 + prices', absorb='C(product_ids)'),
               pyblp.Formulation('1 + prices + sugar + mushy')
           )
           agent_formulation = pyblp.Formulation('0 + income + income_squared + age + child')
           problem = pyblp.Problem(product_formulations, product_data, agent_formulation, agent_data)
           problem

        To solve the problem, we'll use the same starting values as :ref:`Nevo (2000) <n00>`. We'll also use a
        non-default unbounded optimization routine that is similar to the default one in Matlab and we'll halt
        estimation after one GMM step for the sake of speed in this example.

        .. ipython:: python

           sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
           pi = [
              [ 5.4819,  0,      0.2037,  0     ],
              [15.8935, -1.2000, 0,       2.6342],
              [-0.2506,  0,      0.0511,  0     ],
              [ 1.2650,  0,     -0.8091,  0     ]
           ]
           results = problem.solve(sigma, pi, method='1s', optimization=pyblp.Optimization('bfgs'))
           results

        For more examples, refer to the :doc:`Examples </examples>` section.

        """

        # record the amount of time it takes to solve the problem
        step_start_time = time.time()

        # validate the estimation method
        if method not in {'1s', '2s'}:
            raise TypeError("method must be '1s' or '2s'.")

        # configure or validate configurations
        if optimization is None:
            optimization = Optimization('l-bfgs-b')
        if iteration is None:
            iteration = Iteration('squarem', {'tol': 1e-14})
        if not isinstance(optimization, Optimization):
            raise TypeError("optimization must be None or an Optimization instance.")
        if not isinstance(iteration, Iteration):
            raise TypeError("iteration must be None or an Iteration instance.")

        # validate behaviors and types
        if error_behavior not in {'revert', 'punish', 'raise'}:
            raise ValueError("error_behavior must be 'revert', 'punish', or 'raise'.")
        if delta_behavior not in {'last', 'first'}:
            raise ValueError("delta_behavior must be 'last' or 'first'.")
        if fp_type not in {'linear', 'nonlinear'}:
            raise ValueError("fp_type must be 'linear' or 'nonlinear'.")
        if costs_type not in {'linear', 'log'}:
            raise ValueError("costs_type must be 'linear' or 'log'.")
        if W_type not in {'robust', 'unadjusted', 'clustered'}:
            raise ValueError("W_type must be 'robust', 'unadjusted', or 'clustered'.")
        if se_type not in {'robust', 'unadjusted', 'clustered'}:
            raise ValueError("se_type must be 'robust', 'unadjusted', or 'clustered'.")
        if 'clustered' in {W_type, se_type} and 'clustering_ids' not in self.products.dtype.names:
            raise ValueError("W_type or se_type is 'clustered' but clustering_ids were not specified in product_data.")

        # configure or validate costs bounds
        if costs_bounds is None:
            costs_bounds = (-np.inf, +np.inf)
        else:
            if len(costs_bounds) != 2:
                raise ValueError("costs_bounds must be a tuple of the form (lb, ub).")
            costs_bounds = (np.asarray(costs_bounds[0], options.dtype), np.asarray(costs_bounds[1], options.dtype))
            costs_bounds[0][np.isnan(costs_bounds[0])] = -np.inf
            costs_bounds[1][np.isnan(costs_bounds[1])] = +np.inf
            if costs_bounds[0].size != 1:
                raise ValueError(f"The lower bound in costs_bounds is not a float.")
            if costs_bounds[1].size != 1:
                raise ValueError(f"The upper bound in costs_bounds is not a float.")
            if costs_bounds[0] > costs_bounds[1]:
                raise ValueError("The lower bound in costs_bounds cannot be larger than the upper bound.")

        # compress sigma, pi, and rho into theta
        nonlinear_parameters = NonlinearParameters(
            self, sigma, pi, rho, sigma_bounds, pi_bounds, rho_bounds, optimization._supports_bounds
        )
        self._handle_errors(error_behavior, nonlinear_parameters.errors)
        theta = nonlinear_parameters.compress()
        theta_bounds = nonlinear_parameters.compress_bounds()
        if self.K2 > 0 or self.H > 0:
            output("")
            output("Initial Nonlinear Parameters:")
            output(nonlinear_parameters.format())
            output("")
            output("Lower Bounds on Nonlinear Parameters:")
            output(nonlinear_parameters.format_lower_bounds())
            output("")
            output("Upper Bounds on Nonlinear Parameters:")
            output(nonlinear_parameters.format_upper_bounds())

        # compute or load the demand-side weighting matrix
        if WD is None:
            WD, WD_errors = compute_2sls_weights(self.products.ZD)
            self._handle_errors(error_behavior, WD_errors)
        else:
            WD = np.asarray(WD, options.dtype)
            if WD.shape != (self.MD, self.MD):
                raise ValueError(f"WD must have {self.MD} rows and columns.")

        # compute or load the supply-side weighting matrix
        if self.MS == 0:
            WS = np.full((0, 0), np.nan, options.dtype)
        elif WS is None:
            WS, WS_errors = compute_2sls_weights(self.products.ZS)
            self._handle_errors(error_behavior, WS_errors)
        else:
            WS = np.asarray(WS, options.dtype)
            if WS.shape != (self.MS, self.MS):
                raise ValueError(f"WS must have {self.MS} rows and columns.")

        # compute or load initial delta values
        if delta is None:
            true_delta = self._compute_logit_delta(nonlinear_parameters.rho)
        else:
            true_delta = np.c_[np.asarray(delta, options.dtype)]
            if true_delta.shape != (self.N, 1):
                raise ValueError(f"delta must be a vector with {self.N} elements.")

        # initialize marginal costs as prices, which will only be used if there are computation errors during the first
        #   objective evaluation
        true_tilde_costs = np.full((self.N, 0), np.nan, options.dtype)
        if self.K3 > 0:
            if costs_type == 'linear':
                true_tilde_costs = self.products.prices
            else:
                assert costs_type == 'log'
                true_tilde_costs = np.log(self.products.prices)

        # initialize Jacobians of xi and omega with respect to theta as all zeros, which will only be used if there are
        #   computation errors during the first objective evaluation
        xi_jacobian = np.zeros((self.N, nonlinear_parameters.P), options.dtype)
        omega_jacobian = np.full_like(xi_jacobian, 0 if self.K3 > 0 else np.nan, options.dtype)

        # initialize the objective as a large number and its gradient as all zeros, which will only be used if there are
        #   computation errors during the first objective evaluation
        objective = np.array(1e10, options.dtype)
        gradient = np.zeros((nonlinear_parameters.P, 1), options.dtype)

        # iterate over each GMM step
        step = 1
        last_results = None
        while True:
            # initialize IV models for demand- and supply-side linear parameter estimation
            demand_iv = IV(self.products.X1, self.products.ZD, WD)
            supply_iv = IV(self.products.X3, self.products.ZS, WS)
            self._handle_errors(error_behavior, demand_iv.errors + supply_iv.errors)

            # wrap computation of progress information with step-specific information
            compute_step_progress = functools.partial(
                self._compute_progress, nonlinear_parameters, demand_iv, supply_iv, WD, WS, error_behavior,
                error_punishment, delta_behavior, iteration, fp_type, costs_type, costs_bounds
            )

            # initialize optimization progress
            iteration_mappings: List[Dict[Hashable, int]] = []
            evaluation_mappings: List[Dict[Hashable, int]] = []
            smallest_objective = smallest_gradient = np.inf
            last_progress = Progress(
                self, nonlinear_parameters, WD, WS, theta, objective, gradient, true_delta, true_delta,
                true_tilde_costs, xi_jacobian, omega_jacobian
            )

            # define the objective function
            def wrapper(
                    new_theta: Array, current_iterations: int, current_evaluations: int) -> (
                    Union[float, Tuple[float, Array]]):
                """Compute and output progress associated with a single objective evaluation."""
                nonlocal iteration_mappings, evaluation_mappings, smallest_objective, smallest_gradient, last_progress
                assert optimization is not None
                progress = last_progress = compute_step_progress(
                    new_theta, last_progress, optimization._compute_gradient
                )
                iteration_mappings.append(progress.iteration_mapping)
                evaluation_mappings.append(progress.evaluation_mapping)
                formatted_progress = progress.format(
                    optimization, step, current_iterations, current_evaluations, smallest_objective,
                    smallest_gradient
                )
                if formatted_progress:
                    output(formatted_progress)
                smallest_objective = min(smallest_objective, progress.objective)
                smallest_gradient = min(smallest_gradient, progress.gradient_norm)
                return (progress.objective, progress.gradient) if optimization._compute_gradient else progress.objective

            # optimize theta
            iterations = evaluations = 0
            optimization_start_time = optimization_end_time = time.time()
            if nonlinear_parameters.P > 0:
                output("")
                output(f"Starting optimization for step {step} ...")
                output("")
                theta, converged, iterations, evaluations = optimization._optimize(theta, theta_bounds, wrapper)
                status = "completed" if converged else "failed"
                optimization_end_time = time.time()
                optimization_time = optimization_end_time - optimization_start_time
                if not converged:
                    self._handle_errors(error_behavior, [exceptions.ThetaConvergenceError()])
                output("")
                output(f"Optimization {status} after {format_seconds(optimization_time)}.")

            # use progress information computed at the optimal theta to compute results for the step
            output("")
            output(f"Computing results for step {step} ...")
            final_progress = compute_step_progress(theta, last_progress, compute_gradient=nonlinear_parameters.P > 0)
            results = ProblemResults(
                final_progress, last_results, step_start_time, optimization_start_time, optimization_end_time,
                iterations, evaluations + 1, iteration_mappings, evaluation_mappings, costs_type, center_moments,
                W_type, se_type
            )
            self._handle_errors(error_behavior, results._errors)
            output(f"Computed results after {format_seconds(results.total_time - results.optimization_time)}.")
            output("")
            output(results)

            # store the last results and return results from the final step
            last_results = results
            if method != '2s' or step == 2:
                return results

            # update vectors and matrices
            true_delta = results.true_delta
            true_tilde_costs = results.true_tilde_costs
            xi_jacobian = results.xi_by_theta_jacobian
            omega_jacobian = results.omega_by_theta_jacobian
            WD = results.updated_WD
            WS = results.updated_WS
            step += 1
            step_start_time = time.time()

    def _compute_progress(
            self, nonlinear_parameters: NonlinearParameters, demand_iv: IV, supply_iv: IV, WD: Array, WS: Array,
            error_behavior: str, error_punishment: float, delta_behavior: str, iteration: Iteration, fp_type: str,
            costs_type: str, costs_bounds: Bounds, theta: Array, last_progress: 'Progress', compute_gradient: bool) -> (
            'Progress'):
        """Compute demand- and supply-side contributions. Then, form the GMM objective value and its gradient. Finally,
        handle any errors that were encountered before structuring relevant progress information.
        """
        errors: List[Error] = []

        # expand the nonlinear parameters
        sigma, pi, rho = nonlinear_parameters.expand(theta)

        # compute demand-side contributions
        true_delta, xi_jacobian, delta, beta, true_xi, iterations, evaluations, demand_errors = (
            self._compute_demand_contributions(
                nonlinear_parameters, demand_iv, iteration, fp_type, sigma, pi, rho, last_progress, compute_gradient
            )
        )
        errors.extend(demand_errors)

        # compute supply-side contributions
        true_tilde_costs = tilde_costs = true_omega = np.full((self.N, 0), np.nan, options.dtype)
        omega_jacobian = np.full((self.N, nonlinear_parameters.P), np.nan, options.dtype)
        clipped_costs_indices = np.zeros((self.N, 1), np.bool)
        gamma = np.full((self.K3, 1), np.nan, options.dtype)
        if self.K3 > 0:
            true_tilde_costs, omega_jacobian, tilde_costs, gamma, true_omega, clipped_costs_indices, supply_errors = (
                self._compute_supply_contributions(
                    nonlinear_parameters, demand_iv, supply_iv, costs_type, costs_bounds, beta, sigma, pi, rho,
                    true_delta, xi_jacobian, last_progress, compute_gradient
                )
            )
            errors.extend(supply_errors)

        # compute the objective value
        demand_moments = self.products.ZD.T @ true_xi
        supply_moments = self.products.ZS.T @ true_omega
        objective = demand_moments.T @ WD @ demand_moments
        if self.K3 > 0:
            objective += supply_moments.T @ WS @ supply_moments

        # replace the objective with its last value if its computation failed, which is unlikely but possible
        if not np.isfinite(np.squeeze(objective)):
            objective = last_progress.objective
            errors.append(exceptions.ObjectiveReversionError())

        # compute the gradient
        gradient = np.full_like(theta, np.nan, options.dtype)
        if compute_gradient:
            gradient = 2 * ((xi_jacobian.T @ self.products.ZD) @ WD @ demand_moments)
            if self.K3 > 0:
                gradient += 2 * ((omega_jacobian.T @ self.products.ZS) @ WS @ supply_moments)

        # replace any invalid elements in the gradient with their last values
        if compute_gradient:
            bad_indices = ~np.isfinite(gradient)
            if np.any(bad_indices):
                gradient[bad_indices] = last_progress.gradient[bad_indices]
                errors.append(exceptions.GradientReversionError(bad_indices))

        # handle any errors
        if errors:
            if error_behavior == 'raise':
                raise exceptions.MultipleErrors(errors)
            if error_behavior == 'revert':
                objective *= error_punishment
            else:
                assert error_behavior == 'punish'
                objective = np.array(error_punishment)
                if compute_gradient:
                    gradient = np.zeros_like(theta)

        # select the delta that will be used in the next objective evaluation
        if delta_behavior == 'last':
            next_delta = true_delta
        else:
            assert delta_behavior == 'first'
            next_delta = last_progress.next_delta

        # structure progress
        return Progress(
            self, nonlinear_parameters, WD, WS, theta, objective, gradient, next_delta, true_delta, true_tilde_costs,
            xi_jacobian, omega_jacobian, delta, tilde_costs, true_xi, true_omega, beta, gamma, iterations,
            evaluations, clipped_costs_indices, errors
        )

    def _compute_demand_contributions(
            self, nonlinear_parameters: NonlinearParameters, demand_iv: IV, iteration: Iteration, fp_type: str,
            sigma: Array, pi: Array, rho: Array, last_progress: 'Progress', compute_gradient: bool) -> (
            Tuple[Array, Array, Array, Array, Array, Dict[Hashable, int], Dict[Hashable, int], List[Error]]):
        """Compute delta and the Jacobian of xi (equivalently, of delta) with respect to theta market-by-market. If
        necessary, revert problematic elements to their last values. Lastly, recover beta and compute xi.
        """
        errors: List[Error] = []

        # initialize delta and its Jacobian along with fixed point information so that they can be filled
        iterations: Dict[Hashable, int] = {}
        evaluations: Dict[Hashable, int] = {}
        true_delta = np.zeros((self.N, 1), options.dtype)
        xi_jacobian = np.full((self.N, nonlinear_parameters.P), np.nan, options.dtype)

        # when possible and when a gradient isn't needed, compute delta with a closed-form solution
        if self.K2 == 0 and (nonlinear_parameters.P == 0 or not compute_gradient):
            true_delta = self._compute_logit_delta(rho)
        else:
            # define a factory for solving the demand side of problem markets
            def market_factory(s: Hashable) -> Tuple[ProblemMarket, Array, NonlinearParameters, Iteration, str, bool]:
                """Build a market along with arguments used to compute delta and its Jacobian."""
                market_s = ProblemMarket(self, s, sigma, pi, rho)
                initial_delta_s = last_progress.next_delta[self._product_market_indices[s]]
                return market_s, initial_delta_s, nonlinear_parameters, iteration, fp_type, compute_gradient

            # compute delta and its Jacobian market-by-market
            generator = generate_items(self.unique_market_ids, market_factory, ProblemMarket.solve_demand)
            for t, (true_delta_t, xi_jacobian_t, errors_t, iterations[t], evaluations[t]) in generator:
                true_delta[self._product_market_indices[t]] = true_delta_t
                xi_jacobian[self._product_market_indices[t]] = xi_jacobian_t
                errors.extend(errors_t)

        # replace invalid elements in delta with their last values
        bad_indices = ~np.isfinite(true_delta)
        if np.any(bad_indices):
            true_delta[bad_indices] = last_progress.true_delta[bad_indices]
            errors.append(exceptions.DeltaReversionError(bad_indices))

        # replace invalid elements in its Jacobian with their last values
        if compute_gradient:
            bad_indices = ~np.isfinite(xi_jacobian)
            if np.any(bad_indices):
                xi_jacobian[bad_indices] = last_progress.xi_jacobian[bad_indices]
                errors.append(exceptions.XiByThetaJacobianReversionError(bad_indices))

        # absorb any demand-side fixed effects
        delta = true_delta
        if self._absorb_demand_ids is not None:
            delta, delta_errors = self._absorb_demand_ids(delta)
            errors.extend(delta_errors)

        # recover beta and compute xi
        beta, true_xi = demand_iv.estimate(delta)
        return true_delta, xi_jacobian, delta, beta, true_xi, iterations, evaluations, errors

    def _compute_supply_contributions(
            self, nonlinear_parameters: NonlinearParameters, demand_iv: IV, supply_iv: IV, costs_type: str,
            costs_bounds: Bounds, beta: Array, sigma: Array, pi: Array, rho: Array, true_delta: Array,
            xi_jacobian: Array, last_progress: 'Progress', compute_gradient: bool) -> (
            Tuple[Array, Array, Array, Array, Array, Array, List[Error]]):
        """Compute transformed marginal costs and the Jacobian of omega (equivalently, of transformed marginal costs)
        with respect to theta market-by-market. If necessary, revert problematic elements to their last values. Lastly,
        recover gamma and compute omega.
        """
        errors: List[Error] = []

        # compute the Jacobian of beta with respect to theta, which is needed to compute other Jacobians
        beta_jacobian = np.full((self.K1, nonlinear_parameters.P), np.nan, options.dtype)
        if compute_gradient:
            beta_jacobian = demand_iv.estimate(xi_jacobian, residuals=False)

        # initialize transformed marginal costs, their Jacobian, and indices of clipped costs so that they can be filled
        true_tilde_costs = np.zeros((self.N, 1), options.dtype)
        omega_jacobian = np.full((self.N, nonlinear_parameters.P), np.nan, options.dtype)
        clipped_costs_indices = np.zeros((self.N, 1), np.bool)

        # define a factory for solving the supply side of problem markets
        def market_factory(
                s: Hashable) -> Tuple[ProblemMarket, Array, Array, Array, NonlinearParameters, str, Bounds, bool]:
            """Build a market along with arguments used to compute transformed marginal costs and their Jacobian."""
            market_s = ProblemMarket(self, s, sigma, pi, rho, beta, true_delta)
            last_true_tilde_costs_s = last_progress.true_tilde_costs[self._product_market_indices[s]]
            xi_jacobian_s = xi_jacobian[self._product_market_indices[s]]
            return (
                market_s, last_true_tilde_costs_s, xi_jacobian_s, beta_jacobian, nonlinear_parameters,
                costs_type, costs_bounds, compute_gradient
            )

        # compute transformed marginal costs and their Jacobian market-by-market
        generator = generate_items(self.unique_market_ids, market_factory, ProblemMarket.solve_supply)
        for t, (true_tilde_costs_t, omega_jacobian_t, clipped_costs_indices_t, errors_t) in generator:
            true_tilde_costs[self._product_market_indices[t]] = true_tilde_costs_t
            omega_jacobian[self._product_market_indices[t]] = omega_jacobian_t
            clipped_costs_indices[self._product_market_indices[t]] = clipped_costs_indices_t
            errors.extend(errors_t)

        # replace invalid transformed marginal costs with their last values
        bad_indices = ~np.isfinite(true_tilde_costs)
        if np.any(bad_indices):
            true_tilde_costs[bad_indices] = last_progress.true_tilde_costs[bad_indices]
            errors.append(exceptions.CostsReversionError(bad_indices))

        # replace invalid elements in their Jacobian with their last values
        if compute_gradient:
            bad_indices = ~np.isfinite(omega_jacobian)
            if np.any(bad_indices):
                omega_jacobian[bad_indices] = last_progress.omega_jacobian[bad_indices]
                errors.append(exceptions.OmegaByThetaJacobianReversionError(bad_indices))

        # absorb any supply-side fixed effects
        tilde_costs = true_tilde_costs
        if self._absorb_supply_ids is not None:
            tilde_costs, tilde_costs_errors = self._absorb_supply_ids(tilde_costs)
            errors.extend(tilde_costs_errors)

        # recover gamma and compute omega
        gamma, true_omega = supply_iv.estimate(tilde_costs)
        return true_tilde_costs, omega_jacobian, tilde_costs, gamma, true_omega, clipped_costs_indices, errors

    def _compute_logit_delta(self, rho: Array) -> Array:
        """Compute the delta that solves the simple Logit (or nested Logit) model."""
        delta = np.log(self.products.shares)
        for t in self.unique_market_ids:
            shares_t = self.products.shares[self._product_market_indices[t]]
            outside_share_t = 1 - shares_t.sum()
            delta[self._product_market_indices[t]] -= np.log(outside_share_t)
            if self.H > 0:
                groups_t = Groups(self.products.nesting_ids[self._product_market_indices[t]])
                group_shares_t = shares_t / groups_t.expand(groups_t.sum(shares_t))
                if rho.size == 1:
                    rho_t = np.full_like(shares_t, float(rho))
                else:
                    rho_t = groups_t.expand(rho[np.searchsorted(self.unique_nesting_ids, groups_t.unique)])
                delta[self._product_market_indices[t]] -= rho_t * np.log(group_shares_t)
        return delta

    @staticmethod
    def _handle_errors(error_behavior: str, errors: List[Error]) -> None:
        """Either raise or output information about any errors."""
        if errors:
            if error_behavior == 'raise':
                raise exceptions.MultipleErrors(errors)
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")


class Problem(PrimitiveProblem):
    r"""A BLP problem.

    This class is initialized with relevant data and solved with :meth:`Problem.solve`.

    In both `product_data` and `agent_data`, fields with multiple columns can be either matrices or can be broken up
    into multiple one-dimensional fields with column index suffixes that start at zero. For example, if there are three
    columns of demand-side instruments, the `demand_instruments` field in `product_data`, which in this case should be a
    matrix with three columns, can be replaced by three one-dimensional fields: `demand_instruments0`,
    `demand_instruments1`, and `demand_instruments2`.

    Parameters
    ----------
    product_formulations : `Formulation or tuple of Formulation`
        :class:`Formulation` configuration or tuple of up to three :class:`Formulation` configurations for the matrix
        of linear product characteristics, :math:`X_1`, for the matrix of nonlinear product characteristics,
        :math:`X_2`, and for the matrix of cost characteristics, :math:`X_3`, respectively. If the formulation for
        :math:`X_3` is not specified or is ``None``, a supply side will not be estimated. Similarly, if the formulation
        for :math:`X_2` is not specified or is ``None``, the Logit model will be estimated.

        Variable names should correspond to fields in `product_data`. The ``shares`` variable should not be included in
        any of the formulations and ``prices`` should be included in the formulation for :math:`X_1` or :math:`X_2` (or
        both). The `absorb` argument of :class:`Formulation` can be used to absorb fixed effects into :math:`X_1` and
        :math:`X_3`, but not :math:`X_2`.

    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **shares** : (`numeric`) - Market shares, :math:`s`.

            - **prices** : (`numeric`) - Product prices, :math:`p`.

            - **demand_instruments** : (`numeric`) - Demand-side instruments, :math:`Z_D`.

        If a formulation for :math:`X_3` is specified in `product_formulations`, the following fields are also required,
        since they will be used to estimate the supply side of the problem:

            - **firm_ids** : (`object, optional`) - IDs that associate products with firms. Any columns after the first
              can be used to compute post-estimation outputs for firm changes, such as mergers.

            - **supply_instruments** : (`numeric, optional`) - Supply-side instruments, :math:`Z_S`.

        In addition to supply-side estimation, the `firm_ids` field is also needed to compute some post-estimation
        outputs. If `firm_ids` are specified, custom ownership matrices can be specified as well:

            - **ownership** : (`numeric, optional`) - Custom stacked :math:`J_t \times J_t` ownership matrices,
              :math:`O`, for each market :math:`t`, which can be built with :func:`build_ownership`. By default,
              standard ownership matrices are built only when they are needed. If specified, each stack is associated
              with a `firm_ids` column and must have as many columns as there are products in the market with the most
              products.

        To estimate a nested Logit or random coefficients nested Logit (RCNL) model, nesting groups must be specified:

            - **nesting_ids** (`object, optional`) - IDs that associate products with nesting groups. When these IDs are
              specified, `rho` in :meth:`Problem.solve`, the vector of parameters that measure within nesting group
              correlation, must be specified as well.

        Finally, clustering groups can be specified to account for arbitrary within-group correlation while computing
        standard errors and weighting matrices:

            - **clustering_ids** (`object, optional`) - Cluster group IDs, which will be used when estimating standard
              errors and updating weighting matrices if `covariance_type` in :meth:`Problem.solve` is ``'clustered'``.

        Along with `market_ids`, `firm_ids`, `nesting_ids`, `clustering_ids`, and `prices`, the names of any additional
        fields can be used as variables in `product_formulations`.

    agent_formulation : `Formulation, optional`
        :class:`Formulation` configuration for the matrix of observed agent characteristics called demographics,
        :math:`d`, which will only be included in the model if this formulation is specified. Since demographics are
        only used if there are nonlinear product characteristics, this formulation should only be specified if
        :math:`X_2` is formulated in `product_formulations`. Variable names should correspond to fields in `agent_data`.
    agent_data : `structured array-like, optional`
        Each row corresponds to an agent. Markets can have differing numbers of agents. Since simulated agents are only
        used if there are nonlinear product characteristics, agent data should only be specified if :math:`X_2` is
        formulated in `product_formulations`. If agent data are specified, market IDs are required:

            - **market_ids** : (`object`) - IDs that associate agents with markets. The set of distinct IDs should be
              the same as the set of IDs in `product_data`. If `integration` is specified, there must be at least as
              many rows in each market as the number of nodes and weights that are built for each market.

        If `integration` is not specified, the following fields are required:

            - **weights** : (`numeric, optional`) - Integration weights, :math:`w`.

            - **nodes** : (`numeric, optional`) - Unobserved agent characteristics called integration nodes,
              :math:`\nu`. If there are more than :math:`K_2` columns, only the first :math:`K_2` will be used.

        Along with `market_ids`, the names of any additional fields can be used as variables in `agent_formulation`.

    integration : `Integration, optional`
        :class:`Integration` configuration for how to build nodes and weights for integration over agent utilities,
        which will replace any `nodes` and `weights` fields in `agent_data`. This configuration is required if `nodes`
        and `weights` in `agent_data` are not specified. It should not be specified if :math:`X_2` is not formulated
        in `product_formulations`.

    Attributes
    ----------
    product_formulations : `Formulation or tuple of Formulation`
        :class:`Formulation` configurations for :math:`X_1`, :math:`X_2`, and :math:`X_3`, respectively.
    agent_formulation : `Formulation`
        :class:`Formulation` configuration for :math:`d`.
    products : `Products`
        Product data structured as :class:`Products`, which consists of data taken from `product_data` along with
        matrices build according to :attr:`Problem.product_formulations`.
    agents : `Agents`
        Agent data structured as :class:`Agents`, which consists of data taken from `agent_data` or built by
        `integration` along with any demographics formulated by `agent_formulation`.
    unique_market_ids : `ndarray`
        Unique market IDs in product and agent data.
    unique_nesting_ids : `ndarray`
        Unique nesting IDs in product data.
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
    ED : `int`
        Number of absorbed demand-side fixed effects, :math:`E_D`.
    ES : `int`
        Number of absorbed supply-side fixed effects, :math:`E_S`.
    H : `int`
        Number of nesting groups, :math:`H`.

    Example
    -------
    In this example, we'll set up the fake cereal problem from :ref:`Nevo (2000) <n00>`.

    .. ipython:: python

       product_data = np.recfromcsv(pyblp.data.NEVO_PRODUCTS_LOCATION, encoding='utf-8')
       agent_data = np.recfromcsv(pyblp.data.NEVO_AGENTS_LOCATION, encoding='utf-8')
       product_formulations = (
           pyblp.Formulation('0 + prices', absorb='C(product_ids)'),
           pyblp.Formulation('1 + prices + sugar + mushy')
       )
       agent_formulation = pyblp.Formulation('0 + income + income_squared + age + child')
       problem = pyblp.Problem(product_formulations, product_data, agent_formulation, agent_data)
       problem

    We'll solve this example problem in :meth:`Problem.solve`. For more examples, refer to the
    :doc:`Examples </examples>` section.

    """

    def __init__(
            self, product_formulations: Union[Formulation, Sequence[Optional[Formulation]]], product_data: Mapping,
            agent_formulation: Optional[Formulation] = None, agent_data: Optional[Mapping] = None,
            integration: Optional[Integration] = None) -> None:
        """Initialize the underlying economy with product and agent data."""

        # validate and normalize product formulations
        if isinstance(product_formulations, Formulation):
            product_formulations = [product_formulations]
        elif isinstance(product_formulations, collections.Sequence) and len(product_formulations) <= 3:
            product_formulations = list(product_formulations)
        else:
            raise TypeError("product_formulations must be a Formulation instance or a tuple of up to three instances.")
        product_formulations.extend([None] * (3 - len(product_formulations)))

        # initialize the underlying economy with structured product and agent data
        products = Products(product_formulations, product_data)
        agents = Agents(products, agent_formulation, agent_data, integration)
        super().__init__(product_formulations, agent_formulation, products, agents)


class Progress(object):
    """Structured information about estimation progress."""

    problem: PrimitiveProblem
    nonlinear_parameters: NonlinearParameters
    WD: Array
    WS: Array
    theta: Array
    objective: Array
    gradient: Array
    next_delta: Array
    true_delta: Array
    true_tilde_costs: Array
    xi_jacobian: Array
    omega_jacobian: Array
    delta: Optional[Array]
    tilde_costs: Optional[Array]
    true_xi: Optional[Array]
    true_omega: Optional[Array]
    beta: Optional[Array]
    gamma: Optional[Array]
    iteration_mapping: Dict[Hashable, int]
    evaluation_mapping: Dict[Hashable, int]
    errors: List[Error]
    gradient_norm: Array

    def __init__(
            self, problem: PrimitiveProblem, nonlinear_parameters: NonlinearParameters, WD: Array, WS: Array,
            theta: Array, objective: Array, gradient: Array, next_delta: Array, true_delta: Array,
            true_tilde_costs: Array, xi_jacobian: Array, omega_jacobian: Array, delta: Optional[Array] = None,
            tilde_costs: Optional[Array] = None, true_xi: Optional[Array] = None, true_omega: Optional[Array] = None,
            beta: Optional[Array] = None, gamma: Optional[Array] = None,
            iteration_mapping: Optional[Dict[Hashable, int]] = None,
            evaluation_mapping: Optional[Dict[Hashable, int]] = None, clipped_costs_indices: Optional[Array] = None,
            errors: Optional[List[Error]] = None) -> None:
        """Initialize progress information. Optional parameters will not be specified when preparing for the first
        objective evaluation.
        """
        self.problem = problem
        self.nonlinear_parameters = nonlinear_parameters
        self.WD = WD
        self.WS = WS
        self.theta = theta
        self.objective = objective
        self.gradient = gradient
        self.next_delta = next_delta
        self.true_delta = true_delta
        self.true_tilde_costs = true_tilde_costs
        self.xi_jacobian = xi_jacobian
        self.omega_jacobian = omega_jacobian
        self.delta = delta
        self.tilde_costs = tilde_costs
        self.true_xi = true_xi
        self.true_omega = true_omega
        self.beta = beta
        self.gamma = gamma
        self.iteration_mapping = iteration_mapping or {}
        self.evaluation_mapping = evaluation_mapping or {}
        self.clipped_costs_indices = clipped_costs_indices
        self.errors = errors or []
        with np.errstate(invalid='ignore'):
            self.gradient_norm = np.array(np.nan, options.dtype) if gradient.size == 0 else np.abs(gradient).max()

    def format(
            self, optimization: Optimization, step: int, current_iterations: int, current_evaluations: int,
            smallest_objective: Array, smallest_gradient: Array) -> str:
        """Format a universal display of optimization progress as a string. The first iteration will include the
        progress table header. If there are any errors, information about them will be formatted as well, regardless of
        whether or not a universal display is to be used. The smallest_objective is the smallest objective value
        encountered so far during optimization.
        """

        # build the header of the universal display
        header = [
            ("GMM", "Step"), ("Optimization", "Iterations"), ("Objective", "Evaluations"),
            ("Fixed Point", "Iterations"), ("Contraction", "Evaluations"), ("Objective", "Value"),
            ("Objective", "Improvement")
        ]
        if optimization._compute_gradient:
            header.extend([("Gradient", "Infinity Norm"), ("Gradient", "Improvement")])
        header.append(("", "Theta"))

        # build the formatter of the universal display
        widths = [max(len(k1), len(k2), options.digits + 6 if i > 4 else 0) for i, (k1, k2) in enumerate(header[:-1])]
        widths.append(max(len(header[-1][0]), len(header[-1][1]), self.theta.size * (options.digits + 8) - 2))
        formatter = TableFormatter(widths)

        # if this is the first iteration, include the header
        lines: List[str] = []
        if optimization._universal_display and current_evaluations == 1:
            lines.extend([formatter([k[0] for k in header]), formatter([k[1] for k in header], underline=True)])

        # include information about any errors
        if self.errors:
            preamble = (
                "At least one error was encountered. As long as the optimization routine does not get stuck at values "
                "of theta that give rise to errors, this is not necessarily a problem. If the errors persist or seem "
                "to be impacting the optimization results, consider setting an error punishment or following any of "
                "the other suggestions below:"
            )
            lines.extend(["", preamble, str(exceptions.MultipleErrors(self.errors)), ""])

        # include the progress update
        if optimization._universal_display:
            objective_improved = np.isfinite(smallest_objective) and self.objective < smallest_objective
            gradient_improved = np.isfinite(smallest_gradient) and self.gradient_norm < smallest_gradient
            values = [
                step,
                current_iterations,
                current_evaluations,
                sum(self.iteration_mapping.values()),
                sum(self.evaluation_mapping.values()),
                format_number(float(self.objective)),
                format_number(float(smallest_objective - self.objective)) if objective_improved else "",
            ]
            if optimization._compute_gradient:
                values.extend([
                    format_number(float(self.gradient_norm)),
                    format_number(float(smallest_gradient - self.gradient_norm)) if gradient_improved else "",
                ])
            values.append(", ".join(format_number(x) for x in self.theta))
            lines.append(formatter(values))

        # combine the lines into one string
        return "\n".join(lines)


class ProblemMarket(Market):
    """A single market in a problem."""

    def solve_demand(
            self, initial_delta: Array, nonlinear_parameters: NonlinearParameters, iteration: Iteration, fp_type: str,
            compute_gradient: bool) -> Tuple[Array, Array, List[Error], int, int]:
        """Compute the mean utility for this market that equates market shares to observed values by solving a fixed
        point problem. Then, if compute_gradient is True, compute the Jacobian of xi (equivalently, of delta) with
        respect to theta. If necessary, replace null elements in delta with their last values before computing its
        Jacobian.
        """
        errors: List[Error] = []

        # configure NumPy to identify floating point errors
        with np.errstate(divide='call', over='call', under='ignore', invalid='call'):
            np.seterrcall(lambda *_: errors.append(exceptions.DeltaFloatingPointError()))

            # compute delta either with a closed-form solution or by solving a fixed point problem
            if self.K2 == 0:
                assert self.H > 0
                converged = True
                iterations = evaluations = 0
                outside_share = 1 - self.products.shares.sum()
                group_shares = self.products.shares / self.groups.expand(self.groups.sum(self.products.shares))
                delta = np.log(self.products.shares) - np.log(outside_share) - self.rho * np.log(group_shares)
            elif fp_type == 'linear':
                log_shares = np.log(self.products.shares)
                contraction = lambda d: d + log_shares - np.log(self.compute_probabilities(d) @ self.agents.weights)
                delta, converged, iterations, evaluations = iteration._iterate(initial_delta, contraction)
            else:
                assert fp_type == 'nonlinear'
                exp_mu = np.exp(self.mu)
                compute_probabilities = functools.partial(self.compute_probabilities, mu=exp_mu, linear=False)
                if self.H > 0:
                    contraction = lambda d: d * self.products.shares / (compute_probabilities(d) @ self.agents.weights)
                else:
                    compute_quotient = functools.partial(compute_probabilities, numerator=exp_mu)
                    contraction = lambda d: self.products.shares / (compute_quotient(d) @ self.agents.weights)
                exp_delta, converged, iterations, evaluations = iteration._iterate(np.exp(initial_delta), contraction)
                delta = np.log(exp_delta)
            if not converged:
                errors.append(exceptions.DeltaConvergenceError())

        # if the gradient is to be computed, replace invalid values in delta with the last computed values before
        #   computing its Jacobian
        xi_jacobian = np.full((self.J, nonlinear_parameters.P), np.nan, options.dtype)
        if compute_gradient:
            valid_delta = delta.copy()
            bad_delta_indices = ~np.isfinite(delta)
            valid_delta[bad_delta_indices] = initial_delta[bad_delta_indices]
            xi_jacobian, jacobian_errors = self.compute_xi_by_theta_jacobian(nonlinear_parameters, valid_delta)
            errors.extend(jacobian_errors)
        return delta, xi_jacobian, errors, iterations, evaluations

    def solve_supply(
            self, initial_tilde_costs: Array, xi_jacobian: Array, beta_jacobian: Array,
            nonlinear_parameters: NonlinearParameters, costs_type: str, costs_bounds: Bounds,
            compute_gradient: bool) -> Tuple[Array, Array, Array, List[Error]]:
        """Compute transformed marginal costs for this market. Then, if compute_gradient is True, compute the Jacobian
        of omega (equivalently, of transformed marginal costs) with respect to theta. If necessary, replace null
        elements in transformed marginal costs with their last values before computing their Jacobian.
        """
        errors: List[Error] = []

        # configure NumPy to identify floating point errors
        with np.errstate(divide='call', over='call', under='ignore', invalid='call'):
            np.seterrcall(lambda *_: errors.append(exceptions.CostsFloatingPointError()))

            # compute marginal costs
            eta, eta_errors = self.compute_eta()
            errors.extend(eta_errors)
            costs = self.products.prices - eta

            # clip marginal costs that are outside of acceptable bounds
            clipped_costs_indices = (costs < costs_bounds[0]) | (costs > costs_bounds[1])
            costs = np.clip(costs, *costs_bounds)

            # take the log of marginal costs under a log-linear specification
            if costs_type == 'linear':
                tilde_costs = costs
            else:
                assert costs_type == 'log'
                if np.any(costs <= 0):
                    errors.append(exceptions.NonpositiveCostsError())
                with np.errstate(all='ignore'):
                    tilde_costs = np.log(costs)

        # if the gradient is to be computed, replace invalid transformed marginal costs with their last computed
        #   values before computing their Jacobian, which is zero for clipped marginal costs
        omega_jacobian = np.full((self.J, nonlinear_parameters.P), np.nan, options.dtype)
        if compute_gradient:
            valid_tilde_costs = tilde_costs.copy()
            bad_costs_indices = ~np.isfinite(tilde_costs)
            valid_tilde_costs[bad_costs_indices] = initial_tilde_costs[bad_costs_indices]
            omega_jacobian, jacobian_errors = self.compute_omega_by_theta_jacobian(
                valid_tilde_costs, xi_jacobian, beta_jacobian, nonlinear_parameters, costs_type
            )
            errors.extend(jacobian_errors)
            omega_jacobian[clipped_costs_indices.flat] = 0
        return tilde_costs, omega_jacobian, clipped_costs_indices, errors
