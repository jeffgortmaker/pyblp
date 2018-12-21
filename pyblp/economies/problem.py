"""Economy-level BLP problem functionality."""

import abc
import collections
import functools
import time
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .economy import Economy
from .. import exceptions, options
from ..configurations.formulation import Formulation
from ..configurations.integration import Integration
from ..configurations.iteration import Iteration
from ..configurations.optimization import Optimization
from ..markets.problem_market import ProblemMarket
from ..parameters import Parameters
from ..primitives import Agents, Products
from ..results.problem_results import ProblemResults
from ..utilities.basics import (
    Array, Bounds, Error, Groups, RecArray, TableFormatter, format_number, format_seconds, generate_items, output,
    update_matrices
)
from ..utilities.statistics import IV, compute_2sls_weights, compute_gmm_moments_mean, compute_gmm_moments_jacobian_mean


class ProblemEconomy(Economy):
    """An abstract BLP problem."""

    @abc.abstractmethod
    def __init__(
            self, product_formulations: Sequence[Optional[Formulation]], agent_formulation: Optional[Formulation],
            products: RecArray, agents: RecArray) -> None:
        """Initialize the underlying economy with product and agent data."""
        super().__init__(product_formulations, agent_formulation, products, agents)

    def solve(
            self, sigma: Optional[Any] = None, pi: Optional[Any] = None, rho: Optional[Any] = None,
            beta: Optional[Any] = None, gamma: Optional[Any] = None, sigma_bounds: Optional[Tuple[Any, Any]] = None,
            pi_bounds: Optional[Tuple[Any, Any]] = None, rho_bounds: Optional[Tuple[Any, Any]] = None,
            beta_bounds: Optional[Tuple[Any, Any]] = None, gamma_bounds: Optional[Tuple[Any, Any]] = None,
            delta: Optional[Any] = None, W: Optional[Any] = None, method: str = '2s',
            optimization: Optional[Optimization] = None, check_optimality: str = 'both', error_behavior: str = 'revert',
            error_punishment: float = 1, delta_behavior: str = 'first', iteration: Optional[Iteration] = None,
            fp_type: str = 'safe', costs_type: str = 'linear', costs_bounds: Optional[Tuple[Any, Any]] = None,
            center_moments: bool = True, W_type: str = 'robust', se_type: str = 'robust') -> ProblemResults:
        r"""Solve the problem.

        The problem is solved in one or more GMM steps. During each step, any parameters in :math:`\hat{\theta}` are
        optimized to minimize the GMM objective value. If there are no parameters in :math:`\hat{\theta}` (for example,
        in the Logit model there are no nonlinear parameters and all linear parameters can be concentrated out), the
        objective is evaluated once during the step.

        If there are nonlinear parameters, the mean utility, :math:`\delta(\hat{\theta})` is computed market-by-market
        with fixed point iteration. Otherwise, it is computed analytically according to the solution of the Logit model.
        If a supply side is to be estimated, marginal costs, :math:`c(\hat{\theta})`, are also computed
        market-by-market. Linear parameters are then estimated, which are used to recover structural error terms, which
        in turn are used to form the objective value. By default, the objective gradient is computed as well.

        .. note::

           This method supports :func:`parallel` processing. If multiprocessing is used, market-by-market computation of
           :math:`\delta(\hat{\theta})` and, if :math:`X_3` was formulated by ``product_formulations`` in
           :class:`Problem`, of :math:`\tilde{c}(\hat{\theta})`, along with associated Jacobians, will be distributed
           among the processes.

        Parameters
        ----------
        sigma : `array-like, optional`
            Configuration for which elements in the Cholesky decomposition of the covariance matrix that measures
            agents' random taste distribution, :math:`\Sigma`, are fixed at zero and starting values for the other
            elements, which, if not fixed by ``sigma_bounds``, are in the vector of unknown elements, :math:`\theta`.

            Rows and columns correspond to columns in :math:`X_2`, which is formulated according
            ``product_formulations`` in :class:`Problem`. If :math:`X_2` was not formulated, this should not be
            specified, since the Logit model will be estimated.

            Values below the diagonal are ignored. Zeros are assumed to be zero throughout estimation and nonzeros are,
            if not fixed by ``sigma_bounds``, starting values for unknown elements in :math:`\theta`.

        pi : `array-like, optional`
            Configuration for which elements in the matrix of parameters that measures how agent tastes vary with
            demographics, :math:`\Pi`, are fixed at zero and starting values for the other elements, which, if not fixed
            by ``pi_bounds``, are in the vector of unknown elements, :math:`\theta`.

            Rows correspond to the same product characteristics as in ``sigma``. Columns correspond to columns in
            :math:`d`, which is formulated according to ``agent_formulation`` in :class:`Problem`. If :math:`d` was not
            formulated, this should not be specified.

            Zeros are assumed to be zero throughout estimation and nonzeros are, if not fixed by ``pi_bounds``, starting
            values for unknown elements in :math:`\theta`.

        rho : `array-like, optional`
            Configuration for which elements in the vector of parameters that measure within nesting group correlation,
            :math:`\rho`, are fixed at zero and starting values for the other elements, which, if not fixed by
            ``rho_bounds``, are in the vector of unknown elements, :math:`\theta`.

            If there is only one element, it corresponds to all groups defined by the ``nesting_ids`` field of
            ``product_data`` in :class:`Problem`. If there is more than one element, there must be as many elements as
            :math:`H`, the number of distinct nesting groups, and elements correspond to group IDs in the sorted order
            given by :attr:`Problem.unique_nesting_ids`. If nesting IDs were not specified, this should not be specified
            either.

            Zeros are assumed to be zero throughout estimation and nonzeros are, if not fixed by ``rho_bounds``,
            starting values for unknown elements in :math:`\theta`.

        beta: `array-like, optional`
            Configuration for which elements in the vector of demand-side linear parameters, :math:`\beta`, are
            concentrated out of the problem. Usually, this is left unspecified, unless there is a supply side, in which
            case parameters on endogenous product characteristics cannot be concentrated out of the problem. Values
            specify which elements are fixed at zero and starting values for the other elements, which, if not fixed by
            ``beta_bounds``, are in the vector of unknown elements, :math:`\theta`.

            Elements correspond to columns in :math:`X_1`, which is formulated according to ``product_formulations`` in
            :class:`Problem`.

            Both ``None`` and ``numpy.nan`` indicate that the parameter should be concentrated out of the problem. That
            is, it will be estimated, but does not have to be included in :math:`\theta`. Zeros are assumed to be zero
            throughout estimation and nonzeros are, if not fixed by ``beta_bounds``, starting values for unknown
            elements in :math:`\theta`.

        gamma: `array-like, optional`
            Configuration for which elements in the vector of supply-side linear parameters, :math:`\gamma`, are
            concentrated out of the problem. Usually, this is left unspecified. Values specify which elements are fixed
            at zero and starting values for the other elements, which, if not fixed by ``gamma_bounds``, are in the
            vector of unknown elements, :math:`\theta`.

            Elements correspond to columns in :math:`X_3`, which is formulated according to ``product_formulations`` in
            :class:`Problem`. If :math:`X_3` was not formulated, this should not be specified.

            Both ``None`` and ``numpy.nan`` indicate that the parameter should be concentrated out of the problem. That
            is, it will be estimated, but does not have to be included in :math:`\theta`. Zeros are assumed to be zero
            throughout estimation and nonzeros are, if not fixed by ``gamma_bounds``, starting values for unknown
            elements in :math:`\theta`.

        sigma_bounds : `tuple, optional`
            Configuration for :math:`\Sigma` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as ``sigma``. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in ``sigma``. If ``optimization`` does not support bounds, these will be ignored.

            By default, if bounds are supported, the diagonal of ``sigma`` is bounded from below by zero. Conditional on
            :math:`X_2`, :math:`\mu`, and an initial estimate of :math:`\mu`, default bounds for off-diagonal parameters
            are chosen to reduce the need for overflow safety precautions.

            Values below the diagonal are ignored. Lower and upper bounds corresponding to zeros in ``sigma`` are set to
            zero. Setting a lower bound equal to an upper bound fixes the corresponding element, removing it from
            :math:`\theta`. Both ``None`` and ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to
            ``numpy.inf`` in ``ub``.

        pi_bounds : `tuple, optional`
            Configuration for :math:`\Pi` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as ``pi``. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in ``pi``. If ``optimization`` does not support bounds, these will be ignored.

            By default, if bounds are supported, conditional on :math:`X_2`, :math:`d`, and an initial estimate of
            :math:`\mu`, default bounds are chosen to reduce the need for overflow safety precautions.

            Lower and upper bounds corresponding to zeros in ``pi`` are set to zero. Setting a lower bound equal to an
            upper bound fixes the corresponding element, removing it from :math:`\theta`. Both ``None`` and
            ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        rho_bounds : `tuple, optional`
            Configuration for :math:`\rho` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as ``rho``. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in ``rho``. If ``optimization`` does not support bounds, these will be ignored.

            By default, if bounds are supported, all elements are bounded from below by ``0``, which corresponds to the
            simple Logit model. Conditional on an initial estimate of :math:`\mu`, upper bounds are chosen to reduce the
            need for overflow safety precautions, and are less than ``1`` because larger values are inconsistent with
            utility maximization.

            Lower and upper bounds corresponding to zeros in ``rho`` are set to zero. Setting a lower bound equal to an
            upper bound fixes the corresponding element, removing it from :math:`\theta`. Both ``None`` and
            ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        beta_bounds : `tuple, optional`
            Configuration for :math:`\beta` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as ``beta``. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in ``beta``. If ``optimization`` does not support bounds, these will be ignored.

            Usually, this is left unspecified, unless there is a supply side, in which case parameters on endogenous
            product characteristics cannot be concentrated out of the problem. It is generally a good idea to constrain
            such parameters to be nonzero so that the intra-firm Jacobian of shares with respect to prices does not
            become singular.

            By default, all non-concentrated out parameters are unbounded. Bounds should only be specified for
            parameters that are included in :math:`\theta`; that is, those with initial values specified in ``beta``.

            Lower and upper bounds corresponding to zeros in ``beta`` are set to zero. Setting a lower bound equal to an
            upper bound fixes the corresponding element, removing it from :math:`\theta`. Both ``None`` and
            ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        gamma_bounds : `tuple, optional`
            Configuration for :math:`\gamma` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as ``gamma``. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in ``gamma``. If ``optimization`` does not support bounds, these will be ignored.

            By default, all non-concentrated out parameters are unbounded. Bounds should only be specified for
            parameters that are included in :math:`\theta`; that is, those with initial values specified in ``gamma``.

            Lower and upper bounds corresponding to zeros in ``gamma`` are set to zero. Setting a lower bound equal to
            an upper bound fixes the corresponding element, removing it from :math:`\theta`. Both ``None`` and
            ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        delta : `array-like, optional`
            Initial values for the mean utility, :math:`\delta`. If there are any nonlinear parameters, these are the
            values at which the fixed point iteration routine will start during the first objective evaluation. By
            default, the solution to the sample Logit model is used:

            .. math:: \delta_{jt} = \log s_{jt} - \log s_{0t}.

            If there is nesting, the solution to the nested Logit model under the initial ``rho`` is used instead:

            .. math:: \delta_{jt} = \log s_{jt} - \log s_{0t} - \rho_{h(j)}\log\frac{s_{jt}}{s_{h(j)t}}

            where

            .. math:: s_{h(j)t} = \sum_{k\in\mathscr{J}_{h(j)t}} s_{kt}.

        W : `array-like, optional`
            Starting values for the weighting matrix, :math:`W`. By default, the 2SLS weighting matrix,
            :math:`(Z'Z)^{-1}`, is used.
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
            ``Optimization('slsqp', {'ftol': 1e-12})`` is used. Routines that do not support bounds will ignore
            ``sigma_bounds`` and ``pi_bounds``. Choosing a routine that does not use analytic gradients will slow down
            estimation.
        check_optimality : `str, optional`
            How to check for optimality after the optimization routine finishes. The following configurations are
            supported:

                - ``'gradient'`` - Analytically compute the gradient after optimization finishes, but do not compute the
                  Hessian. Since Jacobians needed to compute standard errors will already be computed, gradient
                  computation will not take a long time. This option may be useful it Hessian computation takes a long
                  time when, for example, there are a large number of parameters.

                - ``'both'`` (default) - Also compute the Hessian with central finite differences after optimization
                  finishes. Specifically, analytically compute the gradient :math:`2P` times, perturbing each of the
                  :math:`P` parameters by :math:`\pm\epsilon / 2` where :math:`\epsilon` is the square root of the
                  machine precision.

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
                  large ``error_punishment`` can be helpful for routines that do not use analytic gradients.

                - ``'raise'`` - Raise an exception.

        error_punishment : `float, optional`
            How to scale the GMM objective value after an error. By default, the objective value is not scaled.
        delta_behavior : `str, optional`
            Configuration for the values at which the fixed point computation of :math:`\delta(\hat{\theta})` in each
            market will start. This configuration is only relevant if there are unfixed nonlinear parameters over which
            to optimize. The following behaviors are supported:

                - ``'first'`` (default) - Start at the values configured by ``delta`` during the first GMM step, and at
                  the values computed by the last GMM step for each subsequent step.

                - ``'last'`` - Start at the values of :math:`\delta(\hat{\theta})` computed during the last objective
                  evaluation, or, if this is the first evaluation, at the values configured by ``delta``. This behavior
                  tends to speed up computation but may introduce some instability into estimation.

        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem used to compute
            :math:`\delta(\hat{\theta})` in each market. This configuration is only relevant if there are nonlinear
            parameters, since :math:`\delta` can be estimated analytically in the Logit model. By default,
            ``Iteration('squarem', {'tol': 1e-14})`` is used.
        fp_type : `str, optional`
            Configuration for the type of contraction mapping used to compute :math:`\delta(\hat{\theta})`. The
            following types of contraction mappings are supported:

                - ``'safe'`` (default) - The standard linear contraction mapping,

                  .. math:: \delta_{jt} \leftarrow \delta_{jt} + \log s_{jt} - \log s_{jt}(\delta, \hat{\theta}),

                  with safeguards against numerical overflow. Specifically, during choice probability computation, the
                  maximum utility of each agent is subtracted away before utilities are exponentiated, and the logit
                  expression is re-scaled accordingly.

                - ``'linear'`` - Standard linear contraction mapping, but without safeguards against numerical overflow.
                  This option may be preferable to ``'safe'`` if utilities are reasonably small.

                - ``'nonlinear'`` - Exponentiated version,

                  .. math:: \exp(\delta_{jt}) \leftarrow \exp(\delta_{jt})s_{jt} / s_{jt}(\delta, \hat{\theta}),

                  which can be faster because fewer logarithms need to be calculated. Additionally, when there are no
                  nesting parameters, as in :ref:`references:Brunner, Heiss, Romahn, and Weiser (2017)`,
                  :math:`\exp(\delta)` is cancelled out of the numerator in the expression for
                  :math:`s(\delta, \hat{\theta})`, which slightly reduces the computational burden. This formulation can
                  also help mitigate problems stemming from any negative integration weights; however, it is generally
                  less stable than the linear versions.

            This option is only relevant if there are nonlinear parameters, since :math:`\delta` can be estimated
            analytically in the Logit model.

            Also note that when there are nesting parameters, the contraction is dampened by :math:`1 - \rho` as in
            :ref:`references:Grigolon and Verboven (2014)`. Although necessary, this dampening implies a slower rate of
            convergence, especially for large values of :math:`\rho`.

        costs_type : `str, optional`
            Marginal cost specification. The following specifications are supported:

                - ``'linear'`` (default) - Linear specification: :math:`\tilde{c} = c`.

                - ``'log'`` - Log-linear specification: :math:`\tilde{c} = \log c`.

            This specification is only relevant if :math:`X_3` was formulated by ``product_formulations`` in
            :class:`Problem`.

        costs_bounds : `tuple, optional`
            Configuration for :math:`c` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are floats.
            This is only relevant if :math:`X_3` was formulated by ``product_formulations`` in :class:`Problem`. By
            default, marginal costs are unbounded.

            When ``costs_type`` is ``'log'``, nonpositive :math:`c(\hat{\theta})` values can create problems when
            computing :math:`\tilde{c}(\hat{\theta}) = \log c(\hat{\theta})`. One solution is to set ``lb`` to a small
            number. Rows in Jacobians associated with clipped marginal costs will be zero.

            Both ``None`` and ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        center_moments : `bool, optional`
            Whether to center the sample moments before using them to update weighting matrices. By default, sample
            moments are centered. This has no effect if ``W_type`` is ``'unadjusted'``.
        W_type : `str, optional`
            How to update the weighting matrix. This has no effect if ``method`` is ``'1s'``. Often, ``se_type`` should
            be the same. The following types are supported:

                - ``'robust'`` (default) - Heteroscedasticity robust weighting matrices.

                - ``'unadjusted'`` - Homoskedastic weighting matrices. Errors are always centered when computing this
                  type of weighting matrix, so ``center_moments`` has no effect.

                - ``'clustered'`` - Clustered weighting matrices, which account for arbitrary within-group correlation.
                  Clusters must be defined by the ``clustering_ids`` field of ``product_data`` in :class:`Problem`.

        se_type : `str, optional`
            How to compute standard errors. Often, ``W_type`` should be the same. The following types are supported:

                - ``'robust'`` (default) - Heteroscedasticity robust standard errors.

                - ``'unadjusted'`` - Homoskedastic standard errors. Unadjusted standard errors are computed under the
                  assumption that weighting matrices are optimal.

                - ``'clustered'`` - Clustered standard errors, which account for arbitrary within-group correlation.
                  Clusters must be defined by the ``clustering_ids`` field of ``product_data`` in :class:`Problem`.

        Returns
        -------
        `ProblemResults`
            :class:`ProblemResults` of the solved problem.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """

        # keep track of how long it takes to solve the problem
        output("Solving the problem ...")
        step_start_time = time.time()

        # validate the estimation method
        if method not in {'1s', '2s'}:
            raise TypeError("method must be '1s' or '2s'.")

        # configure or validate configurations
        if optimization is None:
            optimization = Optimization('slsqp', {'ftol': 1e-12})
        if iteration is None:
            iteration = Iteration('squarem', {'tol': 1e-14})
        if not isinstance(optimization, Optimization):
            raise TypeError("optimization must be None or an Optimization instance.")
        if not isinstance(iteration, Iteration):
            raise TypeError("iteration must be None or an Iteration instance.")

        # validate behaviors and types
        if check_optimality not in {'gradient', 'both'}:
            raise ValueError("check_optimality must be 'gradient' or 'both'.")
        if error_behavior not in {'revert', 'punish', 'raise'}:
            raise ValueError("error_behavior must be 'revert', 'punish', or 'raise'.")
        if delta_behavior not in {'last', 'first'}:
            raise ValueError("delta_behavior must be 'last' or 'first'.")
        if fp_type not in {'safe', 'linear', 'nonlinear'}:
            raise ValueError("fp_type must be 'safe', 'linear', or 'nonlinear'.")
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
                raise ValueError(f"The lower bound in costs_bounds must be None or a float.")
            if costs_bounds[1].size != 1:
                raise ValueError(f"The upper bound in costs_bounds must be None or a float.")
            if costs_bounds[0] > costs_bounds[1]:
                raise ValueError("The lower bound in costs_bounds cannot be larger than the upper bound.")

        # validate parameters before compressing unfixed parameters into theta
        parameters = Parameters(
            self, sigma, pi, rho, beta, gamma, sigma_bounds, pi_bounds, rho_bounds, beta_bounds, gamma_bounds,
            bounded=optimization._supports_bounds, allow_linear_nans=True
        )
        theta = parameters.compress()
        theta_bounds = parameters.compress_bounds()

        # output information about initial parameters and their bounds
        if parameters.fixed or parameters.unfixed:
            output("")
            output(parameters.format("Initial Values"))
            if optimization._supports_bounds:
                output("")
                output(parameters.format_lower_bounds("Lower Bounds"))
                output("")
                output(parameters.format_upper_bounds("Upper Bounds"))

        # compute or load the weighting matrix
        if W is None:
            Z_list = [self.products.ZD]
            if self.MS > 0:
                Z_list.append(self.products.ZS)
            W, W_errors = compute_2sls_weights(Z_list)
            self._handle_errors(error_behavior, W_errors)
        else:
            W = np.asarray(W, options.dtype)
            if W.shape != (self.MD + self.MS, self.MD + self.MS):
                raise ValueError(f"WD must have {self.MD + self.MS} rows and columns.")

        # compute or load initial delta values
        if delta is None:
            delta = self._compute_logit_delta(parameters.rho)
        else:
            delta = np.c_[np.asarray(delta, options.dtype)]
            if delta.shape != (self.N, 1):
                raise ValueError(f"delta must be a vector with {self.N} elements.")

        # initialize marginal costs as prices, which will only be used if there are computation errors during the first
        #   objective evaluation
        tilde_costs = np.full((self.N, 0), np.nan, options.dtype)
        if self.K3 > 0:
            if costs_type == 'linear':
                tilde_costs = self.products.prices
            else:
                assert costs_type == 'log'
                tilde_costs = np.log(self.products.prices)

        # initialize Jacobians of xi and omega with respect to theta as all zeros, which will only be used if there are
        #   computation errors during the first objective evaluation
        xi_jacobian = np.zeros((self.N, parameters.P), options.dtype)
        omega_jacobian = np.full_like(xi_jacobian, 0 if self.K3 > 0 else np.nan, options.dtype)

        # initialize the objective as a large number and its gradient and hessian as all zeros, which will only be used
        #   if there are computation errors during the first objective evaluation
        objective = np.array(1e10, options.dtype)
        gradient = np.zeros((parameters.P, 1), options.dtype)
        hessian = np.zeros((parameters.P, parameters.P), options.dtype)

        # iterate over each GMM step
        step = 1
        last_results = None
        while True:
            # collect inputs into linear parameter estimation
            X_list = [self.products.X1[:, parameters.eliminated_beta_index.flat]]
            Z_list = [self.products.ZD]
            if self.K3 > 0:
                X_list.append(self.products.X3[:, parameters.eliminated_gamma_index.flat])
                Z_list.append(self.products.ZS)

            # initialize an IV model for linear parameter estimation
            iv = IV(X_list, Z_list, W)
            self._handle_errors(error_behavior, iv.errors)

            # wrap computation of progress information with step-specific information
            compute_step_progress = functools.partial(
                self._compute_progress, parameters, iv, W, error_behavior, error_punishment, delta_behavior, iteration,
                fp_type, costs_type, costs_bounds
            )

            # initialize optimization progress
            converged_mappings: List[Dict[Hashable, bool]] = []
            iteration_mappings: List[Dict[Hashable, int]] = []
            evaluation_mappings: List[Dict[Hashable, int]] = []
            smallest_objective = np.inf
            progress = InitialProgress(
                self, parameters, W, theta, objective, gradient, hessian, delta, delta, tilde_costs, xi_jacobian,
                omega_jacobian
            )

            # define the objective function
            def wrapper(
                    new_theta: Array, current_iterations: int, current_evaluations: int) -> (
                    Union[float, Tuple[float, Array]]):
                """Compute and output progress associated with a single objective evaluation."""
                nonlocal converged_mappings, iteration_mappings, evaluation_mappings, smallest_objective, progress
                assert optimization is not None and costs_bounds is not None
                progress = progress = compute_step_progress(
                    new_theta, progress, optimization._compute_gradient, compute_hessian=False
                )
                converged_mappings.append(progress.converged_mapping)
                iteration_mappings.append(progress.iteration_mapping)
                evaluation_mappings.append(progress.evaluation_mapping)
                formatted_progress = progress.format(
                    optimization, costs_bounds, step, current_iterations, current_evaluations, smallest_objective
                )
                if formatted_progress:
                    output(formatted_progress)
                smallest_objective = min(smallest_objective, progress.objective)
                return (progress.objective, progress.gradient) if optimization._compute_gradient else progress.objective

            # optimize theta
            converged = True
            iterations = evaluations = 0
            optimization_start_time = optimization_end_time = time.time()
            if parameters.P > 0:
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
            compute_gradient = parameters.P > 0
            compute_hessian = compute_gradient and check_optimality == 'both'
            final_progress = compute_step_progress(theta, progress, compute_gradient, compute_hessian)
            results = ProblemResults(
                final_progress, last_results, step_start_time, optimization_start_time, optimization_end_time,
                iterations, evaluations + 1, converged_mappings, iteration_mappings, evaluation_mappings, converged,
                costs_type, costs_bounds, center_moments, W_type, se_type
            )
            self._handle_errors(error_behavior, results._errors)
            output(f"Computed results after {format_seconds(results.total_time - results.optimization_time)}.")

            # store the last results and return results from the final step
            last_results = results
            if method != '2s' or step == 2:
                output("")
                output(results)
                return results

            # update vectors and matrices
            delta = results.delta
            tilde_costs = results.tilde_costs
            xi_jacobian = results.xi_by_theta_jacobian
            omega_jacobian = results.omega_by_theta_jacobian
            W = results.updated_W
            step += 1
            step_start_time = time.time()

    def _compute_progress(
            self, parameters: Parameters, iv: IV, W: Array, error_behavior: str, error_punishment: float,
            delta_behavior: str, iteration: Iteration, fp_type: str, costs_type: str, costs_bounds: Bounds,
            theta: Array, progress: 'InitialProgress', compute_gradient: bool, compute_hessian: bool) -> 'Progress':
        """Compute demand- and supply-side contributions before recovering the linear parameters and structural error
        terms. Then, form the GMM objective value and its gradient. Finally, handle any errors that were encountered
        before structuring relevant progress information.
        """
        errors: List[Error] = []

        # expand theta
        sigma, pi, rho, beta, gamma = parameters.expand(theta)

        # compute demand-side contributions
        delta, xi_jacobian, converged, iterations, evaluations, demand_errors = self._compute_demand_contributions(
            parameters, iteration, fp_type, sigma, pi, rho, progress, compute_gradient
        )
        errors.extend(demand_errors)

        # compute supply-side contributions
        if self.K3 == 0:
            tilde_costs = np.full((self.N, 0), np.nan, options.dtype)
            omega_jacobian = np.full((self.N, parameters.P), np.nan, options.dtype)
            clipped_costs = np.zeros((self.N, 1), np.bool)
        else:
            tilde_costs, omega_jacobian, clipped_costs, supply_errors = self._compute_supply_contributions(
                parameters, costs_type, costs_bounds, sigma, pi, rho, beta, delta, xi_jacobian, progress,
                compute_gradient
            )
            errors.extend(supply_errors)

        # subtract contributions of linear parameters in theta
        iv_delta = delta.copy()
        iv_tilde_costs = tilde_costs.copy()
        if not parameters.eliminated_beta_index.all():
            theta_beta = np.c_[beta[~parameters.eliminated_beta_index]]
            iv_delta -= self._compute_true_X1(index=~parameters.eliminated_beta_index.flatten()) @ theta_beta
        if not parameters.eliminated_gamma_index.all():
            theta_gamma = np.c_[gamma[~parameters.eliminated_gamma_index]]
            iv_delta -= self._compute_true_X3(index=~parameters.eliminated_gamma_index.flatten()) @ theta_gamma

        # absorb any fixed effects
        if self._absorb_demand_ids is not None:
            iv_delta, demand_absorption_errors = self._absorb_demand_ids(iv_delta)
            errors.extend(demand_absorption_errors)
        if self._absorb_supply_ids is not None:
            iv_tilde_costs, supply_absorption_errors = self._absorb_supply_ids(iv_tilde_costs)
            errors.extend(supply_absorption_errors)

        # collect inputs into GMM estimation
        X_list = [self.products.X1[:, parameters.eliminated_beta_index.flat]]
        Z_list = [self.products.ZD]
        y_list = [iv_delta]
        jacobian_list = [xi_jacobian]
        if self.K3 > 0:
            X_list.append(self.products.X3[:, parameters.eliminated_gamma_index.flat])
            Z_list.append(self.products.ZS)
            y_list.append(iv_tilde_costs)
            jacobian_list.append(omega_jacobian)

        # recover the linear parameters and structural error terms
        parameters_list, u_list = iv.estimate(X_list, Z_list, W, y_list)
        beta[parameters.eliminated_beta_index] = parameters_list[0].flat
        xi = u_list[0]
        if self.K3 == 0:
            omega = np.full((self.N, 0), np.nan, options.dtype)
        else:
            gamma[parameters.eliminated_gamma_index] = parameters_list[1].flat
            omega = u_list[1]

        # compute the objective value and replace it with its last value if computation failed
        with np.errstate(all='ignore'):
            g_bar = compute_gmm_moments_mean(u_list, Z_list)
            objective = self.N**2 * g_bar.T @ W @ g_bar
        if not np.isfinite(np.squeeze(objective)):
            objective = progress.objective
            errors.append(exceptions.ObjectiveReversionError())

        # compute the gradient and replace any invalid elements with their last values
        gradient = np.full_like(theta, np.nan, options.dtype)
        if compute_gradient:
            with np.errstate(all='ignore'):
                G_bar = compute_gmm_moments_jacobian_mean(jacobian_list, Z_list)
                gradient = self.N**2 * 2 * (G_bar.T @ W @ g_bar)
            bad_gradient_index = ~np.isfinite(gradient)
            if np.any(bad_gradient_index):
                gradient[bad_gradient_index] = progress.gradient[bad_gradient_index]
                errors.append(exceptions.GradientReversionError(bad_gradient_index))

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
            next_delta = delta
        else:
            assert delta_behavior == 'first'
            next_delta = progress.next_delta

        # compute the hessian with central finite differences
        hessian = np.full((parameters.P, parameters.P), np.nan, options.dtype)
        if compute_hessian:
            compute_progress = lambda x: self._compute_progress(
                parameters, iv, W, error_behavior, error_punishment, delta_behavior, iteration, fp_type, costs_type,
                costs_bounds, x, progress, compute_gradient=True, compute_hessian=False
            )
            change = np.sqrt(np.finfo(np.float64).eps)
            for p in range(parameters.P):
                theta1 = theta.copy()
                theta2 = theta.copy()
                theta1[p] += change / 2
                theta2[p] -= change / 2
                hessian[:, [p]] = (compute_progress(theta1).gradient - compute_progress(theta2).gradient) / change

            # enforce shape and symmetry
            hessian = np.c_[hessian + hessian.T] / 2

        # structure progress
        return Progress(
            self, parameters, W, theta, objective, gradient, hessian, next_delta, delta, tilde_costs, xi_jacobian,
            omega_jacobian, xi, omega, beta, gamma, converged, iterations, evaluations, clipped_costs, errors
        )

    def _compute_demand_contributions(
            self, parameters: Parameters, iteration: Iteration, fp_type: str, sigma: Array, pi: Array, rho: Array,
            progress: 'InitialProgress', compute_jacobian: bool) -> (
            Tuple[Array, Array, Dict[Hashable, bool], Dict[Hashable, int], Dict[Hashable, int], List[Error]]):
        """Compute delta and the Jacobian of xi (equivalently, of delta) with respect to theta market-by-market. Revert
        any problematic elements to their last values.
        """
        errors: List[Error] = []

        # initialize delta and its Jacobian along with fixed point information so that they can be filled
        converged: Dict[Hashable, bool] = {}
        iterations: Dict[Hashable, int] = {}
        evaluations: Dict[Hashable, int] = {}
        delta = np.zeros((self.N, 1), options.dtype)
        xi_jacobian = np.full((self.N, parameters.P), np.nan, options.dtype)

        # when possible and when a gradient isn't needed, compute delta with a closed-form solution
        if self.K2 == 0 and (parameters.P == 0 or not compute_jacobian):
            delta = self._compute_logit_delta(rho)
        else:
            # define a factory for solving the demand side of problem markets
            def market_factory(s: Hashable) -> Tuple[ProblemMarket, Array, Parameters, Iteration, str, bool]:
                """Build a market along with arguments used to compute delta and its Jacobian."""
                market_s = ProblemMarket(self, s, sigma, pi, rho)
                initial_delta_s = progress.next_delta[self._product_market_indices[s]]
                return market_s, initial_delta_s, parameters, iteration, fp_type, compute_jacobian

            # compute delta and its Jacobian market-by-market
            generator = generate_items(self.unique_market_ids, market_factory, ProblemMarket.solve_demand)
            for t, (delta_t, xi_jacobian_t, errors_t, converged[t], iterations[t], evaluations[t]) in generator:
                delta[self._product_market_indices[t]] = delta_t
                xi_jacobian[self._product_market_indices[t], :xi_jacobian_t.shape[1]] = xi_jacobian_t
                errors.extend(errors_t)

        # replace invalid elements in delta with their last values
        bad_delta_index = ~np.isfinite(delta)
        if np.any(bad_delta_index):
            delta[bad_delta_index] = progress.delta[bad_delta_index]
            errors.append(exceptions.DeltaReversionError(bad_delta_index))

        # replace invalid elements in its Jacobian with their last values
        if compute_jacobian:
            bad_jacobian_index = ~np.isfinite(xi_jacobian)
            if np.any(bad_jacobian_index):
                xi_jacobian[bad_jacobian_index] = progress.xi_jacobian[bad_jacobian_index]
                errors.append(exceptions.XiByThetaJacobianReversionError(bad_jacobian_index))
        return delta, xi_jacobian, converged, iterations, evaluations, errors

    def _compute_supply_contributions(
            self, parameters: Parameters, costs_type: str, costs_bounds: Bounds, sigma: Array, pi: Array, rho: Array,
            beta: Array, delta: Array, xi_jacobian: Array, progress: 'InitialProgress', compute_jacobian: bool) -> (
            Tuple[Array, Array, Array, List[Error]]):
        """Compute transformed marginal costs and the Jacobian of omega (equivalently, of transformed marginal costs)
        with respect to theta market-by-market. Revert any problematic elements to their last values.
        """
        errors: List[Error] = []

        # initialize transformed marginal costs, their Jacobian, and indices of clipped costs so that they can be filled
        tilde_costs = np.full((self.N, 1), np.nan, options.dtype)
        omega_jacobian = np.full((self.N, parameters.P), np.nan, options.dtype)
        clipped_costs = np.zeros((self.N, 1), np.bool)

        # define a factory for solving the supply side of problem markets
        def market_factory(
                s: Hashable) -> Tuple[ProblemMarket, Array, Array, Parameters, str, Bounds, bool]:
            """Build a market along with arguments used to compute transformed marginal costs and their Jacobian."""
            market_s = ProblemMarket(self, s, sigma, pi, rho, beta, delta)
            last_tilde_costs_s = progress.tilde_costs[self._product_market_indices[s]]
            xi_jacobian_s = xi_jacobian[self._product_market_indices[s]]
            return market_s, last_tilde_costs_s, xi_jacobian_s, parameters, costs_type, costs_bounds, compute_jacobian

        # compute transformed marginal costs and their Jacobian market-by-market
        generator = generate_items(self.unique_market_ids, market_factory, ProblemMarket.solve_supply)
        for t, (tilde_costs_t, omega_jacobian_t, clipped_costs_t, errors_t) in generator:
            tilde_costs[self._product_market_indices[t]] = tilde_costs_t
            omega_jacobian[self._product_market_indices[t], :omega_jacobian_t.shape[1]] = omega_jacobian_t
            clipped_costs[self._product_market_indices[t]] = clipped_costs_t
            errors.extend(errors_t)

        # replace invalid transformed marginal costs with their last values
        bad_tilde_costs_index = ~np.isfinite(tilde_costs)
        if np.any(bad_tilde_costs_index):
            tilde_costs[bad_tilde_costs_index] = progress.tilde_costs[bad_tilde_costs_index]
            errors.append(exceptions.CostsReversionError(bad_tilde_costs_index))

        # replace invalid elements in their Jacobian with their last values
        if compute_jacobian:
            bad_jacobian_index = ~np.isfinite(omega_jacobian)
            if np.any(bad_jacobian_index):
                omega_jacobian[bad_jacobian_index] = progress.omega_jacobian[bad_jacobian_index]
                errors.append(exceptions.OmegaByThetaJacobianReversionError(bad_jacobian_index))
        return tilde_costs, omega_jacobian, clipped_costs, errors

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


class Problem(ProblemEconomy):
    r"""A BLP problem.

    This class is initialized with relevant data and solved with :meth:`Problem.solve`.

    In both ``product_data`` and ``agent_data``, fields with multiple columns can be either matrices or can be broken up
    into multiple one-dimensional fields with column index suffixes that start at zero. For example, if there are three
    columns of excluded demand-side instruments, the ``demand_instruments`` field in ``product_data``, which in this
    case should be a matrix with three columns, can be replaced by three one-dimensional fields:
    ``demand_instruments0``, ``demand_instruments1``, and ``demand_instruments2``.

    Parameters
    ----------
    product_formulations : `Formulation or tuple of Formulation`
        :class:`Formulation` configuration or tuple of up to three :class:`Formulation` configurations for the matrix
        of linear product characteristics, :math:`X_1`, for the matrix of nonlinear product characteristics,
        :math:`X_2`, and for the matrix of cost characteristics, :math:`X_3`, respectively. If the formulation for
        :math:`X_3` is not specified or is ``None``, a supply side will not be estimated. Similarly, if the formulation
        for :math:`X_2` is not specified or is ``None``, the Logit model will be estimated.

        Variable names should correspond to fields in ``product_data``. The ``shares`` variable should not be included
        in any of the formulations and ``prices`` should be included in the formulation for :math:`X_1` or :math:`X_2`
        (or both). The ``absorb`` argument of :class:`Formulation` can be used to absorb fixed effects into :math:`X_1`
        and :math:`X_3`, but not :math:`X_2`. Generally speaking, all exogenous characteristics in :math:`X_2` should
        also be included in :math:`X_1`. The exception is characteristics that are collinear with fixed effects in
        :math:`X_1`.

        Characteristics in :math:`X_1` that do not involve ``prices`` will be combined with the below specified excluded
        demand-side instruments to create the full set of demand-side instruments, :math:`Z_D`. Any fixed effects
        absorbed into :math:`X_1` will also be absorbed into :math:`Z_D`. Similarly, characteristics in :math:`X_3` will
        be combined with the excluded supply-side instruments to create :math:`Z_S`, and any fixed effects absorbed into
        :math:`X_3` will also be absorbed into :math:`Z_S`.

        .. warning::

           Characteristics that involve prices, :math:`p`, should always be formulated with the ``prices`` variable. If
           another name is used, :class:`Problem` will not understand that the characteristic is endogenous, so it may
           be erroneously included in :math:`Z_D`, and derivatives computed with respect to prices (which are computed
           during supply-side estimation and post-estimation routines) will likely be wrong. For example, to include a
           :math:`p^2` characteristic, include ``I(prices**2)`` in a formula instead of manually including a
           ``prices_squared`` variable in ``product_data`` and a formula.

    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **shares** : (`numeric`) - Market shares, :math:`s`.

            - **prices** : (`numeric`) - Product prices, :math:`p`.

            - **demand_instruments** : (`numeric`) - Excluded demand-side instruments, which together with the
              formulated exogenous linear product characteristics (:math:`X_1` except for characteristics involving
              ``prices``, :math:`X_1^p`), constitute the full set of demand-side instruments, :math:`Z_D`.

        If a formulation for :math:`X_3` is specified in ``product_formulations``, the following fields are also
        required, since they will be used to estimate the supply side of the problem:

            - **firm_ids** : (`object, optional`) - IDs that associate products with firms. Any columns after the first
              can be used to compute post-estimation outputs for firm changes, such as mergers.

            - **supply_instruments** : (`numeric, optional`) - Excluded supply-side instruments, which together with the
              formulated cost characteristics, :math:`X_3`, constitute the full set of supply-side instruments,
              :math:`Z_S`.

        In addition to supply-side estimation, the ``firm_ids`` field is also needed to compute some post-estimation
        outputs. If ``firm_ids`` are specified, custom ownership matrices can be specified as well:

            - **ownership** : (`numeric, optional`) - Custom stacked :math:`J_t \times J_t` ownership matrices,
              :math:`O`, for each market :math:`t`, which can be built with :func:`build_ownership`. By default,
              standard ownership matrices are built only when they are needed. If specified, each stack is associated
              with a ``firm_ids`` column and must have as many columns as there are products in the market with the most
              products.

        To estimate a nested Logit or random coefficients nested Logit (RCNL) model, nesting groups must be specified:

            - **nesting_ids** (`object, optional`) - IDs that associate products with nesting groups. When these IDs are
              specified, ``rho`` in :meth:`Problem.solve`, the vector of parameters that measure within nesting group
              correlation, must be specified as well.

        Finally, clustering groups can be specified to account for arbitrary within-group correlation while computing
        standard errors and weighting matrices:

            - **clustering_ids** (`object, optional`) - Cluster group IDs, which will be used when estimating standard
              errors and updating weighting matrices if ``covariance_type`` in :meth:`Problem.solve` is ``'clustered'``.

        Along with ``market_ids``, ``firm_ids``, ``nesting_ids``, ``clustering_ids``, and ``prices``, the names of any
        additional fields can be used as variables in ``product_formulations``.

    agent_formulation : `Formulation, optional`
        :class:`Formulation` configuration for the matrix of observed agent characteristics called demographics,
        :math:`d`, which will only be included in the model if this formulation is specified. Since demographics are
        only used if there are nonlinear product characteristics, this formulation should only be specified if
        :math:`X_2` is formulated in ``product_formulations``. Variable names should correspond to fields in
        ``agent_data``.
    agent_data : `structured array-like, optional`
        Each row corresponds to an agent. Markets can have differing numbers of agents. Since simulated agents are only
        used if there are nonlinear product characteristics, agent data should only be specified if :math:`X_2` is
        formulated in ``product_formulations``. If agent data are specified, market IDs are required:

            - **market_ids** : (`object`) - IDs that associate agents with markets. The set of distinct IDs should be
              the same as the set of IDs in ``product_data``. If ``integration`` is specified, there must be at least as
              many rows in each market as the number of nodes and weights that are built for each market.

        If ``integration`` is not specified, the following fields are required:

            - **weights** : (`numeric, optional`) - Integration weights, :math:`w`.

            - **nodes** : (`numeric, optional`) - Unobserved agent characteristics called integration nodes,
              :math:`\nu`. If there are more than :math:`K_2` columns, only the first :math:`K_2` will be used.

        Along with ``market_ids``, the names of any additional fields can be used as variables in ``agent_formulation``.

    integration : `Integration, optional`
        :class:`Integration` configuration for how to build nodes and weights for integration over agent utilities,
        which will replace any ``nodes`` and ``weights`` fields in ``agent_data``. This configuration is required if
        ``nodes`` and ``weights`` in ``agent_data`` are not specified. It should not be specified if :math:`X_2` is not
        formulated in ``product_formulations``.

    Attributes
    ----------
    product_formulations : `Formulation or tuple of Formulation`
        :class:`Formulation` configurations for :math:`X_1`, :math:`X_2`, and :math:`X_3`, respectively.
    agent_formulation : `Formulation`
        :class:`Formulation` configuration for :math:`d`.
    products : `Products`
        Product data structured as :class:`Products`, which consists of data taken from ``product_data`` along with
        matrices build according to :attr:`Problem.product_formulations`.
    agents : `Agents`
        Agent data structured as :class:`Agents`, which consists of data taken from ``agent_data`` or built by
        ``integration`` along with any demographics formulated by ``agent_formulation``.
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
        Number of demand-side instruments, :math:`M_D`, which is the number of excluded demand-side instruments plus
        :math:`K_1 - K_1^p`.
    MS : `int`
        Number of supply-side instruments, :math:`M_S`, which is the number of excluded supply-side instruments plus
        :math:`K_3`.
    ED : `int`
        Number of absorbed demand-side fixed effects, :math:`E_D`.
    ES : `int`
        Number of absorbed supply-side fixed effects, :math:`E_S`.
    H : `int`
        Number of nesting groups, :math:`H`.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    def __init__(
            self, product_formulations: Union[Formulation, Sequence[Optional[Formulation]]], product_data: Mapping,
            agent_formulation: Optional[Formulation] = None, agent_data: Optional[Mapping] = None,
            integration: Optional[Integration] = None) -> None:
        """Initialize the underlying economy with product and agent data before absorbing fixed effects."""

        # keep track of long it takes to initialize the problem
        output("Initializing the problem ...")
        start_time = time.time()

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

        # absorb any demand-side fixed effects
        if self._absorb_demand_ids is not None:
            output("Absorbing demand-side fixed effects ...")
            self.products.X1, X1_errors = self._absorb_demand_ids(self.products.X1)
            self.products.ZD, ZD_errors = self._absorb_demand_ids(self.products.ZD)
            if X1_errors or ZD_errors:
                raise exceptions.MultipleErrors(X1_errors + ZD_errors)

        # absorb any supply-side fixed effects
        if self._absorb_supply_ids is not None:
            output("Absorbing supply-side fixed effects ...")
            self.products.X3, X3_errors = self._absorb_supply_ids(self.products.X3)
            self.products.ZS, ZS_errors = self._absorb_supply_ids(self.products.ZS)
            if X3_errors or ZS_errors:
                raise exceptions.MultipleErrors(X3_errors + ZS_errors)

        # output information about the initialized problem
        output(f"Initialized the problem after {format_seconds(time.time() - start_time)}.")
        output("")
        output(self)


class OptimalInstrumentProblem(ProblemEconomy):
    """A BLP problem updated with optimal excluded instruments.

    This class can be used exactly like :class:`Problem`.

    """

    def __init__(self, problem: ProblemEconomy, demand_instruments: Array, supply_instruments: Array) -> None:
        """Initialize the underlying economy with updated product data before absorbing fixed effects."""

        # keep track of long it takes to re-create the problem
        output("Re-creating the problem ...")
        start_time = time.time()

        # supplement the excluded demand-side instruments with exogenous characteristics in X1
        X1 = problem._compute_true_X1()
        ZD = demand_instruments
        for index, formulation in enumerate(problem._X1_formulations):
            if 'prices' not in formulation.names:
                ZD = np.c_[ZD, X1[:, [index]]]

        # supplement the excluded supply-side instruments with X3
        X3 = problem._compute_true_X3()
        ZS = np.c_[supply_instruments, X3]

        # update the products array
        updated_products = update_matrices(problem.products, {
            'ZD': (ZD, options.dtype),
            'ZS': (ZS, options.dtype)
        })

        # initialize the underlying economy with structured product and agent data
        super().__init__(problem.product_formulations, problem.agent_formulation, updated_products, problem.agents)

        # absorb any demand-side fixed effects, which have already been absorbed into X1
        if self._absorb_demand_ids is not None:
            output("Absorbing demand-side fixed effects ...")
            self.products.ZD, ZD_errors = self._absorb_demand_ids(self.products.ZD)
            if ZD_errors:
                raise exceptions.MultipleErrors(ZD_errors)

        # absorb any supply-side fixed effects, which have already been absorbed into X3
        if self._absorb_supply_ids is not None:
            output("Absorbing supply-side fixed effects ...")
            self.products.ZS, ZS_errors = self._absorb_supply_ids(self.products.ZS)
            if ZS_errors:
                raise exceptions.MultipleErrors(ZS_errors)

        # output information about the re-created problem
        output(f"Re-created the problem after {format_seconds(time.time() - start_time)}.")
        output("")
        output(self)


class InitialProgress(object):
    """Structured information about initial estimation progress."""

    problem: ProblemEconomy
    parameters: Parameters
    W: Array
    theta: Array
    objective: Array
    gradient: Array
    hessian: Array
    next_delta: Array
    delta: Array
    tilde_costs: Array
    xi_jacobian: Array
    omega_jacobian: Array

    def __init__(
            self, problem: ProblemEconomy, parameters: Parameters, W: Array, theta: Array, objective: Array,
            gradient: Array, hessian: Array, next_delta: Array, delta: Array, tilde_costs: Array, xi_jacobian: Array,
            omega_jacobian: Array) -> None:
        """Store initial progress information."""
        self.problem = problem
        self.parameters = parameters
        self.W = W
        self.theta = theta
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.next_delta = next_delta
        self.delta = delta
        self.tilde_costs = tilde_costs
        self.xi_jacobian = xi_jacobian
        self.omega_jacobian = omega_jacobian


class Progress(InitialProgress):
    """Structured information about estimation progress."""

    xi: Array
    omega: Array
    beta: Array
    gamma: Array
    converged_mapping: Dict[Hashable, bool]
    iteration_mapping: Dict[Hashable, int]
    evaluation_mapping: Dict[Hashable, int]
    clipped_costs: Array
    errors: List[Error]
    gradient_norm: Array

    def __init__(
            self, problem: ProblemEconomy, parameters: Parameters, W: Array, theta: Array, objective: Array,
            gradient: Array, hessian: Array, next_delta: Array, delta: Array, tilde_costs: Array, xi_jacobian: Array,
            omega_jacobian: Array, xi: Array, omega: Array, beta: Array, gamma: Array,
            converged_mapping: Dict[Hashable, bool], iteration_mapping: Dict[Hashable, int],
            evaluation_mapping: Dict[Hashable, int], clipped_costs: Array, errors: List[Error]) -> None:
        """Store progress information."""
        super().__init__(
            problem, parameters, W, theta, objective, gradient, hessian, next_delta, delta, tilde_costs, xi_jacobian,
            omega_jacobian
        )
        self.xi = xi
        self.omega = omega
        self.beta = beta
        self.gamma = gamma
        self.converged_mapping = converged_mapping or {}
        self.iteration_mapping = iteration_mapping or {}
        self.evaluation_mapping = evaluation_mapping or {}
        self.clipped_costs = clipped_costs
        self.errors = errors or []
        with np.errstate(invalid='ignore'):
            self.gradient_norm = np.array(np.nan, options.dtype) if gradient.size == 0 else np.abs(gradient).max()

    def format(
            self, optimization: Optimization, costs_bounds: Bounds, step: int, current_iterations: int,
            current_evaluations: int, smallest_objective: Array) -> str:
        """Format a universal display of optimization progress as a string. The first iteration will include the
        progress table header. If there are any errors, information about them will be formatted as well, regardless of
        whether or not a universal display is to be used. The smallest_objective is the smallest objective value
        encountered so far during optimization.
        """
        lines: List[str] = []

        # build the header of the universal display and structure values
        header = [
            ("GMM", "Step"), ("Optimization", "Iterations"), ("Objective", "Evaluations"),
            ("Fixed Point", "Iterations"), ("Contraction", "Evaluations"), ("Objective", "Value"),
            ("Objective", "Improvement")
        ]
        objective_improved = np.isfinite(smallest_objective) and self.objective < smallest_objective
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
            header.append(("Gradient", "Infinity Norm"))
            values.append(format_number(float(self.gradient_norm)))
        if np.isfinite(costs_bounds).any():
            header.append(("Clipped", "Marginal Costs"))
            values.append(self.clipped_costs.sum())
        header.append(("", "Theta"))
        values.append(", ".join(format_number(x) for x in self.theta))

        # build the formatter of the universal display
        widths = [max(len(k1), len(k2), options.digits + 6 if i > 4 else 0) for i, (k1, k2) in enumerate(header[:-1])]
        widths.append(max(len(header[-1][0]), len(header[-1][1]), self.theta.size * (options.digits + 8) - 2))
        formatter = TableFormatter(widths)

        # if this is the first iteration, include the header
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

        # format the values and combine the lines into one string
        if optimization._universal_display:
            lines.append(formatter(values))
        return "\n".join(lines)
