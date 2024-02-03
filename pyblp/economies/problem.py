"""Economy-level BLP problem functionality."""

import abc
import collections.abc
import functools
import time
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.sparse

from .economy import Economy
from .. import exceptions, options
from ..configurations.formulation import Formulation
from ..configurations.integration import Integration
from ..configurations.iteration import Iteration
from ..configurations.optimization import ObjectiveResults, Optimization, OptimizationProgress
from ..markets.problem_market import ProblemMarket
from ..micro import MicroDataset, MicroMoment, Moments
from ..parameters import Parameters
from ..primitives import Agents, Products
from ..results.problem_results import ProblemResults
from ..utilities.algebra import precisely_invert, precisely_identify_collinearity
from ..utilities.basics import (
    Array, Bounds, Error, RecArray, SolverStats, format_number, format_seconds, format_table, generate_items,
    get_indices, output, update_matrices, compute_finite_differences, warn
)
from ..utilities.statistics import IV, compute_gmm_moments_mean, compute_gmm_moments_jacobian_mean


class ProblemEconomy(Economy):
    """An abstract BLP problem."""

    @abc.abstractmethod
    def __init__(
            self, product_formulations: Sequence[Optional[Formulation]], agent_formulation: Optional[Formulation],
            products: RecArray, agents: RecArray, rc_types: Optional[Sequence[str]], epsilon_scale: float,
            costs_type: str) -> None:
        """Initialize the underlying economy with product and agent data."""
        super().__init__(product_formulations, agent_formulation, products, agents, rc_types, epsilon_scale, costs_type)

    def solve(
            self, sigma: Optional[Any] = None, pi: Optional[Any] = None, rho: Optional[Any] = None,
            beta: Optional[Any] = None, gamma: Optional[Any] = None, sigma_bounds: Optional[Tuple[Any, Any]] = None,
            pi_bounds: Optional[Tuple[Any, Any]] = None, rho_bounds: Optional[Tuple[Any, Any]] = None,
            beta_bounds: Optional[Tuple[Any, Any]] = None, gamma_bounds: Optional[Tuple[Any, Any]] = None,
            delta: Optional[Any] = None, method: str = '2s', initial_update: Optional[bool] = None,
            optimization: Optional[Optimization] = None, scale_objective: bool = True, check_optimality: str = 'both',
            finite_differences: bool = False, error_behavior: str = 'revert', error_punishment: float = 1,
            delta_behavior: str = 'first', iteration: Optional[Iteration] = None, fp_type: str = 'safe_linear',
            shares_bounds: Optional[Tuple[Any, Any]] = (1e-300, None), costs_bounds: Optional[Tuple[Any, Any]] = None,
            W: Optional[Any] = None, center_moments: bool = True, W_type: str = 'robust', se_type: str = 'robust',
            covariance_moments_mean: float = 0, micro_moments: Sequence[MicroMoment] = (),
            micro_sample_covariances: Optional[Any] = None,
            resample_agent_data: Optional[Callable[[int], Optional[Mapping]]] = None) -> ProblemResults:
        r"""Solve the problem.

        The problem is solved in one or more GMM steps. During each step, any parameters in :math:`\theta` are optimized
        to minimize the GMM objective value, giving the estimated :math:`\hat{\theta}`. If there are no parameters in
        :math:`\theta` (for example, in the logit model there are no nonlinear parameters and all linear parameters can
        be concentrated out), the objective is evaluated once during the step.

        If there are nonlinear parameters, the mean utility, :math:`\delta(\theta)` is computed market-by-market with
        fixed point iteration. Otherwise, it is computed analytically according to the solution of the logit model. If a
        supply side is to be estimated, marginal costs, :math:`c(\theta)`, are also computed market-by-market. Linear
        parameters are then estimated, which are used to recover structural error terms, which in turn are used to form
        the objective value. By default, the objective gradient is computed as well.

        .. note::

           This method supports :func:`parallel` processing. If multiprocessing is used, market-by-market computation of
           :math:`\delta(\theta)` (and :math:`\tilde{c}(\theta)` if a supply side is estimated), along with associated
           Jacobians, will be distributed among the processes.

        Parameters
        ----------
        sigma : `array-like, optional`
            Configuration for which elements in the lower-triangular Cholesky root of the covariance matrix for
            unobserved taste heterogeneity, :math:`\Sigma`, are fixed at zero and starting values for the other
            elements, which, if not fixed by ``sigma_bounds``, are in the vector of unknown elements, :math:`\theta`.

            Rows and columns correspond to columns in :math:`X_2`, which is formulated according
            ``product_formulations`` in :class:`Problem`. If :math:`X_2` was not formulated, this should not be
            specified, since the logit model will be estimated.

            Values above the diagonal are ignored. Zeros are assumed to be zero throughout estimation and nonzeros are,
            if not fixed by ``sigma_bounds``, starting values for unknown elements in :math:`\theta`. If any columns are
            fixed at zero, only the first few columns of integration nodes (specified in :class:`Problem`) will be used.

            To have nonzero covariances for only a subset of the random coefficients, the characteristics for those
            random coefficients with zero covariances should come first in :math:`X_2`. This can be seen by looking at
            the expression for :math:`\Sigma\Sigma'`, the actual covariance matrix of the random coefficients.

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

            If this is a scalar, it corresponds to all groups defined by the ``nesting_ids`` field of ``product_data``
            in :class:`Problem`. If this is a vector, it must have :math:`H` elements, one for each nesting group.
            Elements correspond to group IDs in the sorted order of :attr:`Problem.unique_nesting_ids`. If nesting IDs
            were not specified, this should not be specified either.

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
            counterpart in ``sigma``. If ``optimization`` does not support bounds, these will be ignored. If bounds are
            supported, the diagonal of ``sigma`` is by default bounded from below by zero.

            Values above the diagonal are ignored. Lower and upper bounds corresponding to zeros in ``sigma`` are set to
            zero. Setting a lower bound equal to an upper bound fixes the corresponding element, removing it from
            :math:`\theta`. Both ``None`` and ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to
            ``numpy.inf`` in ``ub``.

        pi_bounds : `tuple, optional`
            Configuration for :math:`\Pi` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as ``pi``. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in ``pi``. If ``optimization`` does not support bounds, these will be ignored. By default,
            ``pi`` is unbounded.

            Lower and upper bounds corresponding to zeros in ``pi`` are set to zero. Setting a lower bound equal to an
            upper bound fixes the corresponding element, removing it from :math:`\theta`. Both ``None`` and
            ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        rho_bounds : `tuple, optional`
            Configuration for :math:`\rho` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as ``rho``. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in ``rho``. If ``optimization`` does not support bounds, these will be ignored.

            If bounds are supported, ``rho`` is by default bounded from below by ``0``, which corresponds to the simple
            logit model, and bounded from above by ``0.99`` because values greater than ``1`` are inconsistent with
            utility maximization.

            Lower and upper bounds corresponding to zeros in ``rho`` are set to zero. Setting a lower bound equal to an
            upper bound fixes the corresponding element, removing it from :math:`\theta`. Both ``None`` and
            ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        beta_bounds : `tuple, optional`
            Configuration for :math:`\beta` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub`` are of the
            same size as ``beta``. Each element in ``lb`` and ``ub`` determines the lower and upper bound for its
            counterpart in ``beta``. If ``optimization`` does not support bounds, these will be ignored.

            Usually, this is left unspecified unless there is a supply side, in which case parameters on endogenous
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
            default, the solution to the logit model in :eq:`logit_delta` is used. If :math:`\rho` is specified, the
            solution to the nested logit model in :eq:`nested_logit_delta` under the initial ``rho`` is used instead.
        method : `str, optional`
            The estimation routine that will be used. The following methods are supported:

                - ``'1s'`` - One-step GMM.

                - ``'2s'`` (default) - Two-step GMM.

            Iterated GMM can be manually implemented by executing single GMM steps in a loop, in which after the first
            iteration, nonlinear parameters and weighting matrices from the last :class:`ProblemResults` are passed as
            arguments.

        initial_update : `bool, optional`
            Whether to update starting values for the mean utility :math:`\delta` and the weighting matrix :math:`W` at
            the initial parameter values before the first GMM step. This initial update will be called a zeroth step.

            By default, an initial update will not be used unless ``micro_moments`` are specified without an initial
            weighting matrix ``W``.

            .. note::

               When trying multiple parameter starting values to verify that the optimization routine converges to the
               same optimum, using ``initial_update`` is not recommended because different weighting matrices will be
               used for these different runs. A better option is to use ``optimization=Optimization('return')`` at the
               best guess for parameter values and pass :attr:`ProblemResults.updated_W` to ``W`` for each set of
               different parameter starting values.

        optimization : `Optimization, optional`
            :class:`Optimization` configuration for how to solve the optimization problem in each GMM step, which is
            only used if there are unfixed nonlinear parameters over which to optimize. By default,
            ``Optimization('l-bfgs-b', {'ftol': 0, 'gtol': 1e-8})`` is used. If available, ``Optimization('knitro')``
            may be preferable. Generally, it is recommended to consider a number of different optimization routines and
            starting values, verifying that :math:`\hat{\theta}` satisfies both the first and second order conditions.
            Choosing a routine that supports bounds (and configuring bounds) is typically a good idea. Choosing a
            routine that does not use analytic gradients will often down estimation.
        scale_objective : `bool, optional`
            Whether to scale the objective in :eq:`objective` by :math:`N`, the number of observations, in which case
            the objective after two GMM steps is equal to the :math:`J` statistic from :ref:`references:Hansen (1982)`.
            By default, the objective is scaled by :math:`N`.

            In theory the scale of the objective should not matter, but in practice having similar objective values for
            different problem sizes is helpful because similar optimization tolerances can be used.

        check_optimality : `str, optional`
            How to check for optimality (first and second order conditions) after the optimization routine finishes.
            The following configurations are supported:

                - ``'gradient'`` - Analytically compute the gradient after optimization finishes, but do not compute the
                  Hessian. Since Jacobians needed to compute standard errors will already be computed, gradient
                  computation will not take a long time. This option may be useful if Hessian computation takes a long
                  time when, for example, there are a large number of parameters.

                - ``'both'`` (default) - Also compute the Hessian with central finite differences after optimization
                  finishes.

        finite_differences : `bool, optional`
            Whether to use finite differences to compute Jacobians and the gradient instead of analytic expressions.
            Since finite differences comes with numerical approximation error and is typically slower, analytic
            expressions are used by default.

            One situation in which finite differences may be preferable is when there are a sufficiently large number of
            products and integration nodes in individual markets to make computing analytic Jacobians infeasible because
            of memory requirements. Note that an analytic expression for the Hessian has not been implemented, so when
            computed it is always approximated with finite differences.

        error_behavior : `str, optional`
            How to handle any errors. For example, there can sometimes be overflow or underflow when computing
            :math:`\delta(\theta)` at a large :math:`\hat{\theta}`. The following behaviors are supported:

                - ``'revert'`` (default) - Revert problematic values to their last computed values. If there are
                  problematic values during the first objective evaluation, revert values in :math:`\delta(\theta)` to
                  their starting values; in :math:`\tilde{c}(\hat{\theta})`, to prices; in the objective, to ``1e10``;
                  and in other matrices such as Jacobians, to zeros.

                - ``'punish'`` - Set the objective to ``1`` and its gradient to all zeros. This option along with a
                  large ``error_punishment`` can be helpful for routines that do not use analytic gradients.

                - ``'raise'`` - Raise an exception.

        error_punishment : `float, optional`
            How to scale the GMM objective value after an error. By default, the objective value is not scaled.
        delta_behavior : `str, optional`
            Configuration for the values at which the fixed point computation of :math:`\delta(\theta)` in each market
            will start. This configuration is only relevant if there are unfixed nonlinear parameters over which to
            optimize. The following behaviors are supported:

                - ``'first'`` (default) - Start at the values configured by ``delta`` during the first GMM step, and at
                  the values computed by the last GMM step for each subsequent step.

                - ``'logit'`` - Start at the solution to the logit model in :eq:`logit_delta`, or if :math:`\rho` is
                  specified, the solution to the nested logit model in :eq:`nested_logit_delta`. If the initial
                  ``delta`` is left unspecified and there is no nesting parameter being optimized over, this will
                  generally be equivalent to ``'first'``.

                - ``'last'`` - Start at the values of :math:`\delta(\theta)` computed during the last objective
                  evaluation, or, if this is the first evaluation, at the values configured by ``delta``. This behavior
                  tends to speed up computation but may introduce some instability into estimation.

        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem used to compute
            :math:`\delta(\theta)` in each market. This configuration is only relevant if there are nonlinear
            parameters, since :math:`\delta` can be estimated analytically in the logit model. By default,
            ``Iteration('squarem', {'atol': 1e-14})`` is used. Newton-based routines such as ``Iteration('lm'`)`` that
            compute the Jacobian can often be faster (especially when there are nesting parameters), but the
            Jacobian-free SQUAREM routine is used by default because it speed is often comparable and in practice it can
            be slightly more stable.
        fp_type : `str, optional`
            Configuration for the type of contraction mapping used to compute :math:`\delta(\theta)`. The following
            types are supported:

                - ``'safe_linear'`` (default) - The standard linear contraction mapping in :eq:`contraction` (or
                  :eq:`nested_contraction` when there is nesting) with safeguards against numerical overflow.
                  Specifically, :math:`\max_j V_{ijt}` (or :math:`\max_j V_{ijt} / (1 - \rho_{h(j)})` when there is
                  nesting) is subtracted from :math:`V_{ijt}` and the logit expression for choice probabilities in
                  :eq:`probabilities` (or :eq:`nested_probabilities`) is re-scaled accordingly. Such re-scaling is known
                  as the log-sum-exp trick.

                - ``'linear'`` - The standard linear contraction mapping without safeguards against numerical overflow.
                  This option may be preferable to ``'safe_linear'`` if utilities are reasonably small and unlikely to
                  create overflow problems.

                - ``'nonlinear'`` - Iteration over :math:`\exp \delta_{jt}` instead of :math:`\delta_{jt}`. This can be
                  faster than ``'linear'`` because it involves fewer logarithms. Also, following
                  :ref:`references:Brunner, Heiss, Romahn, and Weiser (2017)`, the :math:`\exp \delta_{jt}` term can be
                  cancelled out of the expression because it also appears in the numerator of :eq:`probabilities` in the
                  definition of :math:`s_{jt}(\delta, \theta)`. This second trick only works when there are no
                  nesting parameters.

                - ``'safe_nonlinear'`` - Exponentiated version with minimal safeguards against numerical overflow.
                  Specifically, :math:`\max_j \mu_{ijt}` is subtracted from :math:`\mu_{ijt}`. This helps with stability
                  but is less helpful than subtracting from the full :math:`V_{ijt}`, so this version is less stable
                  than ``'safe_linear'``.

            This option is only relevant if ``sigma`` or ``pi`` are specified because :math:`\delta` can be estimated
            analytically in the logit model with :eq:`logit_delta` and in the nested logit model with
            :eq:`nested_logit_delta`.

        shares_bounds : `tuple, optional`
            Configuration for :math:`s_{jt}(\delta, \theta)` bounds in the contraction in :eq:`contraction` of the form
            ``(lb, ub)``, in which both ``lb`` and ``ub`` are floats or ``None``. By default, simulated shares are
            bounded from below by ``1e-300``. This is only relevant if ``fp_type`` is ``'safe_linear'`` or ``'linear'``.
            Bounding shares in the contraction does nothing with a nonlinear fixed point.

            It can be particularly helpful to bound shares in the contraction from below by a small number to prevent
            the contraction from failing when there are issues with zero or negative simulated shares. Zero shares can
            occur when there are underflow issues and negative shares can occur when there are issues with the numerical
            integration routine having negative integration weights (e.g., for sparse grid integration).

            The idea is that a small lower bound will allow the contraction to converge even when it encounters some
            issues with small or negative shares. However, if these issues are unlikely, disabling this behavior can
            speed up the iteration routine because fewer checks will be done.

            Both ``None`` and ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        costs_bounds : `tuple, optional`
            Configuration for :math:`c_{jt}(\theta)` bounds of the form ``(lb, ub)``, in which both ``lb`` and ``ub``
            are floats or ``None``. This is only relevant if :math:`X_3` was formulated by ``product_formulations`` in
            :class:`Problem`. By default, marginal costs are unbounded.

            When ``costs_type`` in :class:`Problem` is ``'log'``, nonpositive :math:`c(\theta)` values can create
            problems when computing :math:`\tilde{c}(\theta) = \log c(\theta)`. One solution is to set ``lb`` to a small
            number. Rows in Jacobians associated with clipped marginal costs will be zero.

            Both ``None`` and ``numpy.nan`` are converted to ``-numpy.inf`` in ``lb`` and to ``numpy.inf`` in ``ub``.

        W : `array-like, optional`
            Starting values for the weighting matrix, :math:`W`. By default, the 2SLS weighting matrix in :eq:`2sls_W`
            is used, unless there are any ``micro_moments``, in which case an ``initial_update`` will be used to update
            starting values :math:`W` and the mean utility :math:`\delta` at the initial parameter values before the
            first GMM step.
        center_moments : `bool, optional`
            Whether to center each column of the demand- and supply-side moments :math:`g` before updating the weighting
            matrix :math:`W` according to :eq:`W`. By default, the moments are centered. This has no effect if
            ``W_type`` is ``'unadjusted'``.
        W_type : `str, optional`
            How to update the weighting matrix. This has no effect if ``method`` is ``'1s'``. Usually, ``se_type``
            should be the same. The following types are supported:

                - ``'robust'`` (default) - Heteroscedasticity robust weighting matrix defined in :eq:`W` and
                  :eq:`robust_S`.

                - ``'clustered'`` - Clustered weighting matrix defined in :eq:`W` and :eq:`clustered_S`. Clusters must
                  be defined by the ``clustering_ids`` field of ``product_data`` in :class:`Problem`.

                - ``'unadjusted'`` - Homoskedastic weighting matrix defined in :eq:`W` and :eq:`unadjusted_S`.

            This only affects the standard demand- and supply-side block of the updated weighting matrix. If there are
            micro moments, this matrix will be block-diagonal with a micro moment block equal to the inverse of the
            scaled covariance matrix defined in :eq:`scaled_micro_moment_covariances`.

        se_type : `str, optional`
            How to compute parameter covariances and standard errors. Usually, ``W_type`` should be the same. The
            following types are supported:

                - ``'robust'`` (default) - Heteroscedasticity robust covariances defined in :eq:`covariances` and
                  :eq:`robust_S`.

                - ``'clustered'`` - Clustered covariances defined in :eq:`covariances` and :eq:`clustered_S`. Clusters
                  must be defined by the ``clustering_ids`` field of ``product_data`` in :class:`Problem`.

                - ``'unadjusted'`` - Homoskedastic covariances defined in :eq:`unadjusted_covariances`, which are
                  computed under the assumption that the weighting matrix is optimal.

            This only affects the standard demand- and supply-side block of the matrix of averaged moment covariances.
            If there are micro moments, the :math:`S` matrix defined in the expressions referenced above will be
            block-diagonal with a micro moment block equal to the scaled covariance matrix defined in
            :eq:`scaled_micro_moment_covariances`.

        covariance_moments_mean : `float, optional`
            If ``covariance_instruments`` were specified in ``product_data``, this can be used to choose a
            :math:`m \neq 0` in covariance moments :math:`E[g_{C,jt}] = E[(\xi_{jt}\omega_{jt} - m)Z_{C,jt}] = 0` where
            :math:`m` is by default zero. This can be used for sensitivity testing to see how different covariances
            may affect estimates.
        micro_moments : `sequence of MicroMoment, optional`
            Configurations for the :math:`M_M` :class:`MicroMoment` instances that will be added to the standard set of
            moments. By default, no micro moments are used, so :math:`M_M = 0`.

            When micro moments are specified, unless an initial weighting matrix ``W`` is specified as well (with a
            lower right micro moment block that reflects micro moment covariances), an ``initial_update`` will be used
            to update starting values :math:`W` and the mean utility :math:`\delta` at the initial parameter values
            before the first GMM step.

            .. note::

               When trying multiple parameter starting values to verify that the optimization routine converges to the
               same optimum, using ``initial_update`` is not recommended because different weighting matrices will be
               used for these different runs. A better option is to use ``optimization=Optimization('return')`` at the
               best guess for parameter values and pass :attr:`ProblemResults.updated_W` to ``W`` for each set of
               different parameter starting values.

        micro_sample_covariances : `array-like, optional`
            Sample covariance matrix for the :math:`M_M` micro moments. By default, their asymptotic covariance matrix
            is computed according to :eq:`scaled_micro_moment_covariances`. This override could be used, for example, if
            instead of estimating covariances at some estimated :math:`\hat{\theta}`, one wanted to use a boostrap
            procedure to compute their covariances directly from the micro data.

        resample_agent_data : `callable, optional`
            If specified, simulation error in moment covariances will be accounted for by resampling
            :math:`r = 1, \dots, R` sets of agents by iteratively calling this function, which should be of the
            following form::

                resample_agent_data(index) --> agent_data or None

            where ``index`` increments from ``0`` to ``1`` and so on and ``agent_data`` is the corresponding resampled
            agent data, which should be a resampled version of the ``agent_data`` passed to :class:`Problem`. Each
            ``index`` should correspond to a different set of randomly drawn agent data, with different integration
            nodes and demographics. If ``index`` is larger than :math:`R - 1`, this function should return ``None``,
            at which point agents will stop being resampled.

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

        # validate settings
        if method not in {'1s', '2s'}:
            raise TypeError("method must be '1s' or '2s'.")
        if optimization is None:
            optimization = Optimization('l-bfgs-b', {'ftol': 0, 'gtol': 1e-8})
        elif not isinstance(optimization, Optimization):
            raise TypeError("optimization must be None or an Optimization instance.")
        if check_optimality not in {'gradient', 'both'}:
            raise ValueError("check_optimality must be 'gradient' or 'both'.")
        if error_behavior not in {'revert', 'punish', 'raise'}:
            raise ValueError("error_behavior must be 'revert', 'punish', or 'raise'.")
        if not isinstance(error_punishment, (float, int)) or error_punishment < 0:
            raise ValueError("error_punishment must be a positive float.")
        if delta_behavior not in {'last', 'logit', 'first'}:
            raise ValueError("delta_behavior must be 'last', 'logit', or 'first'.")
        iteration = self._coerce_optional_delta_iteration(iteration)
        self._validate_fp_type(fp_type)
        if W_type not in {'robust', 'unadjusted', 'clustered'}:
            raise ValueError("W_type must be 'robust', 'unadjusted', or 'clustered'.")
        if se_type not in {'robust', 'unadjusted', 'clustered'}:
            raise ValueError("se_type must be 'robust', 'unadjusted', or 'clustered'.")
        if 'clustered' in {W_type, se_type}:
            if 'clustering_ids' not in self.products.dtype.names or self.products.clustering_ids.size == 0:
                raise ValueError(
                    "W_type or se_type is 'clustered' but clustering_ids were not specified in product_data."
                )
        if not isinstance(covariance_moments_mean, (int, float)):
            raise ValueError("covariance_moments_mean must be a float.")

        # configure or validate bounds on shares and costs
        shares_bounds = self._coerce_optional_bounds(shares_bounds, 'shares_bounds')
        costs_bounds = self._coerce_optional_bounds(costs_bounds, 'costs_bounds')

        # validate and structure micro moments before outputting related information
        moments = Moments(micro_moments, self)
        micro_moment_covariances = None
        if moments.MM > 0:
            output("")
            output(moments.format("Micro Moments"))
            if micro_sample_covariances is not None:
                micro_moment_covariances = np.c_[np.asarray(micro_sample_covariances, options.dtype)]
                if micro_moment_covariances.shape != (moments.MM, moments.MM):
                    raise ValueError(f"micro_sample_covariances must be a square {moments.MM} by {moments.MM} matrix.")
                self._require_psd(micro_moment_covariances, "micro_sample_covariances")
                self._detect_singularity(micro_moment_covariances, "micro_sample_covariances")

        # determine whether to check micro moment collinearity
        detect_micro_collinearity = (
            moments.MM > 0 and
            options.detect_micro_collinearity and
            (options.collinear_atol > 0 or options.collinear_rtol > 0)
        )

        # validate any agent data resampler
        if resample_agent_data is not None and not callable(resample_agent_data):
            raise TypeError("resample_agent_data must be None or a function.")

        # choose whether to do an initial update
        if initial_update is None:
            initial_update = bool(moments.MM > 0 and W is None)
        elif not initial_update and moments.MM > 0 and W is None:
            raise ValueError("initial_update cannot be False with micro_moments and no initial W specified.")

        # validate parameters before compressing unfixed parameters into theta and outputting related information
        parameters = Parameters(
            self, sigma, pi, rho, beta, gamma, sigma_bounds, pi_bounds, rho_bounds, beta_bounds, gamma_bounds,
            bounded=optimization._supports_bounds, allow_linear_nans=True
        )
        theta = parameters.compress()
        theta_bounds = parameters.compress_bounds()
        if parameters.fixed or parameters.unfixed:
            output("")
            output(parameters.format("Initial Values"))
            if parameters.fixed or optimization._supports_bounds:
                output("")
                output(parameters.format_lower_bounds("Lower Bounds"))
                output("")
                output(parameters.format_upper_bounds("Upper Bounds"))
                output("")

        # warn if it seems like the model is under-identified
        M = self.MD + self.MS + self.MC + moments.MM
        P_total = len(parameters.unfixed) + len(parameters.eliminated)
        if P_total > M:
            warn(
                f"The model may be under-identified. The total number of unfixed parameters is {P_total}, which is "
                f"more than the total number of moments, {M}. Consider checking whether instruments were properly "
                f"specified when initializing the problem, and whether parameters were properly configured when "
                f"solving the problem."
            )

        # load or compute the weighting matrix
        if W is not None:
            W = np.c_[np.asarray(W, options.dtype)]
            if W.shape != (M, M):
                raise ValueError(f"W must be a square {M} by {M} matrix.")
            self._require_psd(W, "W")
            self._detect_singularity(W, "W")
        else:
            S = scipy.linalg.block_diag(
                self.products.ZD.T @ self.products.ZD / self.N,
                self.products.ZS.T @ self.products.ZS / self.N,
                self.products.ZC.T @ self.products.ZC / self.N,
            )
            self._detect_singularity(S, "the 2SLS weighting matrix")
            W, successful = precisely_invert(S)
            if not successful:
                raise ValueError("Failed to compute the 2SLS weighting matrix. There may be instrument collinearity.")

            # an initial update will be used when there are micro moments, so this initial block does not matter
            if moments.MM > 0:
                assert initial_update
                W = scipy.linalg.block_diag(W, np.zeros((moments.MM, moments.MM), options.dtype))

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
            if self.costs_type == 'linear':
                tilde_costs = self.products.prices
            else:
                assert self.costs_type == 'log'
                tilde_costs = np.log(self.products.prices)

        # initialize micro moments as all zeros, which will only be used if there are computation errors during the
        #   first objective evaluation
        micro = np.zeros((moments.MM, 1), options.dtype)

        # initialize Jacobians as all zeros, which will only be used if there are computation errors during the first
        #   objective evaluation
        xi_jacobian = np.zeros((self.N, parameters.P), options.dtype)
        omega_jacobian = np.full((self.N, parameters.P), 0 if self.K3 > 0 else np.nan, options.dtype)
        micro_jacobian = np.zeros((moments.MM, parameters.P), options.dtype)

        # initialize the objective as a large number and its gradient and hessian as all zeros, which will only be used
        #   if there are computation errors during the first objective evaluation
        objective = np.array(1e10, options.dtype)
        gradient = np.zeros((parameters.P, 1), options.dtype)
        hessian = np.zeros((parameters.P, parameters.P), options.dtype)

        # iterate over each GMM step
        step = 0 if initial_update else 1
        last_results = None
        while True:
            # collect inputs into linear parameter estimation
            X_list = [self.products.X1[:, parameters.eliminated_beta_index.flat]]
            Z_list = [self.products.ZD]
            if self.K3 > 0:
                X_list.append(self.products.X3[:, parameters.eliminated_gamma_index.flat])
                Z_list.append(self.products.ZS)

            # initialize an IV model for linear parameter estimation
            iv = IV(X_list, Z_list, W[:self.MD + self.MS, :self.MD + self.MS])
            self._handle_errors(iv.errors, error_behavior)

            # wrap computation of progress information with step-specific information
            compute_step_progress = functools.partial(
                self._compute_progress, parameters, moments, iv, W, scale_objective, error_behavior, error_punishment,
                delta_behavior, iteration, fp_type, shares_bounds, costs_bounds, finite_differences,
                covariance_moments_mean, resample_agent_data
            )

            # initialize optimization progress
            iteration_stats: List[Dict[Hashable, SolverStats]] = []
            smallest_objective = np.inf
            progress = InitialProgress(
                self, parameters, moments, W, theta, objective, gradient, hessian, delta, delta, tilde_costs, micro,
                xi_jacobian, omega_jacobian, micro_jacobian
            )

            # define the objective function
            def wrapper(new_theta: Array, iterations: int, evaluations: int) -> ObjectiveResults:
                """Compute and output progress associated with a single objective evaluation."""
                nonlocal iteration_stats, smallest_objective, progress, detect_micro_collinearity
                assert optimization is not None and shares_bounds is not None and costs_bounds is not None
                progress_start_time = time.time()
                progress = compute_step_progress(
                    new_theta, progress, optimization._compute_gradient, compute_hessian=False,
                    compute_micro_covariances=False, detect_micro_collinearity=detect_micro_collinearity,
                    compute_simulation_covariances=False,
                )
                iteration_stats.append(progress.iteration_stats)
                progress_time = time.time() - progress_start_time
                formatted_progress = progress.format(
                    optimization, shares_bounds, costs_bounds, step, iterations, evaluations, progress_time,
                    smallest_objective
                )
                if formatted_progress:
                    output(formatted_progress)
                smallest_objective = min(smallest_objective, progress.objective)
                detect_micro_collinearity = False
                return (
                    progress.objective,
                    progress.gradient if optimization._compute_gradient else None,
                    OptimizationProgress(progress),
                )

            # optimize theta if there are parameters to optimize and this isn't the initial update step
            optimization_stats = SolverStats()
            optimization_start_time = optimization_end_time = time.time()
            if parameters.P > 0 and step > 0:
                output(f"Starting optimization ...")
                output("")
                theta, optimization_stats = optimization._optimize(theta, theta_bounds, wrapper)
                status = "completed" if optimization_stats.converged else "failed"
                optimization_end_time = time.time()
                optimization_time = optimization_end_time - optimization_start_time
                if not optimization_stats.converged:
                    self._handle_errors([exceptions.ThetaConvergenceError()], error_behavior)
                output("")
                output(f"Optimization {status} after {format_seconds(optimization_time)}.")

            # identify what will be done when computing results
            initial_step = step == 0
            last_step = step == 2 or (method == '1s' and step == 1)
            compute_gradient = parameters.P > 0
            compute_hessian = compute_gradient and check_optimality == 'both' and step > 0
            compute_micro_covariances = moments.MM > 0 and micro_moment_covariances is None
            compute_simulation_covariances = resample_agent_data is not None

            # use progress information computed at the optimal theta to compute results for the step
            if initial_step:
                output("Updating starting values for the weighting matrix and delta ...")
            elif compute_hessian and not last_step:
                output("Computing the Hessian and updating the weighting matrix ...")
            elif compute_hessian:
                output("Computing the Hessian and estimating standard errors ...")
            elif not last_step:
                output("Updating the weighting matrix ...")
            else:
                output("Estimating standard errors ...")
            final_progress = compute_step_progress(
                theta, progress, compute_gradient, compute_hessian, compute_micro_covariances,
                detect_micro_collinearity, compute_simulation_covariances
            )
            iteration_stats.append(final_progress.iteration_stats)
            optimization_stats.evaluations += 1
            detect_micro_collinearity = False
            results = ProblemResults(
                final_progress, last_results, step, last_step, step_start_time, optimization_start_time,
                optimization_end_time, optimization_stats, iteration_stats, scale_objective, shares_bounds,
                costs_bounds, micro_moment_covariances, center_moments, W_type, se_type, covariance_moments_mean
            )
            self._handle_errors(results._errors, error_behavior)
            output(f"Computed results after {format_seconds(results.total_time - results.optimization_time)}.")

            # store the last results and return results from the final step
            last_results = results
            output("")
            if last_step:
                output(results)
                return results
            if step > 0:
                output(results._format_summary())
                output("")

            # update vectors and matrices
            delta = results.delta
            tilde_costs = results.tilde_costs
            xi_jacobian = results.xi_by_theta_jacobian
            omega_jacobian = results.omega_by_theta_jacobian
            W = results.updated_W
            step += 1
            step_start_time = time.time()

    def _compute_progress(
            self, parameters: Parameters, moments: Moments, iv: IV, W: Array, scale_objective: bool,
            error_behavior: str, error_punishment: float, delta_behavior: str, iteration: Iteration, fp_type: str,
            shares_bounds: Bounds, costs_bounds: Bounds, finite_differences: bool, covariance_moments_mean: float,
            resample_agent_data: Optional[Callable[[int], Optional[Mapping]]], theta: Array,
            progress: 'InitialProgress', compute_gradient: bool, compute_hessian: bool,
            compute_micro_covariances: bool, detect_micro_collinearity: bool,
            compute_simulation_covariances: bool, agents_override: Optional[RecArray] = None) -> 'Progress':
        """Compute demand- and supply-side contributions before recovering the linear parameters and structural error
        terms. Then, form the GMM objective value and its gradient. Finally, handle any errors that were encountered
        before structuring relevant progress information.
        """
        errors: List[Error] = []

        # expand theta
        sigma, pi, rho, beta, gamma = parameters.expand(theta)

        # initialize delta, micro moments, their Jacobians, micro moment covariances, micro moment values, indices of
        #   clipped shares, and fixed point statistics so that they can be filled
        delta = np.zeros((self.N, 1), options.dtype)
        micro = np.zeros((moments.MM, 1), options.dtype)
        xi_jacobian = np.zeros((self.N, parameters.P), options.dtype)
        micro_jacobian = np.zeros((moments.MM, parameters.P), options.dtype)
        micro_covariances = np.zeros((moments.MM, moments.MM), options.dtype)
        micro_values = np.full((moments.MM, 1), np.nan, options.dtype)
        clipped_shares = np.zeros((self.N, 1), np.bool_)
        iteration_stats: Dict[Hashable, SolverStats] = {}

        # initialize transformed marginal costs, their Jacobian, and indices of clipped costs so that they can be filled
        if self.K3 == 0:
            tilde_costs = np.full((self.N, 0), np.nan, options.dtype)
            omega_jacobian = np.full((self.N, parameters.P), np.nan, options.dtype)
            clipped_costs = np.zeros((self.N, 1), np.bool_)
        else:
            tilde_costs = np.zeros((self.N, 1), options.dtype)
            omega_jacobian = np.zeros((self.N, parameters.P), options.dtype)
            clipped_costs = np.zeros((self.N, 1), np.bool_)

        # only do market-by-market computation when necessary
        compute_jacobians = compute_gradient and not finite_differences
        if self.K2 == self.K3 == moments.MM == 0 and (parameters.P == 0 or not compute_jacobians):
            delta = self._compute_logit_delta(rho)
        else:
            next_delta = progress.next_delta
            if delta_behavior == 'logit':
                next_delta = self._compute_logit_delta(rho)

            # get market indices for any overridden agents
            agent_market_indices_override = None
            if agents_override is not None:
                agent_market_indices_override = get_indices(agents_override.market_ids)

            def market_factory(
                    s: Hashable) -> (
                    Tuple[
                        ProblemMarket, Array, Array, Array, Moments, Iteration, str, Bounds, Bounds, bool, bool, bool
                    ]):
                """Build a market along with arguments used to compute delta, micro moment contributions, transformed
                marginal costs, and Jacobians.
                """
                agents_override_s = None
                if agents_override is not None:
                    assert agent_market_indices_override is not None
                    agents_override_s = agents_override[agent_market_indices_override[s]]

                market_s = ProblemMarket(self, s, parameters, sigma, pi, rho, beta, agents_override=agents_override_s)
                delta_s = next_delta[self._product_market_indices[s]]
                last_delta_s = progress.delta[self._product_market_indices[s]]
                last_tilde_costs_s = progress.tilde_costs[self._product_market_indices[s]]
                return (
                    market_s, delta_s, last_delta_s, last_tilde_costs_s, moments, iteration, fp_type, shares_bounds,
                    costs_bounds, compute_jacobians, compute_micro_covariances, detect_micro_collinearity
                )

            # if necessary, identify micro datasets in which there could possibly be collinearity issues
            parts_collinearity_candidates: Dict[MicroDataset, List[int]] = {}
            if detect_micro_collinearity:
                for p, part in enumerate(moments.micro_parts):
                    parts_collinearity_candidates.setdefault(part.dataset, []).append(p)
                parts_collinearity_candidates = {d: v for d, v in parts_collinearity_candidates.items() if len(v) > 1}

            # compute delta, contributions to micro moment parts, transformed marginal costs, Jacobians, and
            #   covariances market-by-market
            parts_numerator_mapping: Dict[Hashable, Array] = {}
            parts_denominator_mapping: Dict[Hashable, Array] = {}
            parts_numerator_jacobian_mapping: Dict[Hashable, Array] = {}
            parts_denominator_jacobian_mapping: Dict[Hashable, Array] = {}
            parts_covariances_numerator_mapping: Dict[Hashable, Array] = {}
            parts_collinearity_candidate_values: Dict[Hashable, Dict[MicroDataset, Array]] = {}
            generator = generate_items(self.unique_market_ids, market_factory, ProblemMarket.solve)
            for t, generated_t in generator:
                (
                    delta_t, xi_jacobian_t, parts_numerator_t, parts_denominator_t, parts_numerator_jacobian_t,
                    parts_denominator_jacobian_t, parts_covariances_numerator_t, weights_mapping_t, values_mapping_t,
                    clipped_shares_t, iteration_stats_t, tilde_costs_t, omega_jacobian_t, clipped_costs_t, errors_t
                ) = generated_t

                delta[self._product_market_indices[t]] = delta_t
                xi_jacobian[self._product_market_indices[t], :parameters.P] = xi_jacobian_t
                clipped_shares[self._product_market_indices[t]] = clipped_shares_t
                iteration_stats[t] = iteration_stats_t
                parts_numerator_mapping[t] = scipy.sparse.csr_matrix(parts_numerator_t)
                parts_denominator_mapping[t] = scipy.sparse.csr_matrix(parts_denominator_t)
                if compute_jacobians:
                    parts_numerator_jacobian_mapping[t] = scipy.sparse.csr_matrix(parts_numerator_jacobian_t)
                    parts_denominator_jacobian_mapping[t] = scipy.sparse.csr_matrix(parts_denominator_jacobian_t)
                if compute_micro_covariances:
                    parts_covariances_numerator_mapping[t] = scipy.sparse.csr_matrix(parts_covariances_numerator_t)
                if detect_micro_collinearity:
                    parts_collinearity_candidate_values[t] = {}
                    for dataset, part_indices in parts_collinearity_candidates.items():
                        if dataset in weights_mapping_t:
                            nonzero = np.nonzero(weights_mapping_t[dataset])
                            parts_collinearity_candidate_values[t][dataset] = np.column_stack(
                                [values_mapping_t[p][nonzero].flatten() for p in part_indices]
                            )
                if self.K3 > 0:
                    tilde_costs[self._product_market_indices[t]] = tilde_costs_t
                    clipped_costs[self._product_market_indices[t]] = clipped_costs_t
                    if compute_jacobians:
                        omega_jacobian[self._product_market_indices[t], :parameters.P] = omega_jacobian_t

                errors.extend(errors_t)

            # aggregate micro moments, their Jacobian, and their covariances across all markets (this is done after
            #   market-by-market computation to preserve numerical stability with different market orderings)
            if moments.MM > 0:
                with np.errstate(all='ignore'):
                    # construct micro moment parts
                    parts_numerator = scipy.sparse.csr_matrix((moments.PM, 1), dtype=options.dtype)
                    parts_denominator = scipy.sparse.csr_matrix((moments.PM, 1), dtype=options.dtype)
                    for t in self.unique_market_ids:
                        parts_numerator += parts_numerator_mapping[t]
                        parts_denominator += parts_denominator_mapping[t]

                    parts_numerator = parts_numerator.toarray()
                    parts_denominator = parts_denominator.toarray()
                    parts_values = parts_numerator / parts_denominator

                    # from the parts, construct micro moment values and if needed their gradient too
                    micro_gradients = np.zeros((moments.MM, moments.PM), options.dtype)
                    for m, moment in enumerate(moments.micro_moments):
                        part_indices = [moments.micro_parts.index(p) for p in moment.parts]
                        micro_value = moment.compute_value(parts_values[part_indices])
                        micro_value = np.asarray(micro_value).flatten()
                        if micro_value.size != 1:
                            raise TypeError(f"compute_value of micro moment '{moment}' should return a float.")
                        if not np.isfinite(micro_value):
                            warn(
                                f"compute_value of micro moment '{moment}' returned "
                                f"{format_number(micro_value).strip()}."
                            )

                        micro_values[m] = micro_value

                        if compute_jacobians or compute_micro_covariances:
                            micro_gradient = moment.compute_gradient(parts_values[part_indices])
                            micro_gradient = np.asarray(micro_gradient, options.dtype).flatten()
                            if micro_gradient.size != len(moment.parts):
                                raise ValueError(
                                    f"compute_gradient of micro moment '{moment}' should return an array of size "
                                    f"{len(moment.parts)}, but it returned one of size {micro_gradient.size}."
                                )
                            for p, part in enumerate(moment.parts):
                                if not np.isfinite(micro_gradient[p]):
                                    warn(
                                        f"compute_gradient of micro moment '{moment}' returned "
                                        f"{format_number(micro_gradient[p]).strip()} for part '{part}'."
                                    )

                            micro_gradients[m, part_indices] = micro_gradient

                    # construct micro moments
                    micro = moments.values - micro_values

                    # construct the micro moment Jacobian
                    if compute_jacobians:
                        # construct the micro moment parts Jacobian
                        parts_numerator_jacobian = scipy.sparse.csr_matrix(
                            (moments.PM, parameters.P), dtype=options.dtype
                        )
                        parts_denominator_jacobian = scipy.sparse.csr_matrix(
                            (moments.PM, parameters.P), dtype=options.dtype
                        )
                        for t in self.unique_market_ids:
                            parts_numerator_jacobian += parts_numerator_jacobian_mapping[t]
                            parts_denominator_jacobian += parts_denominator_jacobian_mapping[t]

                        parts_numerator_jacobian = parts_numerator_jacobian.toarray()
                        parts_denominator_jacobian = parts_denominator_jacobian.toarray()
                        parts_jacobian = (
                            (parts_numerator_jacobian - parts_values * parts_denominator_jacobian) / parts_denominator
                        )

                        # construc the micro moment Jacobian with the product rule
                        micro_jacobian = -micro_gradients @ parts_jacobian

                    # construct micro moment covariances from part covariances
                    if compute_micro_covariances:
                        # construct non-centered, non-scaled, non-symmetric part covariances
                        parts_covariances_numerator = scipy.sparse.csr_matrix(
                            (moments.PM, moments.PM), dtype=options.dtype
                        )
                        for t in self.unique_market_ids:
                            parts_covariances_numerator += parts_covariances_numerator_mapping[t]

                        parts_covariances_numerator = parts_covariances_numerator.toarray()
                        parts_covariances = parts_covariances_numerator / parts_denominator

                        # subtract away means from second moments and scale by observation counts
                        for p1, (part1, value1) in enumerate(zip(moments.micro_parts, parts_values)):
                            for p2, (part2, value2) in enumerate(zip(moments.micro_parts, parts_values)):
                                if p2 <= p1 and part1.dataset == part2.dataset:
                                    parts_covariances[p2, p1] -= value1 * value2
                                    parts_covariances[p2, p1] /= part1.dataset.observations

                        # fill the lower triangle
                        lower_indices = np.tril_indices(moments.PM, -1)
                        parts_covariances[lower_indices] = parts_covariances.T[lower_indices]

                        # compute micro moment covariances with the delta method
                        micro_covariances = micro_gradients @ parts_covariances @ micro_gradients.T
                        self._detect_singularity(micro_covariances, "the estimated covariance matrix of micro moments")

                    # detect collinearity between micro moment parts
                    if detect_micro_collinearity:
                        for dataset, part_indices in parts_collinearity_candidates.items():
                            market_ids = self.unique_market_ids if dataset.market_ids is None else dataset.market_ids
                            values = np.row_stack([parts_collinearity_candidate_values[t][dataset] for t in market_ids])
                            collinear, successful = precisely_identify_collinearity(values)
                            common_message = (
                                "To disable collinearity checks, set "
                                "options.collinear_atol = options.collinear_rtol = 0."
                            )
                            if not successful:
                                warn(
                                    f"Failed to compute the QR decomposition for micro dataset '{dataset.name}' while "
                                    f"checking for collinearity issues. {common_message}"
                                )
                            if collinear.any():
                                labels = [moments.micro_parts[p].name for p in part_indices]
                                collinear_labels = ", ".join(f"'{l}'" for l, c in zip(labels, collinear) if c)
                                warn(
                                    f"Detected collinearity issues with the values of micro moment parts "
                                    f"[{collinear_labels}] and at least one other micro moment part based on micro "
                                    f"dataset '{dataset.name}'. {common_message}"
                                )

        # replace invalid elements in delta, micro moments, and transformed marginal costs with their last values
        bad_delta_index = ~np.isfinite(delta)
        if np.any(bad_delta_index):
            delta[bad_delta_index] = progress.delta[bad_delta_index]
            errors.append(exceptions.DeltaReversionError(bad_delta_index))
        if moments.MM > 0:
            bad_micro_index = ~np.isfinite(micro)
            if np.any(bad_micro_index):
                micro[bad_micro_index] = progress.micro[bad_micro_index]
                errors.append(exceptions.MicroMomentsReversionError(bad_micro_index))
        if self.K3 > 0:
            bad_tilde_costs_index = ~np.isfinite(tilde_costs)
            if np.any(bad_tilde_costs_index):
                tilde_costs[bad_tilde_costs_index] = progress.tilde_costs[bad_tilde_costs_index]
                errors.append(exceptions.CostsReversionError(bad_tilde_costs_index))

        # replace invalid elements in the Jacobians with their last values
        if compute_jacobians:
            bad_xi_jacobian_index = ~np.isfinite(xi_jacobian)
            if np.any(bad_xi_jacobian_index):
                xi_jacobian[bad_xi_jacobian_index] = progress.xi_jacobian[bad_xi_jacobian_index]
                errors.append(exceptions.XiByThetaJacobianReversionError(bad_xi_jacobian_index))
            if moments.MM > 0:
                bad_micro_jacobian_index = ~np.isfinite(micro_jacobian)
                if np.any(bad_micro_jacobian_index):
                    micro_jacobian[bad_micro_jacobian_index] = progress.micro_jacobian[bad_micro_jacobian_index]
                    errors.append(exceptions.MicroMomentsByThetaJacobianReversionError(bad_micro_jacobian_index))
            if self.K3 > 0:
                bad_omega_jacobian_index = ~np.isfinite(omega_jacobian)
                if np.any(bad_omega_jacobian_index):
                    omega_jacobian[bad_omega_jacobian_index] = progress.omega_jacobian[bad_omega_jacobian_index]
                    errors.append(exceptions.OmegaByThetaJacobianReversionError(bad_omega_jacobian_index))

        # optionally compute Jacobians with central finite differences
        if compute_gradient and finite_differences and parameters.P > 0:
            def compute_perturbed_stack(perturbed_theta: Array) -> Array:
                """Evaluate a stack of xi, micro moments, and omega at a perturbed parameter vector."""
                perturbed_progress = self._compute_progress(
                    parameters, moments, iv, W, scale_objective, error_behavior, error_punishment, delta_behavior,
                    iteration, fp_type, shares_bounds, costs_bounds, finite_differences=False,
                    covariance_moments_mean=covariance_moments_mean, resample_agent_data=None, theta=perturbed_theta,
                    progress=progress, compute_gradient=False, compute_hessian=False, compute_micro_covariances=False,
                    detect_micro_collinearity=False, compute_simulation_covariances=False,
                )
                perturbed_stack = perturbed_progress.iv_delta
                if moments.MM > 0:
                    perturbed_stack = np.r_[perturbed_stack, perturbed_progress.micro]
                if self.K3 > 0:
                    perturbed_stack = np.r_[perturbed_stack, perturbed_progress.iv_tilde_costs]
                return perturbed_stack

            # compute and unstack the Jacobians
            stack_jacobian = compute_finite_differences(compute_perturbed_stack, theta)
            xi_jacobian = stack_jacobian[:self.N]
            if moments.MM > 0:
                micro_jacobian = stack_jacobian[self.N:self.N + moments.MM]
            if self.K3 > 0:
                omega_jacobian = stack_jacobian[-self.N:]

        # subtract contributions of linear parameters in theta
        iv_delta = delta.copy()
        iv_tilde_costs = tilde_costs.copy()
        if not parameters.eliminated_beta_index.all():
            theta_beta = np.c_[beta[~parameters.eliminated_beta_index]]
            iv_delta -= self._compute_true_X1(index=~parameters.eliminated_beta_index.flatten()) @ theta_beta
        if not parameters.eliminated_gamma_index.all():
            theta_gamma = np.c_[gamma[~parameters.eliminated_gamma_index]]
            iv_tilde_costs -= self._compute_true_X3(index=~parameters.eliminated_gamma_index.flatten()) @ theta_gamma

        # check whether delta and tilde cost Jacobians need to be explicitly converted into xi and omega Jacobians to
        #   account for concentrating out linear parameters (without covariance moments, it turns out that convenient
        #   orthogonality conditions makes this unnecessary, which cuts down significantly on additional computation)
        convert_jacobians = self.MC > 0 and (compute_gradient or finite_differences) and parameters.P > 0

        # absorb any fixed effects
        if self._absorb_demand_ids is not None:
            iv_delta, demand_absorption_errors = self._absorb_demand_ids(iv_delta)
            errors.extend(demand_absorption_errors)
            if convert_jacobians:
                xi_jacobian, demand_jacobian_absorption_errors = self._absorb_demand_ids(xi_jacobian)
                errors.extend(demand_jacobian_absorption_errors)
        if self._absorb_supply_ids is not None:
            iv_tilde_costs, supply_absorption_errors = self._absorb_supply_ids(iv_tilde_costs)
            errors.extend(supply_absorption_errors)
            if convert_jacobians:
                omega_jacobian, supply_jacobian_absorption_errors = self._absorb_supply_ids(omega_jacobian)
                errors.extend(supply_jacobian_absorption_errors)

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
        parameters_list, u_list, jacobian_list = iv.estimate(
            X_list, Z_list, W[:self.MD + self.MS, :self.MD + self.MS], y_list, jacobian_list, convert_jacobians
        )
        beta[parameters.eliminated_beta_index] = parameters_list[0].flat
        if self.K3 == 0:
            xi = u_list[0]
            omega = np.full((self.N, 0), np.nan, options.dtype)
            xi_jacobian = jacobian_list[0]
        else:
            gamma[parameters.eliminated_gamma_index] = parameters_list[1].flat
            xi, omega = u_list
            xi_jacobian, omega_jacobian = jacobian_list

            # add any covariance moments
            if self.MC > 0:
                u_list.append(xi * omega - covariance_moments_mean)
                Z_list.append(self.products.ZC)
                jacobian_list.append(xi_jacobian * omega + xi * omega_jacobian)

        # compute the objective value and replace it with its last value if computation failed
        with np.errstate(all='ignore'):
            mean_g = np.r_[compute_gmm_moments_mean(u_list, Z_list), micro]
            objective = mean_g.T @ W @ mean_g
            if scale_objective:
                objective *= self.N
        if not np.isfinite(np.squeeze(objective)):
            objective = progress.objective
            errors.append(exceptions.ObjectiveReversionError())

        # compute the gradient and replace any invalid elements with their last values
        gradient = np.full_like(progress.gradient, np.nan)
        if compute_gradient:
            with np.errstate(all='ignore'):
                mean_G = np.r_[compute_gmm_moments_jacobian_mean(jacobian_list, Z_list), micro_jacobian]
                gradient = 2 * (mean_G.T @ W @ mean_g)
                if scale_objective:
                    gradient *= self.N

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
                    gradient = np.zeros_like(progress.gradient)

        # select the delta that will be used in the next objective evaluation
        next_delta = delta
        if delta_behavior == 'first':
            next_delta = progress.next_delta

        # optionally compute the Hessian with central finite differences
        hessian = np.full_like(progress.hessian, np.nan)
        if compute_hessian:
            def compute_perturbed_gradient(perturbed_theta: Array) -> Array:
                """Evaluate the gradient at a perturbed parameter vector."""
                perturbed_progress = self._compute_progress(
                    parameters, moments, iv, W, scale_objective, error_behavior, error_punishment, delta_behavior,
                    iteration, fp_type, shares_bounds, costs_bounds, finite_differences, covariance_moments_mean,
                    resample_agent_data, perturbed_theta, progress, compute_gradient=True, compute_hessian=False,
                    compute_micro_covariances=False, detect_micro_collinearity=False,
                    compute_simulation_covariances=False,
                )
                return perturbed_progress.gradient

            # compute the Hessian, enforcing shape and symmetry
            hessian = compute_finite_differences(compute_perturbed_gradient, theta)
            hessian = np.c_[hessian + hessian.T] / 2

        # optionally resample agents to compute simulation covariances
        simulation_covariances = np.zeros((mean_g.size, mean_g.size), options.dtype)
        if compute_simulation_covariances:
            assert callable(resample_agent_data)
            mean_g_samples: List[Array] = []
            while True:
                try:
                    resampled_agent_data = resample_agent_data(len(mean_g_samples))
                except Exception as exception:
                    raise RuntimeError("Failed to resample agents because of the above exception.") from exception
                if resampled_agent_data is None:
                    break
                try:
                    resampled_agents = Agents(self.products, self.agent_formulation, resampled_agent_data)
                except Exception as exception:
                    raise RuntimeError("Failed to use resampled agents because of the above exception.") from exception

                resampled_progress = self._compute_progress(
                    parameters, moments, iv, W, scale_objective, error_behavior, error_punishment, delta_behavior,
                    iteration, fp_type, shares_bounds, costs_bounds, finite_differences, covariance_moments_mean,
                    resample_agent_data, theta, progress, compute_gradient=False, compute_hessian=False,
                    compute_micro_covariances=False, detect_micro_collinearity=False,
                    compute_simulation_covariances=False, agents_override=resampled_agents,
                )
                mean_g_samples.append(resampled_progress.mean_g.flatten())

            if len(mean_g_samples) < 2:
                raise ValueError(
                    "There must be at least 2 resampled sets of agent data to estimate the contribution of simulation "
                    "error to moment covariances."
                )

            simulation_covariances = np.cov(mean_g_samples, rowvar=False)

        # structure progress
        return Progress(
            self, parameters, moments, W, theta, objective, gradient, hessian, next_delta, delta, tilde_costs, micro,
            xi_jacobian, omega_jacobian, micro_jacobian, micro_covariances, micro_values, mean_g,
            simulation_covariances, iv_delta, iv_tilde_costs, xi, omega, beta, gamma, iteration_stats, clipped_shares,
            clipped_costs, errors
        )


class Problem(ProblemEconomy):
    r"""A BLP-type problem.

    This class is initialized with relevant data and solved with :meth:`Problem.solve`.

    Parameters
    ----------
    product_formulations : `Formulation or sequence of Formulation`
        :class:`Formulation` configuration or a sequence of up to three :class:`Formulation` configurations for the
        matrix of demand-side linear product characteristics, :math:`X_1`, for the matrix of demand-side nonlinear
        product characteristics, :math:`X_2`, and for the matrix of supply-side characteristics, :math:`X_3`,
        respectively. If the formulation for :math:`X_3` is not specified or is ``None``, a supply side will not be
        estimated. Similarly, if the formulation for :math:`X_2` is not specified or is ``None``, the logit (or nested
        logit) model will be estimated.

        Variable names should correspond to fields in ``product_data``. The ``shares`` variable should not be included
        in the formulations for :math:`X_1` or :math:`X_2`. The formulation for :math:`X_3` can include shares to allow
        marginal costs to depend on quantity.

        The ``prices`` variable should not be included in the formulation for :math:`X_3`, but it should be included in
        the formulation for :math:`X_1` or :math:`X_2` (or both). The ``absorb`` argument of :class:`Formulation` can be
        used to absorb fixed effects into :math:`X_1` and :math:`X_3`, but not :math:`X_2`. Characteristics in
        :math:`X_2` should generally be included in :math:`X_1`. The typical exception is characteristics that are
        collinear with fixed effects that have been absorbed into :math:`X_1`.

        By default, characteristics in :math:`X_1` that do not involve ``prices``, :math:`X_1^\text{ex}`, will be
        combined with excluded demand-side instruments (specified below) to create the full set of demand-side
        instruments, :math:`Z_D`. Any fixed effects absorbed into :math:`X_1` will also be absorbed into :math:`Z_D`.
        Similarly, characteristics in :math:`X_3` that do not involve ``shares``, :math:`X_3^\text{ex}`, will be
        combined with the excluded supply-side instruments to create :math:`Z_S`, and any fixed effects absorbed into
        :math:`X_3` will also be absorbed into :math:`Z_S`. The ``add_exogenous`` flag can be used to disable this
        behavior.

        .. warning::

           Characteristics that involve prices, :math:`p`, or shares, :math:`s`, should always be formulated with the
           ``prices`` and ``shares`` variables, respectively. If another name is used, :class:`Problem` will not
           understand that the characteristic is endogenous, so it will be erroneously included in :math:`Z_D` or
           :math:`Z_S`, and derivatives computed with respect to prices or shares will likely be wrong. For example, to
           include a :math:`p^2` characteristic, include ``I(prices**2)`` in a formula instead of manually constructing
           and including a ``prices_squared`` variable.

    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **shares** : (`numeric`) - Market shares, :math:`s`, which should be between zero and one, exclusive.
              Outside shares should also be between zero and one. Shares in each market should sum to less than one.

            - **prices** : (`numeric`) - Product prices, :math:`p`.

        If a formulation for :math:`X_3` is specified in ``product_formulations``, firm IDs are also required, since
        they will be used to estimate the supply side of the problem:

            - **firm_ids** : (`object, optional`) - IDs that associate products with firms.

        Excluded instruments are typically specified with the following fields:

            - **demand_instruments** : (`numeric`) - Excluded demand-side instruments, which, together with the
              formulated exogenous demand-side linear product characteristics, :math:`X_1^\text{ex}`, constitute the
              full set of  demand-side instruments, :math:`Z_D`. To instead specify the full matrix :math:`Z_D`, set
              ``add_exogenous`` to ``False``.

            - **supply_instruments** : (`numeric, optional`) - Excluded supply-side instruments, which, together with
              the formulated exogenous supply-side characteristics, :math:`X_3^\text{ex}`, constitute the full set of
              supply-side instruments, :math:`Z_S`. To instead specify the full matrix :math:`Z_S`, set
              ``add_exogenous`` to ``False``.

            - **covariance_instruments** : (`numeric, optional`) - Covariance instruments :math:`Z_C`. If specified,
              additional moments :math:`E[g_{C,jt}] = E[\xi_{jt}\omega_{jt}Z_{C,jt}] = 0` will be added, as in
              :ref:`references:MacKay and Miller (2023)`.

              If any fixed effects are absorbed, :math:`\xi_{jt}` and :math:`\omega_{jt}` in these new covariance
              moments are replaced with :math:`\Delta\xi_{jt}` and/or :math:`\Delta\omega_{jt}` in :eq:`fe`. The default
              2SLS weighting matrix will have an additional :math:`(Z_C'Z_C / N)^{-1}` block after the first two.

              .. warning::

                 Covariance restrictions are still an experimental feature. The way in which they are implemented and
                 used may change somewhat in future releases.

              .. note::

                 Using covariance restrictions to identify a parameter on price can sometimes yield two solutions, where
                 the "upper" solution may be positive (i.e., implying upward-sloping demand). See
                 :ref:`references:MacKay and Miller (2023)` for more discussion of this point. In these cases when the
                 "lower" root is the correct solution, consider imposing a one-sided bound (e.g., zero) on the parameter
                 on price to ensure the appropriate sign using ``beta_bounds`` (if the parameter is in ``beta``) or
                 replacing it with a lognormal coefficient on price via the ``rc_type`` argument to :class:`Problem`.

              .. note::

                 In the current implementation, these covariance restrictions only affect the nonlinear parameters. The
                 linear parameters are estimated using other moments. In the case of overidentification, the estimator
                 may not be fully efficient because of this implementation decision.

        The recommendation in :ref:`references:Conlon and Gortmaker (2020)` is to start with differentiation instruments
        of :ref:`references:Gandhi and Houde (2017)`, which can be built with :func:`build_differentiation_instruments`,
        and then compute feasible optimal instruments with :func:`ProblemResults.compute_optimal_instruments` in the
        second stage.

        For guidance on how to construct instruments and add them to product data, refer to the examples in the
        documentation for the :func:`build_blp_instruments` and :func:`build_differentiation_instruments` functions.

        If ``firm_ids`` are specified, custom ownership matrices can be specified as well:

            - **ownership** : (`numeric, optional`) - Custom stacked :math:`J_t \times J_t` ownership or product
              holding matrices, :math:`\mathscr{H}`, for each market :math:`t`, which can be built with
              :func:`build_ownership`. By default, standard ownership matrices are built only when they are needed to
              reduce memory usage. If specified, there should be as many columns as there are products in the market
              with the most products. Rightmost columns in markets with fewer products will be ignored.

        .. note::

           Fields that can have multiple columns (``demand_instruments``, ``supply_instruments``, and ``ownership``) can
           either be matrices or can be broken up into multiple one-dimensional fields with column index suffixes that
           start at zero. For example, if there are three columns of excluded demand-side instruments, a
           ``demand_instruments`` field with three columns can be replaced by three one-dimensional fields:
           ``demand_instruments0``, ``demand_instruments1``, and ``demand_instruments2``.

        To estimate a nested logit or random coefficients nested logit (RCNL) model, nesting groups must be specified:

            - **nesting_ids** (`object, optional`) - IDs that associate products with nesting groups. When these IDs are
              specified, ``rho`` must be specified in :meth:`Problem.solve` as well.

        It may be convenient to define IDs for different products:

            - **product_ids** (`object, optional`) - IDs that identify products within markets. There can be multiple
              columns.

        Finally, clustering groups can be specified to account for within-group correlation while updating the weighting
        matrix and estimating standard errors:

            - **clustering_ids** (`object, optional`) - Cluster group IDs, which will be used if ``W_type`` or
              ``se_type`` in :meth:`Problem.solve` is ``'clustered'``.

        Along with ``market_ids``, ``firm_ids``, ``nesting_ids``, ``product_ids``, ``clustering_ids``, and ``prices``,
        the names of any additional fields can typically be used as variables in ``product_formulations``. However,
        there are a few variable names such as ``'X1'``, which are reserved for use by :class:`Products`.

    agent_formulation : `Formulation, optional`
        :class:`Formulation` configuration for the matrix of observed agent characteristics called demographics,
        :math:`d`, which will only be included in the model if this formulation is specified. Since demographics are
        only used if there are demand-side nonlinear product characteristics, this formulation should only be specified
        if :math:`X_2` is formulated in ``product_formulations``. Variable names should correspond to fields in
        ``agent_data``. See the information under ``agent_data`` for how to give fields for product-specific
        demographics :math:`d_{ijt}`.

    agent_data : `structured array-like, optional`
        Each row corresponds to an agent. Markets can have differing numbers of agents. Since simulated agents are only
        used if there are demand-side nonlinear product characteristics, agent data should only be specified if
        :math:`X_2` is formulated in ``product_formulations``. If agent data are specified, market IDs are required:

            - **market_ids** : (`object`) - IDs that associate agents with markets. The set of distinct IDs should be
              the same as the set in ``product_data``. If ``integration`` is specified, there must be at least as many
              rows in each market as the number of nodes and weights that are built for the market.

        If ``integration`` is not specified, the following fields are required:

            - **weights** : (`numeric, optional`) - Integration weights, :math:`w`, for integration over agent choice
              probabilities.

            - **nodes** : (`numeric, optional`) - Unobserved agent characteristics called integration nodes,
              :math:`\nu`. If there are more than :math:`K_2` columns (the number of demand-side nonlinear product
              characteristics), only the first :math:`K_2` will be retained. If any columns of ``sigma`` in
              :meth:`Problem.solve` are fixed at zero, only the first few columns of these nodes will be used.

        The convenience function :func:`build_integration` can be useful when constructing custom nodes and weights.

        .. note::

           If ``nodes`` has multiple columns, it can be specified as a matrix or broken up into multiple one-dimensional
           fields with column index suffixes that start at zero. For example, if there are three columns of nodes, a
           ``nodes`` field with three columns can be replaced by three one-dimensional fields: ``nodes0``, ``nodes1``,
           and ``nodes2``.

        It may be convenient to define IDs for different agents:

            - **agent_ids** (`object, optional`) - IDs that identify agents within markets. There can be multiple of the
              same ID within a market.

        Along with ``market_ids`` and ``agent_ids``, the names of any additional fields can be typically be used as
        variables in ``agent_formulation``. Exceptions are the names ``'demographics'`` and ``'availability'``, which
        are reserved for use by :class:`Agents`.

        In addition to standard demographic variables :math:`d_{it}`, it is also possible to specify product-specific
        demographics :math:`d_{ijt}`. A typical example is geographic distance of agent :math:`i` from product
        :math:`j`. If ``agent_formulation`` has, for example, ``'distance'``, instead of including a single
        ``'distance'`` field in ``agent_data``, one should instead include ``'distance0'``, ``'distance1'``,
        ``'distance2'`` and so on, where the index corresponds to the order in which products appear within market in
        ``product_data``. For example, ``'distance5'`` should measure the distance of agents to the fifth product within
        the market, as ordered in ``product_data``. The last index should be the number of products in the largest
        market, minus one. For markets with fewer products than this maximum number, latter columns will be ignored.

        Finally, by default each agent :math:`i` in market :math:`t` is faced with the same choice set of product
        :math:`j`, but it is possible to specify agent-specific availability :math:`a_{ijt}` much in the same way that
        product-specific demographics are specified. To do so, the following field can be specified:

            - **availability** : (`numeric, optional`) - Agent-specific product availability, :math:`a`. Choice
              probabilities in :eq:`probabilities` are modified according to

              .. math:: s_{ijt} = \frac{a_{ijt} \exp V_{ijt}}{1 + \sum_{k \in J_t} a_{ijt} \exp V_{ikt}},

              and similarly for the nested logit model and consumer surplus calculations. By default, all
              :math:`a_{ijt} = 1`. To have a product :math:`j` be unavailable to agent :math:`i`, set
              :math:`a_{ijt} = 0`.

              Agent-specific availability is specified in the same way that product-specific demographics are specified.
              In ``agent_data``, one can include ``'availability0'``, ``'availability1'``, ``'availability2'``, and so
              on, where the index corresponds to the order in which products appear within market in ``product_data``.
              The last index should be the number of products in the largest market, minus one. For markets with fewer
              products than this maximum number, latter columns will be ignored.

    integration : `Integration, optional`
        :class:`Integration` configuration for how to build nodes and weights for integration over agent choice
        probabilities, which will replace any ``nodes`` and ``weights`` fields in ``agent_data``. This configuration is
        required if ``nodes`` and ``weights`` in ``agent_data`` are not specified. It should not be specified if
        :math:`X_2` is not formulated in ``product_formulations``.

        If this configuration is specified, :math:`K_2` columns of nodes (the number of demand-side nonlinear product
        characteristics) will be built. However, if ``sigma`` in :meth:`Problem.solve` is left unspecified or
        specified with columns fixed at zero, fewer columns will be used.

    rc_types : `sequence of str, optional`
        Random coefficient types:

            - ``'linear'`` (default) - The random coefficient is as defined in :eq:`mu`. All elliptical distributions
              are supported, including the normal distribution.

            - ``'log'`` - The random coefficient's column in :eq:`mu` is exponentiated before being pre-multiplied by
              :math:`X_2`. It will take on values bounded from below by zero. All log-elliptical distributions are
              supported, including the lognormal distribution.

            - ``'logit'`` - The random coefficient's column in :eq:`mu` is passed through the inverse logit function
              before being pre-multiplied by :math:`X_2`. It will take on values bounded from below by zero and above by
              one.

        The list should have as many strings as there are columns in :math:`X_2`. Each string determines the type of the
        random coefficient on the corresponding product characteristic in :math:`X_2`.

        A typical example of when to use ``'log'`` is to have a lognormal coefficient on prices. Implementing this
        typically involves having an ``I(-prices)`` in the formulation for :math:`X_2`, and instead of including
        ``prices`` in :math:`X_1`, including a ``1`` in the ``agent_formulation``. Then the corresponding coefficient in
        :math:`\Pi` will serve as the mean parameter for the lognormal random coefficient on negative
        prices, :math:`-p_{jt}`.

    epsilon_scale : `float, optional`
        Factor by which the Type I Extreme Value idiosyncratic preference term, :math:`\epsilon_{ijt}`, is scaled. By
        default, :math:`\epsilon_{ijt}` is not scaled. The typical use of this parameter is to approximate the pure
        characteristics model of :ref:`references:Berry and Pakes (2007)` by choosing a value smaller than ``1.0``. As
        this scaling factor approaches zero, the model approaches the pure characteristics model in which there is no
        idiosyncratic preference term.

        In practice, this is implemented by dividing :math:`V_{ijt} = \delta_{jt} + \mu_{ijt}` by the scaling factor
        when solving for the mean utility :math:`\delta_{jt}`. For small scaling factors, this leads to large values
        of :math:`V_{ijt}`, which when exponentiated in the logit expression can lead to overflow issues discussed in
        :ref:`references:Berry and Pakes (2007)`. The safe versions of the contraction mapping discussed in the
        documentation for ``fp_type`` in :meth:`Problem.solve` (which is used by default) eliminate overflow issues at
        the cost of introducing fewer (but still common for a small scaling factor) underflow issues. Throughout the
        contraction mapping, some values of the simulated shares :math:`s_{jt}(\delta, \theta)` can underflow to zero,
        causing the contraction to fail when taking logs. By default, ``shares_bounds`` in :meth:`Problem.solve` bounds
        these simulated shares from below by ``1e-300``, which eliminates these underflow issues at the cost of making
        it more difficult for iteration routines to converge.

        With this in mind, scaling epsilon is not supported for nonlinear contractions, and is also not supported when
        there are nesting groups, since these further complicate the problem. In practice, if the goal is to approximate
        the pure characteristics model, it is a good idea to slowly decrease the scale of epsilon (e.g., starting with
        ``0.5``, trying ``0.1``, etc.) until the contraction begins to fail. To further decrease the scale, there are a
        few things that can help. One is passing a different :class:`Iteration` configuration to ``iteration`` in
        :meth:`Problem.solve`, such as ``'lm'``, which can be robust in this situation. Another is to set
        ``pyblp.options.dtype = np.longdouble`` when on a system that supports extended precision (see
        :mod:`~pyblp.options` for more information about this) and choose a smaller lower bound by configuring
        ``shares_bounds`` in :meth:`Problem.solve`. Ultimately the model will stop being solvable at a certain point,
        and this point will vary by problem, so approximating the pure characteristics model requires some degree of
        experimentation.

    costs_type : `str, optional`
        Functional form of the marginal cost function :math:`\tilde{c} = f(c)` in :eq:`costs`. The following
        specifications are supported:

            - ``'linear'`` (default) - Linear specification: :math:`\tilde{c} = c`.

            - ``'log'`` - Log-linear specification: :math:`\tilde{c} = \log c`.

        This specification is only relevant if :math:`X_3` is formulated.

    add_exogenous : `bool, optional`
        Whether to add characteristics in :math:`X_1` that do not involve prices, :math:`X_1^\text{ex}`, to the
        ``demand_instruments`` field in ``product_data`` (including absorbed fixed effects), and similarly, whether
        to add characteristics in :math:`X_3` that do not involve shares, :math:`X_3^\text{ex}`, to the
        ``supply_instruments`` field. This is by default ``True`` so that only excluded instruments need to be
        specified.

        If this is set to ``False``, ``demand_instruments`` and ``supply_instruments`` should specify the full sets of
        demand- and supply-side instruments, :math:`Z_D` and :math:`Z_S`, and fixed effects should be manually absorbed
        (for example, with the :func:`build_matrix` function). This behavior can be useful, for example, when price is
        not the only endogenous product characteristic over which consumers have preferences. This model could be
        correctly estimated by manually adding the truly exogenous characteristics in :math:`X_1` to :math:`Z_D`.

        .. warning::

           If this flag is set to ``False`` because there are multiple endogenous product characteristics, care should
           be taken when including a supply side or computing optimal instruments. These routines assume that price is
           the only endogenous variable over which consumers have preferences.

    Attributes
    ----------
    product_formulations : `Formulation or sequence of Formulation`
        :class:`Formulation` configurations for :math:`X_1`, :math:`X_2`, and :math:`X_3`, respectively.
    agent_formulation : `Formulation`
        :class:`Formulation` configuration for :math:`d`.
    products : `Products`
        Product data structured as :class:`Products`, which consists of data taken from ``product_data`` along with
        matrices built according to :attr:`Problem.product_formulations`. The :func:`data_to_dict` function can be
        used to convert this into a more usable data type.
    agents : `Agents`
        Agent data structured as :class:`Agents`, which consists of data taken from ``agent_data`` or built by
        ``integration`` along with any demographics built according to :attr:`Problem.agent_formulation`. The
        :func:`data_to_dict` function can be used to convert this into a more usable data type.
    unique_market_ids : `ndarray`
        Unique market IDs in product and agent data.
    unique_firm_ids : `ndarray`
        Unique firm IDs in product data.
    unique_nesting_ids : `ndarray`
        Unique nesting group IDs in product data.
    unique_product_ids : `ndarray`
        Unique product IDs in product data.
    unique_agent_ids : `ndarray`
        Unique agent IDs in agent data.
    rc_types : `list of str`
        Random coefficient types.
    epsilon_scale : `float`
        Factor by which the Type I Extreme Value idiosyncratic preference term, :math:`\epsilon_{ijt}`, is scaled.
    costs_type : `str`
        Functional form of the marginal cost function :math:`\tilde{c} = f(c)`.
    T : `int`
        Number of markets, :math:`T`.
    N : `int`
        Number of products across all markets, :math:`N`.
    F : `int`
        Number of firms across all markets, :math:`F`.
    I : `int`
        Number of agents across all markets, :math:`I`.
    K1 : `int`
        Number of demand-side linear product characteristics, :math:`K_1`.
    K2 : `int`
        Number of demand-side nonlinear product characteristics, :math:`K_2`.
    K3 : `int`
        Number of supply-side product characteristics, :math:`K_3`.
    D : `int`
        Number of demographic variables, :math:`D`.
    MD : `int`
        Number of demand-side instruments, :math:`M_D`, which is typically the number of excluded demand-side
        instruments plus the number of exogenous demand-side linear product characteristics, :math:`K_1^\text{ex}`.
    MS : `int`
        Number of supply-side instruments, :math:`M_S`, which is typically the number of excluded supply-side
        instruments plus the number of exogenous supply-side linear product characteristics, :math:`K_3^\text{ex}`.
    MC : `int`
        Number of covariance instruments, :math:`M_C`.
    ED : `int`
        Number of absorbed dimensions of demand-side fixed effects, :math:`E_D`.
    ES : `int`
        Number of absorbed dimensions of supply-side fixed effects, :math:`E_S`.
    H : `int`
        Number of nesting groups, :math:`H`.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    def __init__(
            self, product_formulations: Union[Formulation, Sequence[Optional[Formulation]]], product_data: Mapping,
            agent_formulation: Optional[Formulation] = None, agent_data: Optional[Mapping] = None,
            integration: Optional[Integration] = None, rc_types: Optional[Sequence[str]] = None,
            epsilon_scale: float = 1.0, costs_type: str = 'linear', add_exogenous: bool = True) -> None:
        """Initialize the underlying economy with product and agent data before absorbing fixed effects."""

        # keep track of long it takes to initialize the problem
        output("Initializing the problem ...")
        start_time = time.time()

        # validate and normalize product formulations
        if isinstance(product_formulations, Formulation):
            product_formulations = [product_formulations]
        elif isinstance(product_formulations, collections.abc.Sequence) and len(product_formulations) <= 3:
            product_formulations = list(product_formulations)
        else:
            raise TypeError("product_formulations must be a Formulation instance or a sequence of up to three of them.")
        product_formulations.extend([None] * (3 - len(product_formulations)))

        # initialize the underlying economy with structured product and agent data
        products = Products(product_formulations, product_data, add_exogenous=add_exogenous)
        agents = Agents(products, agent_formulation, agent_data, integration)
        super().__init__(product_formulations, agent_formulation, products, agents, rc_types, epsilon_scale, costs_type)

        # absorb any demand-side fixed effects
        if self._absorb_demand_ids is not None:
            output("Absorbing demand-side fixed effects ...")
            self.products.X1, X1_errors = self._absorb_demand_ids(self.products.X1)
            self._handle_errors(X1_errors)
            if add_exogenous:
                self.products.ZD, ZD_errors = self._absorb_demand_ids(self.products.ZD)
                self._handle_errors(ZD_errors)

        # absorb any supply-side fixed effects
        if self._absorb_supply_ids is not None:
            output("Absorbing supply-side fixed effects ...")
            self.products.X3, X3_errors = self._absorb_supply_ids(self.products.X3)
            self._handle_errors(X3_errors)
            if add_exogenous:
                self.products.ZS, ZS_errors = self._absorb_supply_ids(self.products.ZS)
                self._handle_errors(ZS_errors)

        # detect any problems with the product data
        self._detect_collinearity(add_exogenous)

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
        super().__init__(
            problem.product_formulations, problem.agent_formulation, updated_products, problem.agents,
            rc_types=problem.rc_types, epsilon_scale=problem.epsilon_scale, costs_type=problem.costs_type
        )

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

        # detect any collinearity issues with the updated instruments
        self._detect_collinearity(added_exogenous=True)

        # output information about the re-created problem
        output(f"Re-created the problem after {format_seconds(time.time() - start_time)}.")
        output("")
        output(self)


class ImportanceSamplingProblem(ProblemEconomy):
    """A BLP problem updated after importance sampling.

    This class can be used exactly like :class:`Problem`.

    """

    def __init__(self, problem: ProblemEconomy, sampled_agents: RecArray) -> None:
        """Initialize the underlying economy with updated agent data."""

        # keep track of long it takes to re-create the problem
        output("Re-creating the problem ...")
        start_time = time.time()

        # initialize the underlying economy with structured product and agent data
        super().__init__(
            problem.product_formulations, problem.agent_formulation, problem.products, sampled_agents,
            rc_types=problem.rc_types, epsilon_scale=problem.epsilon_scale, costs_type=problem.costs_type
        )

        # output information about the re-created problem
        output(f"Re-created the problem after {format_seconds(time.time() - start_time)}.")
        output("")
        output(self)


class InitialProgress(object):
    """Structured information about initial estimation progress."""

    problem: ProblemEconomy
    parameters: Parameters
    moments: Moments
    W: Array
    theta: Array
    objective: Array
    gradient: Array
    hessian: Array
    next_delta: Array
    delta: Array
    tilde_costs: Array
    micro: Array
    xi_jacobian: Array
    omega_jacobian: Array
    micro_jacobian: Array

    def __init__(
            self, problem: ProblemEconomy, parameters: Parameters, moments: Moments, W: Array, theta: Array,
            objective: Array, gradient: Array, hessian: Array, next_delta: Array, delta: Array, tilde_costs: Array,
            micro: Array, xi_jacobian: Array, omega_jacobian: Array, micro_jacobian: Array) -> None:
        """Store initial progress information, computing the projected gradient and the reduced Hessian."""
        self.problem = problem
        self.parameters = parameters
        self.moments = moments
        self.W = W
        self.theta = theta
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.next_delta = next_delta
        self.delta = delta
        self.tilde_costs = tilde_costs
        self.micro = micro
        self.xi_jacobian = xi_jacobian
        self.omega_jacobian = omega_jacobian
        self.micro_jacobian = micro_jacobian


class Progress(InitialProgress):
    """Structured information about estimation progress."""

    micro_covariances: Array
    micro_values: Array
    mean_g: Array
    xi: Array
    omega: Array
    beta: Array
    gamma: Array
    iteration_stats: Dict[Hashable, SolverStats]
    clipped_shares: Array
    clipped_costs: Array
    errors: List[Error]
    projected_gradient: Array
    reduced_hessian: Array
    projected_gradient_norm: Array

    def __init__(
            self, problem: ProblemEconomy, parameters: Parameters, moments: Moments, W: Array, theta: Array,
            objective: Array, gradient: Array, hessian: Array, next_delta: Array, delta: Array, tilde_costs: Array,
            micro: Array, xi_jacobian: Array, omega_jacobian: Array, micro_jacobian: Array, micro_covariances: Array,
            micro_values: Array, mean_g: Array, simulation_covariances: Array, iv_delta: Array, iv_tilde_costs: Array,
            xi: Array, omega: Array, beta: Array, gamma: Array, iteration_stats: Dict[Hashable, SolverStats],
            clipped_shares: Array, clipped_costs: Array, errors: List[Error]) -> None:
        """Store progress information, compute the projected gradient and its norm, and compute the reduced Hessian."""
        super().__init__(
            problem, parameters, moments, W, theta, objective, gradient, hessian, next_delta, delta, tilde_costs, micro,
            xi_jacobian, omega_jacobian, micro_jacobian
        )
        self.micro_covariances = micro_covariances
        self.micro_values = micro_values
        self.mean_g = mean_g
        self.simulation_covariances = simulation_covariances
        self.iv_delta = iv_delta
        self.iv_tilde_costs = iv_tilde_costs
        self.xi = xi
        self.omega = omega
        self.beta = beta
        self.gamma = gamma
        self.iteration_stats = iteration_stats or {}
        self.clipped_shares = clipped_shares
        self.clipped_costs = clipped_costs
        self.errors = errors or []

        # compute the projected gradient and the reduced Hessian
        self.projected_gradient = self.gradient.copy()
        self.reduced_hessian = self.hessian.copy()
        for p, (lb, ub) in enumerate(self.parameters.compress_bounds()):
            if not lb < theta[p] < ub:
                self.reduced_hessian[p] = self.reduced_hessian[:, p] = 0
                with np.errstate(invalid='ignore'):
                    if theta[p] <= lb:
                        self.projected_gradient[p] = min(0, self.gradient[p])
                    elif theta[p] >= ub:
                        self.projected_gradient[p] = max(0, self.gradient[p])

        # compute the norm of the projected gradient
        self.projected_gradient_norm = np.array(np.nan, options.dtype)
        if gradient.size > 0:
            with np.errstate(invalid='ignore'):
                self.projected_gradient_norm = np.abs(self.projected_gradient).max()

    def format(
            self, optimization: Optimization, shares_bounds: Bounds, costs_bounds: Bounds, step: int, iterations: int,
            evaluations: int, progress_time: float, smallest_objective: Array) -> str:
        """Format a universal display of optimization progress as a string. The first iteration will include the
        progress table header. If there are any errors, information about them will be formatted as well, regardless of
        whether or not a universal display is to be used. The smallest_objective is the smallest objective value
        encountered so far during optimization.
        """
        lines: List[str] = []

        # include information about any errors
        if self.errors:
            preamble = (
                "At least one error was encountered. As long as the optimization routine does not get stuck at values "
                "of theta that give rise to errors, this is not necessarily a problem. If the errors persist or seem "
                "to be impacting the optimization results, consider setting an error punishment or following any of "
                "the other suggestions below:"
            )
            lines.extend(["", preamble, str(exceptions.MultipleErrors(self.errors)), ""])

        # only output errors if the solver's display is being used
        if not optimization._universal_display:
            return "\n".join(lines)

        # construct the leftmost part of the table that always shows up
        header = [
            ("GMM", "Step"), ("Computation", "Time"), ("Optimization", "Iterations"), ("Objective", "Evaluations"),
            ("Fixed Point", "Iterations"), ("Contraction", "Evaluations")
        ]
        values = [
            str(step),
            format_seconds(progress_time),
            str(iterations),
            str(evaluations),
            str(sum(s.iterations for s in self.iteration_stats.values())),
            str(sum(s.evaluations for s in self.iteration_stats.values()))
        ]

        # add a count of any clipped shares or marginal costs
        if np.isfinite(shares_bounds).any():
            header.append(("Clipped", "Shares"))
            values.append(str(self.clipped_shares.sum()))
        if np.isfinite(costs_bounds).any():
            header.append(("Clipped", "Costs"))
            values.append(str(self.clipped_costs.sum()))

        # add information about the objective
        header.extend([("Objective", "Value"), ("Objective", "Improvement")])
        values.append(format_number(self.objective))
        improvement = smallest_objective - self.objective
        if np.isfinite(improvement) and improvement > 0:
            values.append(format_number(smallest_objective - self.objective))
        else:
            values.append(" " * len(format_number(improvement)))

        # add information about the gradient
        if optimization._compute_gradient:
            header.append(("Projected", "Gradient Norm") if self.parameters.any_bounds else ("Gradient", "Norm"))
            values.append(format_number(self.projected_gradient_norm))

        # add information about theta
        header.append(("", "Theta"))
        values.append(", ".join(format_number(x) for x in self.theta))

        # add a space and an extra header every 50 evaluations
        include_header = (evaluations - 1) % 50 == 0
        if include_header and evaluations > 1:
            lines.append("")

        # format the table
        lines.append(format_table(header, values, include_border=False, include_header=include_header))
        return "\n".join(lines)
