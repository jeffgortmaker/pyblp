"""Optimization routines."""

import contextlib
import functools
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import scipy.optimize

from .. import options
from ..utilities.basics import Array, Bounds, Options, SolverStats, StringRepresentation, format_options


# objective function types
ObjectiveResults = Tuple[float, Optional[Array], Optional['OptimizationProgress']]
ObjectiveFunction = Callable[[Array], ObjectiveResults]

# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import Progress, ProblemEconomy  # noqa
    from ..results.problem_results import ProblemResults  # noqa


class Optimization(StringRepresentation):
    r"""Configuration for solving optimization problems.

    Parameters
    ----------
    method : `str or callable`
        The optimization routine that will be used. The following routines support parameter bounds and use analytic
        gradients:

            - ``'knitro'`` - Uses an installed version of
              `Artleys Knitro <https://www.artelys.com/solvers/knitro/>`_. Python 3 is supported by Knitro version 10.3
              and newer. A number of environment variables most likely need to be configured properly, such as
              ``KNITRODIR``, ``ARTELYS_LICENSE``, ``LD_LIBRARY_PATH`` (on Linux), and ``DYLD_LIBRARY_PATH`` (on
              Mac OS X). For more information, refer to the
              `Knitro installation guide <https://www.artelys.com/docs/knitro//1_introduction/installation.html>`_.

            - ``'slsqp'`` - Uses the :func:`scipy.optimize.minimize` SLSQP routine.

            - ``'trust-constr'`` - Uses the :func:`scipy.optimize.minimize` trust-region routine.

            - ``'l-bfgs-b'`` - Uses the :func:`scipy.optimize.minimize` L-BFGS-B routine.

            - ``'tnc'`` - Uses the :func:`scipy.optimize.minimize` TNC routine.

        The following routines also use analytic gradients but will ignore parameter bounds (not bounding the problem
        may create issues if the optimizer tries out large parameter values that create overflow errors):

            - ``'cg'`` - Uses the :func:`scipy.optimize.minimize` CG routine.

            - ``'bfgs'`` - Uses the :func:`scipy.optimize.minimize` BFGS routine.

            - ``'newton-cg'`` - Uses the :func:`scipy.optimize.minimize` Newton-CG routine.

        The following routines do not use analytic gradients and will also ignore parameter bounds (without analytic
        gradients, optimization will likely be much slower):

            - ``'nelder-mead'`` - Uses the :func:`scipy.optimize.minimize` Nelder-Mead routine.

            - ``'powell'`` - Uses the :func:`scipy.optimize.minimize` Powell routine.

        The following trivial routine can be used to evaluate an objective at specific parameter values:

            - ``'return'`` - Assume that the initial parameter values are the optimal ones.

        Also accepted is a custom callable method with the following form::

            method(initial, bounds, objective_function, iteration_callback, **options) -> (final, converged)

        where ``initial`` is an array of initial parameter values, ``bounds`` is a list of ``(min, max)`` pairs for each
        element in ``initial``, ``objective_function`` is a callable objective function of the form specified below,
        ``iteration_callback`` is a function that should be called without any arguments after each major iteration (it
        is used to record the number of major iterations), ``options`` are specified below, ``final`` is an array of
        optimized parameter values, and ``converged`` is a flag for whether the routine converged.

        The ``objective_function`` has the following form:

            objective_function(theta) -> (objective, gradient, progress)

        where ``gradient`` is ``None`` if ``compute_gradient is ``False`` and ``progress`` is an
        :class:`OptimizationProgress` object that contains additional information about optimization progress so far,
        which may be helpful for debugging or to inform non-standard optimization routines.

    method_options : `dict, optional`
        Options for the optimization routine.

        For any non-custom ``method`` other than ``'knitro'`` and ``'return'``, these options will be passed to
        ``options`` in :func:`scipy.optimize.minimize`, with the exception of ``'keep_feasible'``, which is by default
        ``True`` and is passed to any ``scipy.optimize.Bounds``. Refer to the SciPy documentation for information about
        which options are available for each optimization routine.

        If ``method`` is ``'knitro'``, these options should be
        `Knitro user options <https://www.artelys.com/docs/knitro//3_referenceManual/userOptions.html>`_. The
        non-standard ``knitro_dir`` option can also be specified. The following options have non-standard default
        values:

            - **knitro_dir** : (`str`) - By default, the KNITRODIR environment variable is used. Otherwise, this
              option should point to the installation directory of Knitro, which contains direct subdirectories such as
              ``'examples'`` and ``'lib'``. For example, on Windows this option could be
              ``'/Program Files/Artleys3/Knitro 10.3.0'``.

            - **algorithm** : (`int`) - The optimization algorithm to be used. The default value is ``1``, which
              corresponds to the Interior/Direct algorithm.

            - **gradopt** : (`int`) - How the objective's gradient is computed. The default value is ``1`` if
              ``compute_gradient`` is ``True`` and is ``2`` otherwise, which corresponds to estimating the gradient with
              finite differences.

            - **hessopt** : (`int`) - How the objective's Hessian is computed. The default value is ``2``, which
              corresponds to computing a quasi-Newton BFGS Hessian.

            - **honorbnds** : (`int`) - Whether to enforce satisfaction of simple variable bounds. The default value is
              ``1``, which corresponds to enforcing that the initial point and all subsequent solution estimates satisfy
              the bounds.

    compute_gradient : `bool, optional`
        Whether to compute an analytic objective gradient during optimization, which must be ``False`` if ``method``
        does not use analytic gradients, and must be ``True`` if ``method`` is ``'newton-cg'``, which requires an
        analytic gradient.

        By default, analytic gradients are computed. Not using an analytic gradient will likely slow
        down estimation a good deal. If ``False``, an analytic gradient may still be computed once at the end of
        optimization to compute optimization results. To always use finite differences, ``finite_differences`` in
        :meth:`Problem.solve` can be set to ``True``.

    universal_display : `bool, optional`
        Whether to format optimization progress such that the display looks the same for all routines. By default, the
        universal display is used and some ``method_options`` are used to prevent default displays from showing up.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/optimization.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    _optimizer: functools.partial
    _description: str
    _method_options: Options
    _supports_bounds: bool
    _compute_gradient: bool
    _universal_display: bool

    def __init__(
            self, method: Union[str, Callable], method_options: Optional[Options] = None, compute_gradient: bool = True,
            universal_display: bool = True) -> None:
        """Validate the method and set default options."""
        simple_methods = {
            'nelder-mead': (functools.partial(scipy_optimizer), "the Nelder-Mead algorithm implemented in SciPy"),
            'powell': (functools.partial(scipy_optimizer), "the modified Powell algorithm implemented in SciPy")
        }
        unbounded_methods = {
            'cg': (functools.partial(scipy_optimizer), "the conjugate gradient algorithm implemented in SciPy"),
            'bfgs': (functools.partial(scipy_optimizer), "the BFGS algorithm implemented in SciPy"),
            'newton-cg': (functools.partial(scipy_optimizer), "the Newton-CG algorithm implemented in SciPy")
        }
        bounded_methods = {
            'l-bfgs-b': (functools.partial(scipy_optimizer), "the L-BFGS-B algorithm implemented in SciPy"),
            'tnc': (functools.partial(scipy_optimizer), "the truncated Newton algorithm implemented in SciPy"),
            'slsqp': (functools.partial(scipy_optimizer), "Sequential Least SQuares Programming implemented in SciPy"),
            'trust-constr': (functools.partial(scipy_optimizer), "trust-region routine implemented in SciPy"),
            'knitro': (functools.partial(knitro_optimizer), "an installed version of Artleys Knitro"),
            'return': (functools.partial(return_optimizer), "a trivial routine that returns the initial parameters")
        }
        methods = {**simple_methods, **unbounded_methods, **bounded_methods}

        # validate the configuration
        if method not in methods and not callable(method):
            raise ValueError(f"method must be one of {list(methods)} or a callable object.")
        if method_options is not None and not isinstance(method_options, dict):
            raise ValueError("method_options must be None or a dict.")
        if method in simple_methods and compute_gradient:
            raise ValueError(f"compute_gradient must be False when method is '{method}'.")
        if method == 'newton-cg' and not compute_gradient:
            raise ValueError(f"compute_gradient must be True when method is '{method}'.")

        # initialize class attributes
        self._compute_gradient = compute_gradient
        self._universal_display = universal_display
        self._supports_bounds = callable(method) or method in bounded_methods

        # options are by default empty
        if method_options is None:
            method_options = {}

        # options are simply passed along to custom methods
        if callable(method):
            self._optimizer = functools.partial(method)
            self._description = "a custom method"
            self._method_options = method_options
            return

        # identify the non-custom optimizer, configure arguments, and set default options
        self._method_options: Options = {}
        self._optimizer, self._description = methods[method]
        self._optimizer = functools.partial(self._optimizer, compute_gradient=compute_gradient)
        if method == 'knitro':
            self._method_options.update({
                'hessopt': 2,
                'algorithm': 1,
                'honorbnds': 1,
                'gradopt': 1 if compute_gradient else 2,
                'knitro_dir': os.environ.get('KNITRODIR'),
                'outlev': 4 if not universal_display and options.verbose else 0
            })
        elif method != 'return':
            self._optimizer = functools.partial(self._optimizer, method=method)
            if not universal_display and options.verbose:
                self._method_options['disp'] = True
                if method in {'l-bfgs-b', 'slsqp'}:
                    self._method_options['iprint'] = 2
                elif method == 'trust-constr':
                    self._method_options['verbose'] = 3

        # update the default options
        self._method_options.update(method_options)

        # validate options for non-SciPy routines
        if method == 'return' and self._method_options:
            raise ValueError("The return method does not support any options.")
        if method == 'knitro':
            # get the location of the Knitro installation
            knitro_dir = self._method_options.pop('knitro_dir')
            if not isinstance(knitro_dir, (Path, str)):
                raise OSError(
                    "If specified, the knitro_dir optimization option must point to the Knitro installation directory."
                    "Otherwise, the KNITRODIR environment variable must be configured."
                )

            # add relevant paths
            for subdir in ['lib', 'examples/Python']:
                full_path = Path(knitro_dir) / subdir
                if not full_path.is_dir():
                    raise OSError(
                        f"Failed to find the directory '{full_path}'. Make sure a supported version of Knitro is "
                        f"properly installed and that the KNITRODIR environment variable exists. Alternatively, the "
                        f"knitro_dir optimization option should point to the Knitro installation directory."
                    )
                sys.path.append(str(full_path))

            # make sure that Knitro can be initialized
            with knitro_context_manager():
                pass

    def __str__(self) -> str:
        """Format the configuration as a string."""
        description = f"{self._description} {'with' if self._compute_gradient else 'without'} analytic gradients"
        return f"Configured to optimize using {description} and options {format_options(self._method_options)}."

    def _optimize(
            self, initial: Array, bounds: Optional[Iterable[Tuple[float, float]]],
            verbose_objective_function: Callable[[Array, int, int], ObjectiveResults]) -> Tuple[Array, SolverStats]:
        """Optimize parameters to minimize a scalar objective."""

        # initialize counters
        iterations = evaluations = 0

        def iteration_callback() -> None:
            """Count the number of major iterations."""
            nonlocal iterations
            iterations += 1

        def objective_wrapper(raw_values: Any) -> ObjectiveResults:
            """Normalize arrays so they work with all types of routines. Also count the total number of contraction
            evaluations.
            """
            nonlocal evaluations
            evaluations += 1
            raw_values = np.asanyarray(raw_values)
            values = raw_values.reshape(initial.shape).astype(initial.dtype, copy=False)
            objective, gradient, progress = verbose_objective_function(values, iterations, evaluations)
            return (
                float(objective),
                None if gradient is None else gradient.astype(raw_values.dtype, copy=False).flatten(),
                progress,
            )

        # normalize values
        raw_initial = initial.astype(np.float64, copy=False).flatten()
        raw_bounds = None if bounds is None or not self._supports_bounds else [(float(l), float(u)) for l, u in bounds]

        # solve the problem and convert the raw final values to the same data type and shape as the initial values
        raw_final, converged = self._optimizer(
            raw_initial, raw_bounds, objective_wrapper, iteration_callback, **self._method_options
        )
        final = np.asanyarray(raw_final).astype(initial.dtype, copy=False).reshape(initial.shape)
        stats = SolverStats(converged, iterations, evaluations)
        return final, stats


class OptimizationProgress(object):
    r"""Information about the current progress of optimization.

    They key attributes of this class needed to define a custom optimization routine are ``objective`` and ``gradient``.
    Many other attributes of :class:`ProblemResults` are also included, and can be used to help define an alternative
    optimization routine (e.g., Gauss-Newton), to debug issues, or to add custom ad-hoc moments to the configured
    problem.

    Attributes
    ----------
    problem : `Problem`
        :class:`Problem` that created this progress.
    fp_converged : `ndarray`
        Flags for convergence of the iteration routine used to compute :math:`\delta(\theta)` in each market. Values are
        in the same order as :attr:`Problem.unique_market_ids`.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute :math:`\delta(\theta)` in each
        market. Values are in the same order as :attr:`Problem.unique_market_ids`.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute :math:`\delta(\theta)` was evaluated in each market. Values
        are in the same order as :attr:`Problem.unique_market_ids`.
    theta : `ndarray`
        Unfixed parameters, :math:`\theta`, in the following order: :math:`\Sigma`, :math:`\Pi`, :math:`\rho`,
        non-concentrated out elements from :math:`\beta`, and non-concentrated out elements from :math:`\gamma`.
    sigma : `ndarray`
        Cholesky root of the covariance matrix for unobserved taste heterogeneity, :math:`\Sigma`.
    sigma_squared : `ndarray`
        Covariance matrix for unobserved taste heterogeneity, :math:`\Sigma\Sigma'`.
    pi : `ndarray`
        Parameters that measures how agent tastes vary with demographics, :math:`\Pi`.
    rho : `ndarray`
        Parameters that measure within nesting group correlations, :math:`\rho`.
    beta : `ndarray`
        Demand-side linear parameters, :math:`\beta`.
    gamma : `ndarray`
        Supply-side linear parameters, :math:`\gamma`.
    sigma_bounds : `tuple`
        Bounds for :math:`\Sigma` that were used during optimization, which are of the form ``(lb, ub)``.
    pi_bounds : `tuple`
        Bounds for :math:`\Pi` that were used during optimization, which are of the form ``(lb, ub)``.
    rho_bounds : `tuple`
        Bounds for :math:`\rho` that were used during optimization, which are of the form ``(lb, ub)``.
    beta_bounds : `tuple`
        Bounds for :math:`\beta` that were used during optimization, which are of the form ``(lb, ub)``.
    gamma_bounds : `tuple`
        Bounds for :math:`\gamma` that were used during optimization, which are of the form ``(lb, ub)``.
    sigma_labels : `list of str`
        Variable labels for rows and columns of :math:`\Sigma`, which are derived from the formulation for :math:`X_2`.
    pi_labels : `list of str`
        Variable labels for columns of :math:`\Pi`, which are derived from the formulation for demographics.
    rho_labels : `list of str`
        Variable labels for :math:`\rho`. If :math:`\rho` is not a scalar, this is :attr:`Problem.unique_nesting_ids`.
    beta_labels : `list of str`
        Variable labels for :math:`\beta`, which are derived from the formulation for :math:`X_1`.
    gamma_labels : `list of str`
        Variable labels for :math:`\gamma`, which are derived from the formulation for :math:`X_3`.
    theta_labels : `list of str`
        Variable labels for :math:`\theta`, which are derived from the above labels.
    delta : `ndarray`
        Mean utility, :math:`\delta(\theta)`.
    clipped_shares : `ndarray`
        Vector of booleans indicating whether the associated simulated shares were clipped during the last fixed point
        iteration to compute :math:`\delta(\theta)`. All elements will be ``False`` if ``shares_bounds`` in
        :meth:`Problem.solve` is disabled (by default shares are bounded from below by a small number to alleviate
        issues with underflow and negative shares).
    tilde_costs : `ndarray`
        Estimated transformed marginal costs, :math:`\tilde{c}(\theta)` from :eq:`costs`. If ``costs_bounds`` were
        specified in :meth:`Problem.solve`, :math:`c` may have been clipped.
    clipped_costs : `ndarray`
        Vector of booleans indicating whether the associated marginal costs were clipped. All elements will be ``False``
        if ``costs_bounds`` in :meth:`Problem.solve` was not specified.
    xi : `ndarray`
        Unobserved demand-side product characteristics, :math:`\xi(\theta)`, or equivalently, the demand-side structural
        error term. When there are demand-side fixed effects, this is :math:`\Delta\xi(\theta)` in :eq:`fe`. That is,
        fixed effects are not included.
    omega : `ndarray`
        Unobserved supply-side product characteristics, :math:`\omega(\theta)`, or equivalently, the supply-side
        structural error term. When there are supply-side fixed effects, this is :math:`\Delta\omega(\theta)` in
        :eq:`fe`. That is, fixed effects are not included.
    micro : `ndarray`
        Micro moments, :math:`\bar{g}_M`, in :eq:`micro_moment`.
    micro_values : `ndarray`
        Micro moment values, :math:`f_m(v)`. Rows are in the same order as :attr:`ProblemResults.micro`.
    objective : `float`
        GMM objective value, :math:`q(\theta)`, defined in :eq:`objective`. If ``scale_objective`` was ``True`` in
        :meth:`Problem.solve` (which is the default), this value was scaled by :math:`N` so that objective values are
        more comparable across different problem sizes. Note that in some of the BLP literature (and earlier versions of
        this package), this expression was previously scaled by :math:`N^2`.
    xi_by_theta_jacobian : `ndarray`
        :math:`\frac{\partial\xi}{\partial\theta} = \frac{\partial\delta}{\partial\theta}`.
    omega_by_theta_jacobian : `ndarray`
        :math:`\frac{\partial\omega}{\partial\theta} = \frac{\partial\tilde{c}}{\partial\theta}`.
    micro_by_theta_jacobian : `ndarray`
        :math:`\frac{\partial\bar{g}_M}{\partial\theta}`.
    gradient : `ndarray`
        Gradient of the GMM objective, :math:`\nabla q(\theta)`, defined in :eq:`gradient`.
    projected_gradient : `ndarray`
        Projected gradient of the GMM objective. When there are no parameter bounds, this will always be equal to
        :attr:`ProblemResults.gradient`. Otherwise, if an element in :math:`\hat{\theta}` is equal to its lower (upper)
        bound, the corresponding projected gradient value will be truncated at a maximum (minimum) of zero.
    projected_gradient_norm : `ndarray`
        Infinity norm of :attr:`ProblemResults.projected_gradient`.
    W : `ndarray`
        Weighting matrix, :math:`W`, used to compute these results.

    """

    problem: 'ProblemEconomy'
    fp_converged: Array
    fp_iterations: Array
    contraction_evaluations: Array
    theta: Array
    sigma: Array
    sigma_squared: Array
    pi: Array
    rho: Array
    beta: Array
    gamma: Array
    sigma_bounds: Bounds
    pi_bounds: Bounds
    rho_bounds: Bounds
    beta_bounds: Bounds
    gamma_bounds: Bounds
    sigma_labels: List[str]
    pi_labels: List[str]
    rho_labels: List[str]
    beta_labels: List[str]
    gamma_labels: List[str]
    theta_labels: List[str]
    delta: Array
    clipped_shares: Array
    tilde_costs: Array
    clipped_costs: Array
    xi: Array
    omega: Array
    micro: Array
    micro_values: Array
    objective: Array
    xi_by_theta_jacobian: Array
    omega_by_theta_jacobian: Array
    micro_by_theta_jacobian: Array
    gradient: Array
    projected_gradient: Array
    projected_gradient_norm: Array
    W: Array

    def __init__(self, progress: 'Progress'):
        """Structure all the progress."""
        self.sigma, self.pi, self.rho, _, _ = progress.parameters.expand(progress.theta)
        self.problem = progress.problem
        self.W = progress.W
        self.theta = progress.theta
        self.delta = progress.delta
        self.tilde_costs = progress.tilde_costs
        self.micro = progress.micro
        self.micro_values = progress.micro_values
        self.xi_by_theta_jacobian = progress.xi_jacobian
        self.omega_by_theta_jacobian = progress.omega_jacobian
        self.micro_by_theta_jacobian = progress.micro_jacobian
        self.xi = progress.xi
        self.omega = progress.omega
        self.beta = progress.beta
        self.gamma = progress.gamma
        self.objective = progress.objective
        self.gradient = progress.gradient
        self.projected_gradient = progress.projected_gradient
        self.projected_gradient_norm = progress.projected_gradient_norm
        self.clipped_shares = progress.clipped_shares
        self.clipped_costs = progress.clipped_costs

        # structure fixed point information
        stats = progress.iteration_stats
        self.fp_converged = np.array(
            [stats[t].converged if stats else True for t in self.problem.unique_market_ids], dtype=np.int64
        )
        self.fp_iterations = np.array(
            [stats[t].iterations if stats else 0 for t in self.problem.unique_market_ids], dtype=np.int64
        )
        self.contraction_evaluations = np.array(
            [stats[t].evaluations if stats else 0 for t in self.problem.unique_market_ids], dtype=np.int64
        )

        # store information about parameters
        self.sigma_squared = self.sigma @ self.sigma.T
        self.sigma_bounds = progress.parameters.sigma_bounds
        self.pi_bounds = progress.parameters.pi_bounds
        self.rho_bounds = progress.parameters.rho_bounds
        self.beta_bounds = progress.parameters.beta_bounds
        self.gamma_bounds = progress.parameters.gamma_bounds
        self.sigma_labels = progress.parameters.sigma_labels
        self.pi_labels = progress.parameters.pi_labels
        self.rho_labels = progress.parameters.rho_labels
        self.beta_labels = progress.parameters.beta_labels
        self.gamma_labels = progress.parameters.gamma_labels
        self.theta_labels = progress.parameters.theta_labels


def return_optimizer(initial_values: Array, *_: Any, **__: Any) -> Tuple[Array, bool]:
    """Assume the initial values are the optimal ones."""
    success = True
    return initial_values, success


def scipy_optimizer(
        initial_values: Array, bounds: Optional[Iterable[Tuple[float, float]]], objective_function: ObjectiveFunction,
        iteration_callback: Callable[[], None], method: str, compute_gradient: bool, **scipy_options: Any) -> (
        Tuple[Array, bool]):
    """Optimize with a SciPy method."""
    cache: Optional[Tuple[Array, Tuple[float, Optional[Array]]]] = None

    def objective_wrapper(values: Array) -> float:
        """Return a possibly cached objective value."""
        nonlocal cache
        if cache is None or not np.array_equal(values, cache[0]):
            cache = (values.copy(), objective_function(values)[:2])
        return cache[1][0]

    def gradient_wrapper(values: Array) -> Array:
        """Return a possibly cached gradient."""
        nonlocal cache
        if cache is None or not np.array_equal(values, cache[0]):
            cache = (values.copy(), objective_function(values)[:2])
        return cache[1][1]

    # by default use the BFGS approximation for the Hessian
    hess = scipy_options.get('hess', scipy.optimize.BFGS() if method == 'trust-constr' else None)

    # extract and configure any bound feasibility
    if 'keep_feasible' in scipy_options:
        if bounds is not None:
            lb, ub = zip(*bounds)
            bounds = scipy.optimize.Bounds(lb, ub, scipy_options['keep_feasible'])
        scipy_options = scipy_options.copy()
        del scipy_options['keep_feasible']

    # call the SciPy function
    callback = lambda *_: iteration_callback()
    results = scipy.optimize.minimize(
        objective_wrapper, initial_values, method=method, jac=gradient_wrapper if compute_gradient else False,
        hess=hess, bounds=bounds, callback=callback, options=scipy_options
    )
    return results.x, results.success


def knitro_optimizer(
        initial_values: Array, bounds: Optional[Iterable[Tuple[float, float]]], objective_function: ObjectiveFunction,
        iteration_callback: Callable[[], None], compute_gradient: bool, **knitro_options: Any) -> Tuple[Array, bool]:
    """Optimize with Knitro."""
    with knitro_context_manager() as (knitro, knitro_context):
        iterations = 0
        cache: Optional[Tuple[Array, Tuple[float, Optional[Array]]]] = None

        def combined_callback(
                request_code: int, _: Any, __: Any, ___: Any, ____: Any, values: Array, _____: Any,
                objective_store: Array, ______: Any, gradient_store: Array, *_______: Any) -> int:
            """Handle requests to compute either the objective or its gradient (which are cached for when the next
            request is for the same values) and call the iteration callback when there's a new major iteration.
            """
            nonlocal iterations, cache

            # call the iteration callback if this is a new iteration
            current_iterations = knitro.KTR_get_number_iters(knitro_context)
            while iterations < current_iterations:
                iteration_callback()
                iterations += 1

            # compute the objective or used cached values
            if cache is None or not np.array_equal(values, cache[0]):
                cache = (values.copy(), objective_function(values)[:2])
            objective, gradient = cache[1]

            # define a function that normalizes values so they can be digested by Knitro
            normalize = lambda x: min(max(float(x), -sys.maxsize), sys.maxsize)

            # handle request codes
            if request_code == knitro.KTR_RC_EVALFC:
                objective_store[0] = normalize(objective)
                return knitro.KTR_RC_BEGINEND
            if request_code == knitro.KTR_RC_EVALGA:
                assert compute_gradient and gradient is not None
                for index, gradient_value in enumerate(gradient.flatten()):
                    gradient_store[index] = normalize(gradient_value)
                return knitro.KTR_RC_BEGINEND
            return knitro.KTR_RC_CALLBACK_ERR

        # configure Knitro callbacks
        callback_mapping = {
            knitro.KTR_set_func_callback: combined_callback,
            knitro.KTR_set_grad_callback: combined_callback
        }
        for set_callback, callback in callback_mapping.items():
            code = set_callback(knitro_context, callback)
            if code != 0:
                raise RuntimeError(f"Encountered error code {code} when registering {set_callback.__name__}.")

        # configure Knitro parameters
        for key, value in knitro_options.items():
            set_parameter = knitro.KTR_set_param_by_name
            if isinstance(value, str):
                set_parameter = knitro.KTR_set_char_param_by_name
            code = set_parameter(knitro_context, key, value)
            if code != 0:
                raise RuntimeError(f"Encountered error code {code} when configuring '{key}'.")

        # initialize the problem
        bounds = bounds or [(-np.inf, +np.inf)] * initial_values.size
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            code = knitro.KTR_init_problem(
                kc=knitro_context,
                n=initial_values.size,
                xInitial=initial_values,
                lambdaInitial=None,
                objGoal=knitro.KTR_OBJGOAL_MINIMIZE,
                objType=knitro.KTR_OBJTYPE_GENERAL,
                xLoBnds=np.array([b[0] if np.isfinite(b[0]) else -knitro.KTR_INFBOUND for b in bounds]),
                xUpBnds=np.array([b[1] if np.isfinite(b[1]) else +knitro.KTR_INFBOUND for b in bounds]),
                cType=None,
                cLoBnds=None,
                cUpBnds=None,
                jacIndexVars=None,
                jacIndexCons=None,
                hessIndexRows=None,
                hessIndexCols=None
            )
        if code != 0:
            raise RuntimeError(f"Encountered error code {code} when initializing the Knitro problem solver.")

        # solve the problem
        values_store = np.zeros_like(initial_values)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return_code = knitro.KTR_solve(
                kc=knitro_context, x=values_store, lambda_=np.zeros_like(initial_values), evalStatus=0,
                obj=np.array([0], np.float64), c=None, objGrad=None, jac=None, hess=None, hessVector=None,
                userParams=None
            )

        # Knitro was only successful if its return code was 0 (final solution satisfies the termination conditions for
        #   verifying optimality) or between -100 and -199 (a feasible approximate solution was found)
        return values_store, return_code > -200


@contextlib.contextmanager
def knitro_context_manager() -> Iterator[Tuple[Any, Any]]:
    """Import Knitro and initialize its context."""
    try:
        import knitro
    except OSError as exception:
        if 'Win32' in repr(exception):
            raise EnvironmentError("Make sure both Knitro and Python are 32- or 64-bit.") from exception
        raise

    # modify older version of Knitro to work with NumPy
    try:
        # noinspection PyUnresolvedReferences
        import knitroNumPy
        knitro.KTR_array_handler._cIntArray = knitroNumPy._cIntArray
        knitro.KTR_array_handler._cDoubleArray = knitroNumPy._cDoubleArray
        knitro.KTR_array_handler._userArray = knitroNumPy._userArray
        knitro.KTR_array_handler._userToCArray = knitroNumPy._userToCArray
        knitro.KTR_array_handler._cToUserArray = knitroNumPy._cToUserArray
    except ImportError:
        pass

    # create the Knitro context and attempt to free it if anything goes wrong
    knitro_context = None
    try:
        knitro_context = None
        try:
            knitro_context = knitro.KTR_new()
        except RuntimeError as exception:
            if 'Error while initializing parameter' not in str(exception):
                raise
        if not knitro_context:
            raise OSError(
                "Failed to find a Knitro license. Make sure that Knitro is properly installed. You may have to create "
                "the environment variable ARTELYS_LICENSE and set it to the location of the directory with the license "
                "file."
            )
        yield knitro, knitro_context
    finally:
        try:
            knitro.KTR_free(knitro_context)
        except Exception:
            pass
