"""Optimization routines."""

import contextlib
import functools
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import scipy.optimize

from .. import options
from ..utilities.basics import Array, Options, SolverStats, StringRepresentation, format_options


# objective function types
ObjectiveResults = Tuple[float, Optional[Array]]
ObjectiveFunction = Callable[[Array], ObjectiveResults]


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

            objective_function(theta) -> (objective, gradient)

        where ``gradient`` is ``None`` if ``compute_gradient is ``False``.

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
            objective, gradient = verbose_objective_function(values, iterations, evaluations)
            return (
                float(objective),
                None if gradient is None else gradient.astype(raw_values.dtype, copy=False).flatten()
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


def return_optimizer(initial_values: Array, *_: Any, **__: Any) -> Tuple[Array, bool]:
    """Assume the initial values are the optimal ones."""
    success = True
    return initial_values, success


def scipy_optimizer(
        initial_values: Array, bounds: Optional[Iterable[Tuple[float, float]]], objective_function: ObjectiveFunction,
        iteration_callback: Callable[[], None], method: str, compute_gradient: bool, **scipy_options: Any) -> (
        Tuple[Array, bool]):
    """Optimize with a SciPy method."""
    cache: Optional[Tuple[Array, ObjectiveResults]] = None

    def objective_wrapper(values: Array) -> float:
        """Return a possibly cached objective value."""
        nonlocal cache
        if cache is None or not np.array_equal(values, cache[0]):
            cache = (values.copy(), objective_function(values))
        return cache[1][0]

    def gradient_wrapper(values: Array) -> Array:
        """Return a possibly cached gradient."""
        nonlocal cache
        if cache is None or not np.array_equal(values, cache[0]):
            cache = (values.copy(), objective_function(values))
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
        cache: Optional[Tuple[Array, ObjectiveResults]] = None

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
                cache = (values.copy(), objective_function(values))
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
                obj=np.array([0]), c=None, objGrad=None, jac=None, hess=None, hessVector=None, userParams=None
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
