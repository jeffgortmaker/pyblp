"""Optimization routines."""

import os
import sys
import warnings
import functools
from pathlib import Path

import numpy as np

from .. import options


class Optimization(object):
    """Configuration for solving optimization problems.

    Parameters
    ----------
    method : `str or callable`
        The optimization routine that will be used. The following routines support parameter bounds and use analytic
        gradients:

            - ``'knitro'`` - Uses an installed version of
              `Artleys Knitro <https://www.artelys.com/en/optimization-tools/knitro>`_. Python 3 is supported by Knitro
              version 10.3 and newer. A number of environment variables most likely need to be configured properly, such
              as ``KNITRODIR``, ``ARTELYS_LICENSE``, ``LD_LIBRARY_PATH`` (on Linux), and ``DYLD_LIBRARY_PATH`` (on Mac
              OS X). For more information, refer to the
              `Knitro installation guide <https://www.artelys.com/tools/knitro_doc/1_introduction/installation.html>`_.

            - ``'l-bfgs-b'`` - Uses the :func:`scipy.optimize.minimize` L-BFGS-B routine.

            - ``'slsqp'`` - Uses the :func:`scipy.optimize.minimize` SLSQP routine.

            - ``'tnc'`` - Uses the :func:`scipy.optimize.minimize` TNC routine.

        The following routines also use analytic gradients but will ignore parameter bounds:

            - ``'cg'`` - Uses the :func:`scipy.optimize.minimize` CG routine.

            - ``'bfgs'`` - Uses the :func:`scipy.optimize.minimize` BFGS routine.

            - ``'newton-cg'`` - Uses the :func:`scipy.optimize.minimize` Newton-CG routine.

        The following routines do not use analytic gradients and will also ignore parameter bounds:

            - ``'nelder-mead'`` - Uses the :func:`scipy.optimize.minimize` Nelder-Mead routine.

            - ``'powell'`` - Uses the :func:`scipy.optimize.minimize` Powell routine.

        The following trivial routine can be used to evaluate an objective at specific parameter values:

            - ``'return'`` - Assume that the initial parameter values are the optimal ones.

        Also accepted is a custom callable method with the following form::

            method(initial, bounds, objective_function, iteration_callback, **options) -> (final, converged)

        where `initial` is an array of initial parameter values, `bounds` is a list of ``(min, max)`` pairs for each
        element in `initial`, `objective_function` is a callable objective function that accepts an array of parameter
        values and returns either the objective value if `compute_gradient` is ``False`` or a tuple of the objective
        value and its gradient if ``True``, `iteration_callback` is a function that should be called without any
        arguments after each major iteration (it is used to record the number of major iterations), `options` are
        specified below, `final` is an array of optimized parameter values, and `converged` is a flag for whether the
        routine converged.

        To simply evaluate a problem's objective at the initial parameter values, the trivial custom method
        ``lambda x, *_: (x, True)`` can be used.

    method_options : `dict, optional`
        Options for the optimization routine.

        For any non-custom `method` other than ``'knitro'`` and ``'return'``, these options will be passed to `options`
        in :func:`scipy.optimize.minimize`. Refer to the SciPy documentation for information about which options are
        available for each optimization routine.

        If `method` is ``'knitro'``, these options should be
        `Knitro user options <https://www.artelys.com/tools/knitro_doc/3_referenceManual/userOptions.html>`_. The
        non-standard `knitro_dir` option can also be specified. The following options have non-standard default values:

            - **knitro_dir** : (`str`) - By default, the KNITRODIR environment variable is used. Otherwise, this
              option should point to the installation directory of Knitro, which contains direct subdirectories such as
              ``'examples'`` and ``'lib'``. For example, on Windows this option could be
              ``'/Program Files/Artleys3/Knitro 10.3.0'``.

            - **algorithm** : (`int`) - The optimization algorithm to be used. The default value is ``1``, which
              corresponds to the Interior/Direct algorithm.

            - **gradopt** : (`int`) - How the objective's gradient is computed. The default value is ``1`` if
              `compute_gradient` is ``True`` and is ``2`` otherwise, which corresponds to estimating the gradient with
              finite differences.

            - **hessopt** : (`int`) - How the objective's Hessian is computed. The default value is ``2``, which
              corresponds to computing a quasi-Newton BFGS Hessian.

            - **honorbnds** : (`int`) - Whether to enforce satisfaction of simple variable bounds. The default value is
              ``1``, which corresponds to enforcing that the initial point and all subsequent solution estimates satisfy
              the bounds.

    compute_gradient : `bool, optional`
        Whether to compute an analytic objective gradient during optimization, which must be ``False`` if `method` does
        not use analytic gradients, and must be ``True`` if `method` is ``'newton-cg'``, which requires an analytic
        gradient. By default, analytic gradients are computed. If ``False``, an analytic gradient may still be computed
        once at the end of optimization to compute optimization results.
    universal_display : `bool, optional`
        Whether to format optimization progress such that the display looks the same for all routines. By default, the
        universal display is used and some `method_options` are used to prevent default displays from showing up.

    Examples
    --------
    In this example, we'll build a SLSQP configuration with a non-default tolerance.

    .. ipython:: python

       optimization = pyblp.Optimization('slsqp', {'tol': 1e-10})
       optimization

    Next, instead of using a non-custom routine, we'll create a custom method that implements a grid search over
    parameter values between specified bounds.

    .. ipython:: python

       from itertools import product
       def custom_method(initial, bounds, objective_function, iteration_callback):
           best_values = initial
           best_objective = np.inf
           for values in product(*(np.linspace(l, u, 10) for l, u in bounds)):
               objective = objective_function(values)
               if objective < best_objective:
                   best_values = values
                   best_objective = objective
               iteration_callback()
           return best_values, True

    We can then use this custom method to build an optimization configuration.

    .. ipython:: python

       optimization = pyblp.Optimization(custom_method, compute_gradient=False)
       optimization

    For more examples, refer to the :doc:`Examples </examples>` section.

    """

    def __init__(self, method, method_options=None, compute_gradient=True, universal_display=True):
        """Validate the method and set default options."""
        simple_methods = {
            'nelder-mead': (scipy_optimizer, "the Nelder-Mead algorithm implemented in SciPy"),
            'powell': (scipy_optimizer, "the modified Powell algorithm implemented in SciPy")
        }
        unbounded_methods = {
            'cg': (scipy_optimizer, "the conjugate gradient algorithm implemented in SciPy"),
            'bfgs': (scipy_optimizer, "the BFGS algorithm implemented in SciPy"),
            'newton-cg': (scipy_optimizer, "the Newton-CG algorithm implemented in SciPy")
        }
        bounded_methods = {
            'l-bfgs-b': (scipy_optimizer, "the L-BFGS-B algorithm implemented in SciPy"),
            'tnc': (scipy_optimizer, "the truncated Newton algorithm implemented in SciPy"),
            'slsqp': (scipy_optimizer, "Sequential Least SQuares Programming implemented in SciPy"),
            'knitro': (knitro_optimizer, "an installed version of Artleys Knitro"),
            'return': (return_optimizer, "a trivial routine that returns the initial parameter values")
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

        # replace missing options with a dict
        method_options = method_options or {}

        # options are simply passed along to custom methods
        if callable(method):
            self._optimizer = method
            self._method_options = method_options
            self._description = "a custom method"
            return

        # identify the non-custom optimizer, configure arguments, and set default options
        self._method_options = {}
        unwrapped_optimizer, self._description = methods[method]
        self._optimizer = functools.partial(unwrapped_optimizer, compute_gradient=compute_gradient)
        if unwrapped_optimizer == knitro_optimizer:
            self._method_options.update({
                'hessopt': 2,
                'algorithm': 1,
                'honorbnds': 1,
                'gradopt': 1 if compute_gradient else 2,
                'knitro_dir': os.environ.get('KNITRODIR'),
                'outlev': 4 if not universal_display and options.verbose else 0
            })
        elif unwrapped_optimizer == scipy_optimizer:
            self._optimizer = functools.partial(self._optimizer, method=method)
            if not universal_display and options.verbose:
                self._method_options['disp'] = True
                if method in {'l-bfgs-b', 'slsqp'}:
                    self._method_options['iprint'] = 2
        else:
            assert unwrapped_optimizer == return_optimizer

        # validate options for non-custom methods
        self._method_options.update(method_options)
        if unwrapped_optimizer == knitro_optimizer:
            knitro_dir = self._method_options.pop('knitro_dir')
            if not isinstance(knitro_dir, (Path, str)):
                raise ValueError(
                    "If specified, the knitro_dir optimization option must point to the Knitro installation directory."
                    "Otherwise, the KNITRODIR environment variable must be configured."
                )
            for subdir in ['lib', 'examples/Python']:
                full_path = Path(knitro_dir) / subdir
                if not full_path.is_dir():
                    raise OSError(
                        f"Failed to find the directory '{full_path}'. Make sure a supported version of Knitro is "
                        f"properly installed and that the KNITRODIR environment variable exists. Alternatively, the "
                        f"knitro_dir optimization option should point to the Knitro installation directory."
                    )
                sys.path.append(str(full_path))

    def __str__(self):
        """Format the configuration as a string."""
        gradients = "with analytic gradients" if self._compute_gradient else "without analytic gradients"
        return f"Configured to optimize using {self._description} with options {self._method_options} and {gradients}."

    def __repr__(self):
        """Defer to the string representation."""
        return str(self)

    def _optimize(self, initial_values, bounds, verbose_objective_function):
        """Optimize parameters to minimize a scalar objective."""

        # define a callback that counts the number of major iterations
        def iteration_callback():
            iteration_callback.iterations += 1

        # define a wrapper for the objective function that normalizes arrays so they work with all types of routines,
        #   and also counts the number of evaluations
        def objective_wrapper(raw_values):
            objective_wrapper.evaluations += 1
            raw_values = np.asanyarray(raw_values)
            values = raw_values.reshape(initial_values.shape).astype(initial_values.dtype, copy=False)
            returned = verbose_objective_function(values, iteration_callback.iterations, objective_wrapper.evaluations)
            if self._compute_gradient:
                objective, gradient = returned
                return float(objective), gradient.astype(raw_values.dtype, copy=False).flatten()
            return float(returned)

        # initialize the counters and normalize values
        iteration_callback.iterations = objective_wrapper.evaluations = 0
        raw_initial_values = initial_values.astype(np.float64, copy=False).flatten()
        raw_bounds = None if bounds is None or not self._supports_bounds else [(float(l), float(u)) for l, u in bounds]

        # solve the problem and convert the raw final values to the same data type and shape as the initial values
        raw_final_values, converged = self._optimizer(
            raw_initial_values, raw_bounds, objective_wrapper, iteration_callback, **self._method_options
        )
        final_values = np.asanyarray(raw_final_values).astype(initial_values.dtype).reshape(initial_values.shape)
        return final_values, converged, iteration_callback.iterations, objective_wrapper.evaluations


def return_optimizer(initial_values, *args, **kwargs):
    """Assume the initial values are the optimal ones."""
    del args
    del kwargs
    success = True
    return initial_values, success


def scipy_optimizer(initial_values, bounds, objective_function, iteration_callback, method, compute_gradient,
                    **scipy_options):
    """Optimize with a SciPy method."""
    import scipy.optimize
    results = scipy.optimize.minimize(
        objective_function, initial_values, method=method, jac=compute_gradient, bounds=bounds,
        callback=lambda *_: iteration_callback(), options=scipy_options
    )
    return results.x, results.success


def knitro_optimizer(initial_values, bounds, objective_function, iteration_callback, compute_gradient,
                     **knitro_options):
    """Optimize with Knitro."""
    try:
        import knitro
    except OSError as exception:
        if 'Win32' in repr(exception):
            raise EnvironmentError("Make sure both Knitro and Python are 32- or 64-bit.") from exception
        raise

    # modify Knitro to work with numpy
    import knitroNumPy
    knitro.KTR_array_handler._cIntArray = knitroNumPy._cIntArray
    knitro.KTR_array_handler._cDoubleArray = knitroNumPy._cDoubleArray
    knitro.KTR_array_handler._userArray = knitroNumPy._userArray
    knitro.KTR_array_handler._userToCArray = knitroNumPy._userToCArray
    knitro.KTR_array_handler._cToUserArray = knitroNumPy._cToUserArray

    # create the Knitro context and attempt to free it if anything goes wrong
    knitro_context = None
    try:
        knitro_context = knitro.KTR_new()
        if not knitro_context:
            raise RuntimeError(
                "Failed to find a Knitro license. Make sure that Knitro is properly installed. You may have to create "
                "the environment variable ARTELYS_LICENSE and set it to the location of the directory with the license "
                "file."
            )

        # define a function that handles requests to compute either the objective or its gradient (which are cached for
        #   when the next request is for the same values) and calls the iteration callback when there's a new iteration
        def combined_callback(*args):
            request_code, values, objective_store, gradient_store = (args[i] for i in [0, 5, 7, 9])

            # call the iteration callback if this is a new iteration
            iterations = knitro.KTR_get_number_iters(knitro_context)
            while combined_callback.iterations < iterations:
                iteration_callback()
                combined_callback.iterations += 1

            # compute the objective or used cached values
            if combined_callback.cache is not None and np.array_equal(values, combined_callback.cache[0]):
                combined = combined_callback.cache[1]
            else:
                combined = objective_function(values)
                combined_callback.cache = (values.copy(), combined)

            # extract the objective and its gradient
            objective, gradient = combined if compute_gradient else (combined, None)

            # handle request codes
            if request_code == knitro.KTR_RC_EVALFC:
                objective_store[:] = objective
                return knitro.KTR_RC_BEGINEND
            if request_code == knitro.KTR_RC_EVALGA:
                gradient_store[:] = gradient
                return knitro.KTR_RC_BEGINEND
            return knitro.KTR_RC_CALLBACK_ERR

        # initialize an empty cache and the counter
        combined_callback.cache = None
        combined_callback.iterations = 0

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
        bounds = bounds or [None] * initial_values.size
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
    finally:
        try:
            knitro.KTR_free(knitro_context)
        except:
            pass
