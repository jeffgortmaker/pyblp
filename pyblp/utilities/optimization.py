"""Optimization routines."""

import os
import sys
import warnings
import itertools
from pathlib import Path

import numpy as np
import scipy.optimize


class Optimization(object):
    """Configuration for solving optimization problems.

    Parameters
    ----------
    method : `str`
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

        Also accepted is a custom callable method with the following form::

            method(objective_function, initial, bounds, **options) -> final

        where `objective_function` is a callable objective function that accepts an array of parameter values and
        returns either the objective value if `compute_gradient` is ``False`` or a tuple of the objective value and its
        gradient if ``True``, `initial` is an array of initial parameter values, `bounds` is a list of ``(min, max)``
        pairs for each element in `initial`, `options` are specified below, and `final` is an array of optimized
        parameter values.

        To simply evaluate a problem's objective at the initial parameter values, the trivial custom method
        ``lambda f, i, b: i`` can be used.

    method_options : `dict, optional`
        Options for the optimization routine.

        For any non-custom `method` other than ``'knitro'``, these options will be passed to `options` in
        :func:`scipy.optimize.minimize`. Depending on the configured level of verbosity, the `display` and `iprint`
        options may be overridden. Refer to the SciPy documentation for information about which options are available
        for each optimization routine.

        If `method` is ``'knitro'``, these options should be
        `Knitro user options <https://www.artelys.com/tools/knitro_doc/3_referenceManual/userOptions.html>`_. The the
        non-standard `knitro_dir` option can also be specified. Depending on the configured level of verbosity, the
        `outlev` option may be overridden. The following options have non-standard default values:

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

    Examples
    --------
    The following code builds a SLSQP configuration with a non-default tolerance:

    .. ipython:: python

       optimization = pyblp.Optimization('slsqp', {'tol': 1e-10})

    Instead of using a non-custom routine, the following code builds a custom method that implements a grid search over
    parameter values between specified bounds:

    .. ipython:: python

       from itertools import product
       def custom_method(objective_function, initial, bounds):
           best_values = initial
           best_objective = np.inf
           for values in product(*(np.linspace(l, u, 10) for l, u in bounds)):
               objective = objective_function(values)
               if objective < best_objective:
                   best_values = values
                   best_objective = objective
           return best_values

    You can then use this custom method to build an optimization configuration:

    .. ipython:: python

       optimization = pyblp.Optimization(custom_method, compute_gradient=False)

    """

    def __init__(self, method, method_options=None, compute_gradient=True):
        """Validate the method and set default options."""
        simple_methods = {
            'nelder-mead': "the Nelder-Mead algorithm",
            'powell': "the modified Powell algorithm",
        }
        unbounded_methods = {
            'cg': "the conjugate gradient algorithm",
            'bfgs': "the BFGS algorithm",
            'newton-cg': "the Newton-CG algorithm"
        }
        bounded_methods = {
            'l-bfgs-b': "the L-BFGS-B algorithm",
            'tnc': "the truncated Newton (TNC) algorithm",
            'slsqp': "Sequential Least SQuares Programming (SLSQP)",
            'knitro': "an installed version of Artleys Knitro"
        }

        # validate the configuration
        methods = dict(itertools.chain(simple_methods.items(), unbounded_methods.items(), bounded_methods.items()))
        if method not in methods and not callable(method):
            raise ValueError(f"method must be one of {list(methods.keys())} or a callable object.")
        if method_options is not None and not isinstance(method_options, dict):
            raise ValueError("method_options must be None or a dict.")
        if method in simple_methods and compute_gradient:
            raise ValueError(f"compute_gradient must be False when method is '{method}'.")
        if method == 'newton-cg' and not compute_gradient:
            raise ValueError(f"compute_gradient must be True when method is '{method}'.")

        # initialize class attributes
        self._method = method
        self._method_options = {} if method_options is None else method_options.copy()
        self._compute_gradient = compute_gradient
        self._gradient_description = "with analytic gradients" if compute_gradient else "without analytic gradients"
        if callable(method):
            self._method_description = "a custom method"
            self._supports_bounds = True
        else:
            self._method_description = methods[method]
            self._supports_bounds = method in bounded_methods

    def __str__(self):
        """Format the configuration as a string."""
        return (
            f"Configured to optimize with {self._method_description}, non-default options {self._method_options}, and "
            f"{self._gradient_description}."
        )

    def _optimize(self, objective_function, start_values, bounds, verbose):
        """Optimize parameters to minimize a scalar objective, either with a custom method, SciPy, or Knitro. Bounds
        are a list of (lb, ub) tuples and the verbose flag indicates whether default optimization output should be
        displayed.
        """

        # ignore bounds for methods that don't support them
        if not self._supports_bounds:
            bounds = None

        # define a wrapper for the objective function that normalizes arrays so they work with all types of routines
        def objective_wrapper(raw_values):
            raw_values = np.asarray(raw_values)
            values = raw_values.reshape(start_values.shape).astype(start_values.dtype)
            if self._compute_gradient:
                objective, gradient = objective_function(values)
                return float(objective), gradient.astype(np.float64).flatten()
            return float(objective_function(values))

        # normalize the starting values and bounds
        raw_start_values = start_values.astype(np.float64).flatten()
        raw_bounds = None if bounds is None else [(float(l), float(u)) for l, u in bounds]

        # optimize using a custom method
        if callable(self._method):
            raw_optimized_values = self._method(objective_wrapper, raw_start_values, raw_bounds, **self._method_options)
        elif self._method != 'knitro':
            # set default SciPy options
            scipy_options = {'disp': verbose}
            if verbose and self._method in {'l-bfgs-b', 'slsqp'}:
                scipy_options['iprint'] = 2
            scipy_options.update(self._method_options)

            # optimize the wrapper
            minimize_kwargs = {
                'method': self._method,
                'jac': self._compute_gradient,
                'options': scipy_options
            }
            if self._supports_bounds:
                minimize_kwargs['bounds'] = raw_bounds
            raw_optimized_values = scipy.optimize.minimize(objective_wrapper, raw_start_values, **minimize_kwargs).x
        else:
            # set default Knitro options
            knitro_options = {
                'outlev': 4 if verbose else 0,
                'algorithm': 1,
                'gradopt': 1 if self._compute_gradient else 2,
                'hessopt': 2,
                'honorbnds': 1
            }
            knitro_options.update(self._method_options)

            # make sure that the Knitro installation is accessible
            try:
                knitro_path = Path(knitro_options.pop('knitro_dir', os.environ.get('KNITRODIR')))
            except TypeError:
                raise EnvironmentError(
                    "Failed to find the KNITRODIR environment variable. Make sure that a supported version of Knitro "
                    "is properly installed or try specifying the knitro_dir optimization option."
                )
            for subdir in ['lib', 'examples/Python']:
                full_path = knitro_path / subdir
                if not full_path.is_dir():
                    raise OSError(
                        f"Failed to find the directory '{full_path}'. Make sure a supported version of Knitro is "
                        f"properly installed and that the KNITRODIR environment variable exists or that knitro_dir "
                        f"optimization option points to the correct Knitro installation directory."
                    )
                sys.path.append(str(full_path))

            # optimize with Knitro
            knitro = Knitro(objective_wrapper, raw_start_values, raw_bounds, knitro_options, self._compute_gradient)
            with knitro:
                raw_optimized_values = knitro.solve()

        # convert the raw optimized values to the same data type and shape as the starting values
        return np.asarray(raw_optimized_values).astype(start_values.dtype).reshape(start_values.shape)


class Knitro(object):
    """A wrapper for Artleys Knitro bindings, which minimize a scalar objective. Used in a with statement.

    Interaction with Knitro is through the knitro module, which has been modified by an import of the knitroNumPy module
    (both modules are located in the Knitro installation directory) as well as through knitro_context, an instance of
    knitro.KTR_context.
    """

    def __init__(self, objective_function, start_values, bounds, knitro_options, compute_gradient):
        """Import the knitro module, modify it to work with numpy, and initialize optimization settings."""
        try:
            import knitro
        except OSError as exception:
            if 'Win32' in repr(exception):
                raise EnvironmentError("Make sure both Knitro and Python are 32- or 64-bit.") from exception
            raise

        # modify Knitro to work with numpy
        import knitroNumPy
        self.knitro = knitro
        self.knitro.KTR_array_handler._cIntArray = knitroNumPy._cIntArray
        self.knitro.KTR_array_handler._cDoubleArray = knitroNumPy._cDoubleArray
        self.knitro.KTR_array_handler._userArray = knitroNumPy._userArray
        self.knitro.KTR_array_handler._userToCArray = knitroNumPy._userToCArray
        self.knitro.KTR_array_handler._cToUserArray = knitroNumPy._cToUserArray

        # wrap the objective with a function that caches its output
        self.last_values = self.last_objective = self.last_gradient = None
        def cache_objective(values):
            self.last_values = values
            if compute_gradient:
                self.last_objective, self.last_gradient = objective_function(values)
            else:
                self.last_objective = objective_function(values)
                self.last_gradient = None

        # initialize other class attributes
        self.cache_objective = cache_objective
        self.start_values = start_values
        self.lower_bounds = np.array([b[0] if np.isfinite(b[0]) else -self.knitro.KTR_INFBOUND for b in bounds])
        self.upper_bounds = np.array([b[1] if np.isfinite(b[1]) else +self.knitro.KTR_INFBOUND for b in bounds])
        self.knitro_options = knitro_options
        self.compute_gradient = compute_gradient
        self.knitro_context = None
        self.last_call = None

    def __enter__(self):
        """Initialize the Knitro context and configure Knitro settings."""
        self.knitro_context = self.knitro.KTR_new()
        if not self.knitro_context:
            raise RuntimeError(
                "Failed to find a Knitro license. Make sure that Knitro is properly installed. You may have to create "
                "the environment variable ARTELYS_LICENSE and set it to the directory holding you license file."
            )

        # set Knitro parameters
        for key, value in self.knitro_options.items():
            set_parameter = self.knitro.KTR_set_param_by_name
            if isinstance(value, str):
                set_parameter = self.knitro.KTR_set_char_param_by_name
            code = set_parameter(self.knitro_context, key, value)
            if code != 0:
                raise RuntimeError(f"Encountered error code {code} setting the Knitro parameter '{key}'.")

        # set Knitro callbacks
        code = self.knitro.KTR_set_func_callback(self.knitro_context, self.objective_callback)
        if code != 0:
            raise RuntimeError(f"Encountered error code {code} registering the Knitro objective callback.")
        code = self.knitro.KTR_set_grad_callback(self.knitro_context, self.gradient_callback)
        if code != 0:
            raise RuntimeError(f"Encountered error code {code} registering the Knitro gradient callback.")

        # initialize the problem while ignoring noisy warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            code = self.knitro.KTR_init_problem(
                kc=self.knitro_context,
                n=self.start_values.size,
                xInitial=self.start_values,
                lambdaInitial=None,
                objGoal=self.knitro.KTR_OBJGOAL_MINIMIZE,
                objType=self.knitro.KTR_OBJTYPE_GENERAL,
                xLoBnds=self.lower_bounds,
                xUpBnds=self.upper_bounds,
                cType=None,
                cLoBnds=None,
                cUpBnds=None,
                jacIndexVars=None,
                jacIndexCons=None,
                hessIndexRows=None,
                hessIndexCols=None
            )
        if code != 0:
            raise RuntimeError(f"Encountered error code {code} initializing the Knitro problem solver.")
        return self

    def __exit__(self, *_):
        """Attempt to free the Knitro process."""
        try:
            self.knitro.KTR_free(self.knitro_context)
        except:
            pass

    def solve(self):
        """Use the initialized and configured Knitro context to solve the optimization problem."""
        values = np.zeros_like(self.start_values)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return_code = self.knitro.KTR_solve(
                kc=self.knitro_context,
                x=values,
                lambda_=np.zeros_like(self.start_values),
                evalStatus=0,
                obj=np.array([0]),
                c=None,
                objGrad=None,
                jac=None,
                hess=None,
                hessVector=None,
                userParams=None
            )

        # Knitro was only successful if its return code was 0 (final solution satisfies the termination conditions for
        #   verifying optimality) or between -100 and -199 (a feasible approximate solution was found)
        if return_code < -199:
            raise RuntimeError(f"Knitro failed to solve the problem. Knitro return code: {return_code}.")
        return values

    def objective_callback(self, *args):
        """Cache a call to the objective function and updates the objective value. A Knitro status code is returned."""
        request_code, values, objective = args[0], args[5], args[7]
        if request_code != self.knitro.KTR_RC_EVALFC:
            return self.knitro.KTR_RC_CALLBACK_ERR
        self.cache_objective(values)
        objective[:] = self.last_objective
        return self.knitro.KTR_RC_BEGINEND

    def gradient_callback(self, *args):
        """Update the gradient. If parameter values are different from the ones last passed to the objective function,
        cache a call to the objective function with the new parameter values. A Knitro status code is returned.
        """
        request_code, values, gradient = args[0], args[5], args[9]
        if request_code != self.knitro.KTR_RC_EVALGA:
            return self.knitro.KTR_RC_CALLBACK_ERR
        if self.compute_gradient:
            if self.last_gradient is None or not np.array_equal(values, self.last_values):
                self.cache_objective(values)
            gradient[:] = self.last_gradient
        return self.knitro.KTR_RC_BEGINEND
