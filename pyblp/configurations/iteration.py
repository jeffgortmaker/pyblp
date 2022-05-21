"""Fixed-point iteration routines."""

import functools
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import scipy.optimize

from ..utilities.basics import Array, Options, SolverStats, StringRepresentation, format_options


# define contraction function types
ContractionResults = Tuple[Array, Optional[Array], Optional[Array]]
ContractionFunction = Callable[[Array, int, int], ContractionResults]
ContractionWrapper = Callable[[Array], ContractionResults]


class Iteration(StringRepresentation):
    r"""Configuration for solving fixed point problems.

    Parameters
    ----------
    method : `str or callable`
        The fixed point iteration routine that will be used. The following routines do not use analytic Jacobians:

            - ``'simple'`` - Non-accelerated iteration.

            - ``'squarem'`` - SQUAREM acceleration method of :ref:`references:Varadhan and Roland (2008)` and considered
              in the context of the BLP problem in :ref:`references:Reynaerts, Varadhan, and Nash (2012)`. This
              implementation uses a first-order squared non-monotone extrapolation scheme.

            - ``'broyden1'`` - Use the :func:`scipy.optimize.root` Broyden's first Jacobian approximation method, known
              as Broyden's good method.

            - ``'broyden2'`` - Use the :func:`scipy.optimize.root` Broyden's second Jacobian approximation method, known
              as Broyden's bad method.

            - ``'anderson'`` - Use the :func:`scipy.optimize.root` Anderson method.

            - ``'krylov'`` - Use the :func:`scipy.optimize.root` Krylov approximation for inverse Jacobian method.

            - ``'diagbroyden'`` - Use the :func:`scipy.optimize.root` diagonal Broyden Jacobian approximation method.

            - ``'df-sane'`` - Use the :func:`scipy.optimize.root` derivative-free spectral method.

        The following routines can use analytic Jacobians:

            - ``'hybr'`` - Use the :func:`scipy.optimize.root` modification of the Powell hybrid method implemented in
              MINIPACK.

            - ``'lm'`` - Uses the :func:`scipy.optimize.root` modification of the Levenberg-Marquardt algorithm
              implemented in MINIPACK.

        The following trivial routine can be used to simply return the initial values:

            - ``'return'`` - Assume that the initial values are the optimal ones.

        Also accepted is a custom callable method with the following form::

            method(initial, contraction, callback, **options) -> (final, converged)

        where ``initial`` is an array of initial values, ``contraction`` is a callable contraction mapping of the form
        specified below, ``callback`` is a function that should be called without any arguments after each major
        iteration (it is used to record the number of major iterations), ``options`` are specified below, ``final`` is
        an array of final values, and ``converged`` is a flag for whether the routine converged.

        The ``contraction`` function has the following form:

            contraction(x0) -> (x1, weights, jacobian)

        where ``weights`` are either ``None`` or a vector of weights that should multiply ``x1 - x`` before computing
        the norm of the differences, and ``jacobian`` is ``None`` if ``compute_jacobian`` is ``False``.

        Regardless of the chosen routine, if there are any computational issues that create infinities or null values,
        ``final`` will be the second to last iteration's values.

    method_options : `dict, optional`
        Options for the fixed point iteration routine.

        For routines other and ``'simple'``, ``'squarem'``, and ``'return'``, these options will be passed to
        ``options`` in :func:`scipy.optimize.root`. Refer to the SciPy documentation for information about which options
        are available. By default, the ``tol_norm`` option is configured to use the infinity norm for SciPy methods
        other than ``'hybr'`` and ``'lm'``, for which a norm cannot be specified.

        The ``'simple'`` and ``'squarem'`` methods support the following options:

            - **max_evaluations** : (`int`) - Maximum number of contraction mapping evaluations. The default value is
              ``5000``.

            - **atol** : (`float`) - Absolute tolerance for convergence of the configured norm. The default value is
              ``1e-14``. To use only a relative tolerance, set this to zero.

            - **rtol** (`float`) - Relative tolerance for convergence of the configured norm. The default value is zero;
              that is, only absolute tolerance is used by default.

            - **norm** : (`callable`) - The norm to be used. By default, the :math:`\ell^\infty`-norm is used. If
              specified, this should be a function that accepts an array of differences and that returns a scalar norm.

        The ``'squarem'`` routine accepts additional options that mirror those in the
        `SQUAREM <https://cran.r-project.org/web/packages/SQUAREM/index.html>`_ package, written in R by Ravi Varadhan,
        which identifies the step length with :math:`-\alpha` from :ref:`references:Varadhan and Roland (2008)`:

            - **scheme** : (`int`) - The default value is ``3``, which corresponds to S3 in
              :ref:`references:Varadhan and Roland (2008)`. Other acceptable schemes are ``1`` and ``2``, which
              correspond to S1 and S2.

            - **step_min** : (`float`) - The initial value for the minimum step length. The default value is ``1.0``.

            - **step_max** : (`float`) - The initial value for the maximum step length. The default value is ``1.0``.

            - **step_factor** : (`float`) - When the step length exceeds ``step_max``, it is set equal to ``step_max``,
              but ``step_max`` is scaled by this factor. Similarly, if ``step_min`` is negative and the step length is
              below ``step_min``, it is set equal to ``step_min`` and ``step_min`` is scaled by this factor. The default
              value is ``4.0``.

    compute_jacobian : `bool, optional`
        Whether to compute an analytic Jacobian during iteration. By default, analytic Jacobians are not computed, and
        if a ``method`` is selected that supports analytic Jacobians, they will by default be numerically approximated.
    universal_display : `bool, optional`
        Whether to format iteration progress such that the display looks the same for all routines. By default, the
        universal display is not used and no iteration progress is displayed. Setting this to ``True`` can be helpful
        for debugging iteration issues. For example, iteration may get stuck above the configured termination tolerance.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/iteration.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    _iterator: functools.partial
    _description: str
    _method_options: Options
    _compute_jacobian: bool
    _universal_display: bool

    def __init__(self, method: Union[str, Callable], method_options: Optional[Options] = None,
                 compute_jacobian: bool = False, universal_display: bool = False) -> None:
        """Validate the method and configure default options."""
        simple_methods = {
            'simple': (functools.partial(simple_iterator), "no acceleration"),
            'squarem': (functools.partial(squarem_iterator), "the SQUAREM acceleration method"),
            'broyden1': (functools.partial(scipy_iterator), "Broyden's good method implemented in SciPy"),
            'broyden2': (functools.partial(scipy_iterator), "Broyden's bad method implemented in SciPy"),
            'anderson': (functools.partial(scipy_iterator), "Anderson's method implemented in SciPy"),
            'diagbroyden': (functools.partial(scipy_iterator), "Broyden's diagonal method implemented in SciPy"),
            'krylov': (functools.partial(scipy_iterator), "Krylov method implemented in SciPy"),
            'df-sane': (functools.partial(scipy_iterator), "the derivative-free spectral method implemented in SciPy"),
        }
        complex_methods = {
            'hybr': (
                functools.partial(scipy_iterator),
                "modification of the Powell hybrid method implemented in MINIPACK via SciPy"
            ),
            'lm': (
                functools.partial(scipy_iterator),
                "modification of the Levenberg-Marquardt algorithm implemented in MINIPACK via SciPy"
            ),
            'return': (functools.partial(return_iterator), "a trivial routine that returns the initial values")
        }
        methods = {**simple_methods, **complex_methods}

        # validate the configuration
        if method not in methods and not callable(method):
            raise ValueError(f"method must be one of {list(methods.keys())} or a callable object.")
        if method_options is not None and not isinstance(method_options, dict):
            raise ValueError("method_options must be None or a dict.")
        if method in simple_methods and compute_jacobian:
            raise ValueError(f"compute_jacobian must be False when method is '{method}'.")

        # initialize class attributes
        self._compute_jacobian = compute_jacobian
        self._universal_display = universal_display

        # options are by default empty
        if method_options is None:
            method_options = {}

        # options are simply passed along to custom methods
        if callable(method):
            self._iterator = functools.partial(method)
            self._description = "a custom method"
            self._method_options = method_options
            return

        # identify the non-custom iterator and set default options
        self._method_options: Options = {}
        self._iterator, self._description = methods[method]
        if method in {'simple', 'squarem'}:
            self._method_options.update({
                'atol': 1e-14,
                'rtol': 0,
                'max_evaluations': 5000,
                'norm': infinity_norm
            })
            if method == 'squarem':
                self._method_options.update({
                    'scheme': 3,
                    'step_min': 1.0,
                    'step_max': 1.0,
                    'step_factor': 4.0
                })
        elif method != 'return':
            self._iterator = functools.partial(self._iterator, method=method, compute_jacobian=compute_jacobian)
            if method in {'broyden1', 'broyden2', 'anderson', 'diagbroyden', 'krylov', 'df-sane'}:
                self._method_options['fnorm' if method == 'df-sane' else 'tol_norm'] = infinity_norm

        # update the default options
        self._method_options.update(method_options)

        # validate options for non-SciPy routines
        if method == 'return' and self._method_options:
            raise ValueError("The return method does not support any options.")
        if method in {'simple', 'squarem'}:
            if not isinstance(self._method_options['atol'], (float, int)) or self._method_options['atol'] < 0:
                raise ValueError("The iteration option atol must be a nonnegative float.")
            if not isinstance(self._method_options['rtol'], (float, int)) or self._method_options['rtol'] < 0:
                raise ValueError("The iteration option rtol must be a nonnegative float.")
            if self._method_options['atol'] == self._method_options['rtol'] == 0:
                raise ValueError("atol and rtol cannot both be zero.")
            if not isinstance(self._method_options['max_evaluations'], int):
                raise ValueError("The iteration option max_evaluations must be an int.")
            if self._method_options['max_evaluations'] < 1:
                raise ValueError("The iteration option max_evaluations must be a positive int.")
            if not callable(self._method_options['norm']):
                raise ValueError("The iteration option norm must be callable.")
        if method == 'squarem':
            if self._method_options['scheme'] not in {1, 2, 3}:
                raise ValueError("The iteration option scheme must be 1, 2, or 3.")
            if not isinstance(self._method_options['step_min'], float):
                raise ValueError("The iteration option step_min must be a float.")
            if not isinstance(self._method_options['step_max'], float) or self._method_options['step_max'] <= 0:
                raise ValueError("The iteration option step_max must be a positive float.")
            if self._method_options['step_min'] > self._method_options['step_max']:
                raise ValueError("The iteration option step_min must be smaller than step_max.")
            if not isinstance(self._method_options['step_factor'], float) or self._method_options['step_factor'] <= 0:
                raise ValueError("The iteration option step_factor must be a positive float.")

    def __str__(self) -> str:
        """Format the configuration as a string."""
        description = f"{self._description} {'with' if self._compute_jacobian else 'without'} analytic Jacobians"
        return f"Configured to iterate using {description} with options {format_options(self._method_options)}."

    def _iterate(self, initial: Array, contraction: ContractionFunction) -> Tuple[Array, SolverStats]:
        """Solve a fixed point iteration problem."""

        # initialize counters
        iterations = evaluations = 0

        def iteration_callback() -> None:
            """Count the number of major iterations."""
            nonlocal iterations
            iterations += 1

        def contraction_wrapper(raw_values: Any) -> ContractionResults:
            """Normalize arrays so they work with all types of routines. Also count the total number of contraction
            evaluations.
            """
            nonlocal evaluations
            evaluations += 1
            if not isinstance(raw_values, np.ndarray):
                raw_values = np.asarray(raw_values)
            values = raw_values.reshape(initial.shape).astype(initial.dtype, copy=False)
            values, weights, jacobian = contraction(values, iterations, evaluations)
            return (
                values.astype(raw_values.dtype, copy=False).reshape(raw_values.shape),
                None if weights is None else weights.astype(raw_values.dtype, copy=False).reshape(raw_values.shape),
                None if jacobian is None else jacobian.astype(raw_values.dtype, copy=False)
            )

        # normalize the starting values
        raw_initial = initial.astype(np.float64, copy=False).flatten()

        # solve the problem and convert the raw final values to the same data type and shape as the initial values
        raw_final, converged = self._iterator(
            raw_initial, contraction_wrapper, iteration_callback, **self._method_options
        )
        final = np.asarray(raw_final).astype(initial.dtype, copy=False).reshape(initial.shape)
        stats = SolverStats(converged, iterations, evaluations)
        return final, stats


def infinity_norm(x: Array) -> float:
    """Compute the infinity norm of a vector."""
    return np.abs(x).max()


def return_iterator(initial: Array, *_: Any, **__: Any) -> Tuple[Array, bool]:
    """Assume the initial values are the optimal ones."""
    success = True
    return initial, success


def scipy_iterator(
        initial: Array, contraction: ContractionWrapper, iteration_callback: Callable[[], None], method: str,
        compute_jacobian: bool, **scipy_options: Any) -> Tuple[Array, bool]:
    """Apply a SciPy root finding method."""

    # define method-specific options so iteration callbacks and norm weighting works properly
    weights_cache = np.ones_like(initial)
    scipy_options = scipy_options.copy()
    if method in {'hybr', 'lm'}:
        callback = None
        scipy_options['diag'] = weights_cache
    else:
        callback = lambda *_: iteration_callback()
        norm_key = 'fnorm' if method == 'df-sane' else 'tol_norm'
        norm = scipy_options.get(norm_key, infinity_norm)
        scipy_options[norm_key] = lambda x: norm(weights_cache * x)

    # record whether non-finite values were encountered during fixed point iteration
    failed = False

    def contraction_wrapper(x: Array) -> Union[Tuple[Array, Array], Array]:
        """Transform the fixed point into a root-finding problem, check for errors, and call the callback function here
        if calling it isn't supported by the routine.
        """
        nonlocal failed, weights_cache

        # attempt to evaluate the contraction and check for bad values
        x0, (x, weights, jacobian) = x, contraction(x)
        if not all_finite(x, weights, jacobian):
            x = x0
            weights = None
            if jacobian is not None:
                jacobian = np.zeros_like(jacobian)
            failed = True

        # record the completion of an iteration if the method doesn't support iteration callbacks
        if callback is None:
            iteration_callback()

        # update the weights
        if weights is not None:
            weights_cache[:] = weights

        # transform the fixed point into a root-finding problem
        if jacobian is None:
            return x0 - x
        return x0 - x, np.eye(x.size) - jacobian

    # call the routine
    results = scipy.optimize.root(
        contraction_wrapper, initial, method=method, jac=compute_jacobian or None, callback=callback,
        options=scipy_options
    )
    return results.x, not failed and results.success


def simple_iterator(
        initial: Array, contraction: ContractionWrapper, iteration_callback: Callable[[], None], max_evaluations: int,
        atol: float, rtol: float, norm: Callable[[Array], float]) -> Tuple[Array, bool]:
    """Apply simple fixed point iteration with no acceleration."""
    x = initial
    failed = False
    evaluations = 0
    while True:
        # contraction step
        x0, (x, weights) = x, contraction(x)[:2]
        if not all_finite(x, weights):
            x = x0
            failed = True
            break

        # record the completion of a major iteration, which is the same here as a contraction evaluation
        iteration_callback()

        # check for convergence
        evaluations += 1
        if evaluations >= max_evaluations or termination_check(x, x - x0, weights, atol, rtol, norm):
            break

    # determine whether there was convergence
    converged = not failed and evaluations < max_evaluations
    return x, converged


def squarem_iterator(
        initial: Array, contraction: ContractionWrapper, iteration_callback: Callable[[], None], max_evaluations: int,
        atol: float, rtol: float, norm: Callable[[Array], float], scheme: int, step_min: float, step_max: float,
        step_factor: float) -> Tuple[Array, bool]:
    """Apply the SQUAREM acceleration method for fixed point iteration."""
    x = initial
    failed = False
    evaluations = 0
    while True:
        # first step
        x0, (x, weights) = x, contraction(x)[:2]
        if not all_finite(x, weights):
            x = x0
            failed = True
            break

        # check for convergence
        g0 = x - x0
        evaluations += 1
        if evaluations >= max_evaluations or termination_check(x, g0, weights, atol, rtol, norm):
            break

        # second step
        x1, (x, weights) = x, contraction(x)[:2]
        if not all_finite(x, weights):
            x = x1
            failed = True
            break

        # check for convergence
        g1 = x - x1
        evaluations += 1
        if evaluations >= max_evaluations or termination_check(x, g1, weights, atol, rtol, norm):
            break

        # compute the step length
        r = g0
        v = g1 - g0
        with np.errstate(divide='ignore'):
            if scheme == 1:
                alpha = (r.T @ v) / (v.T @ v)
            elif scheme == 2:
                alpha = (r.T @ r) / (r.T @ v)
            else:
                alpha = -np.sqrt((r.T @ r) / (v.T @ v))

        # bound the step length and update its bounds
        alpha = -np.maximum(step_min, np.minimum(step_max, -alpha))
        if -alpha == step_max:
            step_max *= step_factor
        if -alpha == step_min and step_min < 0:
            step_min *= step_factor

        # acceleration step
        x2, x = x, x0 - 2 * alpha * r + alpha**2 * v
        x3, (x, weights) = x, contraction(x)[:2]
        if not all_finite(x, weights):
            x = x2
            failed = True
            break

        # record the completion of a major iteration
        iteration_callback()

        # check for convergence
        evaluations += 1
        if evaluations >= max_evaluations or termination_check(x, x - x3, weights, atol, rtol, norm):
            break

    # determine whether there was convergence
    converged = not failed and evaluations < max_evaluations
    return x, converged


def all_finite(*arrays: Optional[Array]) -> bool:
    """Validate that multiple arrays are either None or all finite."""
    return all(a is None or np.isfinite(a).all() for a in arrays)


def termination_check(
        x: Array, residual: Array, weights: Optional[Array], atol: float, rtol: float,
        norm: Callable[[Array], float]) -> bool:
    """Check whether the residual indicates that iteration should be terminated."""
    tol = atol
    if rtol > 0:
        tol += rtol * norm(weight(x, weights))
    return norm(weight(residual, weights)) < tol


def weight(x: Array, weights: Optional[Array]) -> Array:
    """Optionally weight an array."""
    if weights is None:
        return x
    return weights * x
