"""Fixed-point iteration routines."""

import numpy as np


class Iteration(object):
    r"""Configuration for solving fixed point problems.

    Parameters
    ----------
    method : `str`
        The fixed point iteration routine that will be used. One of the following:

            - ``'simple'`` - Non-accelerated iteration.

            - ``'squarem'`` - SQUAREM acceleration method of :ref:`Varadhan and Roland (2008) <vr08>` and considered in
              the context of the BLP problem in :ref:`Reynaerts, Varadhan, and Nash (2012) <rvn12>`. This implementation
              uses a first-order squared non-monotone extrapolation scheme. If there are any errors during the
              acceleration step, it uses the last values for the next iteration of the algorithm.

        Also accepted is a custom callable method with the following form::

            method(initial, contraction, callback, **options) -> (final, converged)

        where `initial` is an array of initial values, `contraction` is a callable contraction mapping, `callback` is a
        function that should be called without any arguments after each major iteration (it is used to record the number
        of major iterations), `options` are specified below, `final` is an array of final values, and `converged` is a
        flag for whether the routine converged.

    method_options : `dict, optional`
        Options for the fixed point iteration routine. Both non-custom routines support the following options:

            - **max_evaluations** : (`int`) - Maximum number of contraction mapping evaluations. The default value is
              ``10000``.

            - **tol** : (`float`) - Tolerance for convergence of the configured norm. The default value is ``1e-12``.

            - **norm** : (`callable`) - The norm to be used. By default, the :math:`\ell^2`-norm is used. If specified,
              this should be a function that accepts an array of differences and that returns a scalar norm.

        The ``'squarem'`` routine accepts additional options that mirror those in the
        `SQUAREM <https://cran.r-project.org/web/packages/SQUAREM/index.html>`_ package, written in R by Ravi Varadhan,
        which identifies the step length with :math:`-\alpha` from :ref:`Varadhan and Roland (2008) <vr08>`:

            - **scheme** : (`int`) - The default value is ``3``, which corresponds to S3 in
              :ref:`Varadhan and Roland (2008) <vr08>`. Other acceptable schemes are ``1`` and ``2``, which correspond
              to S1 and S2.

            - **step_min** : (`float`) - The initial value for the minimum step length. The default value is ``1.0``.

            - **step_max** : (`float`) - The initial value for the maximum step length. The default value is ``1.0``.

            - **step_factor** : (`float`) - When the step length exceeds `step_max`, it is set equal to `step_max`, but
              `step_max` is scaled by this factor. Similarly, if `step_min` is negative and the step length is below
              `step_min`, it is set equal to `step_min` and `step_min` is scaled by this factor. The default value is
              ``4.0``.

    Examples
    --------
    The following code builds a SQUAREM configuration with a :math:`\ell^\infty`-norm that uses scheme S1 from
    :ref:`Varadhan and Roland (2008) <vr08>`:

    .. ipython:: python

       iteration = pyblp.Iteration('squarem', {'norm': lambda x: np.abs(x).max(), 'scheme': 1})

    Instead of using a non-custom routine, the following code builds a custom method that implements a version of simple
    iteration, which, for the sake of having a nontrivial example, arbitrarily identifies a major iteration with three
    objective evaluations:

    .. ipython:: python

       def custom_method(initial, contraction, callback, max_evaluations, tol, norm):
           evaluations = 1
           x = contraction(initial)
           while evaluations < max_evaluations and norm(x) < tol:
               evaluations += 1
               x = contraction(x)
               if evaluations % 3 == 0:
                   callback()
           return x, evaluations < max_evaluations

    You can then use this custom method to build an iteration configuration:

    .. ipython:: python

       iteration = pyblp.Iteration(custom_method)

    """

    def __init__(self, method, method_options=None):
        """Validate the method and configure default options."""
        methods = {
            'squarem': (squarem_iterator, "the SQUAREM acceleration method"),
            'simple': (simple_iterator, "no acceleration")
        }

        # validate the configuration
        if method not in methods and not callable(method):
            raise ValueError(f"method must be one of {list(methods.keys())} or a callable object.")
        if method_options is not None and not isinstance(method_options, dict):
            raise ValueError("method_options must be None or a dict.")

        # replace missing options with a dict
        method_options = method_options or {}

        # options are simply passed along to custom methods
        if callable(method):
            self._iterator = method
            self._method_options = method_options
            self._description = "a custom method"
            return

        # identify the non-custom iterator and set default options
        self._iterator, self._description = methods[method]
        self._method_options = {
            'tol': 1e-12,
            'max_evaluations': 10000,
            'norm': np.linalg.norm
        }
        if self._iterator == squarem_iterator:
            self._method_options.update({
                'scheme': 3,
                'step_min': 1.0,
                'step_max': 1.0,
                'step_factor': 4.0
            })

        # validate options for non-custom methods
        invalid = [k for k in method_options if k not in self._method_options]
        if invalid:
            raise KeyError(f"The following are not valid iteration options: {invalid}.")
        self._method_options.update(method_options)
        if not isinstance(self._method_options['tol'], float) or self._method_options['tol'] <= 0:
            raise ValueError("The iteration option tol must be a positive float.")
        if not isinstance(self._method_options['max_evaluations'], int) or self._method_options['max_evaluations'] < 1:
            raise ValueError("The iteration option max_evaluations must be a positive integer.")
        if not callable(self._method_options['norm']):
            raise ValueError("The iteration option norm must be callable.")
        if self._iterator == squarem_iterator:
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

    def __str__(self):
        """Format the configuration as a string."""
        strings = {k: f'{v.__module__}.{v.__qualname__}' if callable(v) else v for k, v in self._method_options.items()}
        return f"Configured to iterate using {self._description} with options {strings}."

    def _iterate(self, initial_values, contraction):
        """Solve a fixed point iteration problem."""

        # define a callback, which counts the number of major iterations
        def iteration_callback():
            iteration_callback.iterations += 1

        # define a wrapper for the contraction, which normalizes arrays so they work with all types of routines, and
        #   also counts the total number of contraction evaluations
        def contraction_wrapper(raw_values):
            contraction_wrapper.evaluations += 1
            raw_values = np.asarray(raw_values)
            values = raw_values.reshape(initial_values.shape).astype(initial_values.dtype)
            return np.asarray(contraction(values)).astype(np.float64).reshape(raw_values.shape)

        # initialize the counters and normalize the starting values
        iteration_callback.iterations = contraction_wrapper.evaluations = 0
        raw_initial_values = initial_values.astype(np.float64).flatten()

        # solve the problem and convert the raw final values to the same data type and shape as the initial values
        raw_final_values, converged = self._iterator(
            raw_initial_values,
            contraction_wrapper,
            iteration_callback,
            **self._method_options
        )
        final_values = np.asarray(raw_final_values).reshape(initial_values.shape).astype(initial_values.dtype)
        return final_values, converged, iteration_callback.iterations, contraction_wrapper.evaluations


def squarem_iterator(x, contraction, iteration_callback, max_evaluations, tol, norm, scheme, step_min, step_max, step_factor):
    """Apply the SQUAREM acceleration method for fixed point iteration. The fixed point array and a flag for whether the
    routine converged are both returned.
    """
    evaluations = 0
    while True:
        # first step
        x0, x = x, contraction(x)
        g0 = x - x0
        evaluations += 1
        if evaluations >= max_evaluations or not np.isfinite(x).all() or norm(g0) < tol:
            break

        # second step
        x1, x = x, contraction(x)
        g1 = x - x1
        evaluations += 1
        if evaluations >= max_evaluations or not np.isfinite(x).all() or norm(g1) < tol:
            break

        # compute the step length
        r = g0
        v = g1 - g0
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
        with np.errstate(all='ignore'):
            x2, x = x, x0 - 2 * alpha * r + (alpha ** 2) * v
            x3, x = x, contraction(x)

        # record the completion of a major iteration
        iteration_callback()

        # revert to the last evaluation if there were errors
        if not np.isfinite(x).all():
            x = x2
            continue

        # check for convergence
        evaluations += 1
        if evaluations >= max_evaluations or not np.isfinite(x).all() or norm(x - x3) < tol:
            break

    # determine whether there was convergence
    return x, evaluations < max_evaluations


def simple_iterator(x, contraction, iteration_callback, max_evaluations, tol, norm):
    """Perform simple fixed point iteration with no acceleration. The fixed point array and a flag for whether the
    routine converged are both returned.
    """
    evaluations = 0
    while True:
        # for simple iteration, a contraction evaluation is the same as a major iteration
        x0, x = x, contraction(x)
        iteration_callback()
        evaluations += 1
        if evaluations >= max_evaluations or not np.isfinite(x).all() or norm(x - x0) < tol:
            break

    # determine whether there was convergence
    return x, evaluations < max_evaluations
