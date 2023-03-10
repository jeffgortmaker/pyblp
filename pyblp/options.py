r"""Global options.

Attributes
----------
digits : `int`
    Number of digits displayed by status updates. The default number of digits is ``7``. The number of digits can be
    changed to, for example, ``2``, with ``pyblp.options.digits = 2``.
verbose : `bool`
    Whether to output status updates. By default, verbosity is turned on. Verbosity can be turned off with
    ``pyblp.options.verbose = False``.
verbose_tracebacks : `bool`
    Whether to include full tracebacks in error messages. By default, full tracebacks are turned off. These can be
    useful when attempting to find the source of an error message. Tracebacks can be turned on with
    ``pyblp.options.verbose_tracebacks = True``.
verbose_output : `callable`
    Function used to output status updates. The default function is simply ``print``. The function can be changed, for
    example, to include an indicator that statuses are from this package, with
    ``pyblp.verbose_output = lambda x: print(f"pyblp: {x}")``.
flush_output : `bool`
    Whether to call ``sys.stdout.flush()`` after outputting a status update. By default, output is not flushed to
    standard output. To force standard output flushes after every status update, set
    ``pyblp.options.flush_output = True``. This may be particularly desirable for R users who are calling PyBLP from
    `reticulate <https://github.com/rstudio/reticulate>`_, since standard output is typically not automatically flushed
    to the screen in this environment. If PyBLP is imported as ``pyblp``, this setting can be enabled in R with
    ``pyblp$options$flush_output <- TRUE``.
dtype : `dtype`
    The data type used for internal calculations, which is by default ``numpy.float64``. The other recommended option is
    ``numpy.longdouble``, which is the only extended precision floating point type currently supported by NumPy.
    Although this data type will be used internally, ``numpy.float64`` will be used when passing arrays to optimization
    and fixed point routines, which may not support extended precision. The library underlying :mod:`scipy.linalg`,
    which is used for matrix inversion, may also use ``numpy.float64``.

    One instance in which extended precision can be helpful in the BLP problem is when there are a large number of near
    zero choice probabilities with small integration weights, which, under standard precision are called zeros when in
    aggregate they are nonzero.

    The precision of ``numpy.longdouble`` depends on the platform on which NumPy is installed. If the platform in use
    does not support extended precision, using ``numpy.longdouble`` may lead to unreliable results. For example, on
    Windows, NumPy is usually compiled such that ``numpy.longdouble`` often behaves like ``numpy.float64``. Precisions
    can be compared with :class:`numpy.finfo` by running ``numpy.finfo(numpy.float64)`` and
    ``numpy.finfo(numpy.longdouble)``. For more information, refer to
    `this discussion <https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html#extended-precision>`_.

    If extended precisions is supported, the data type can be switched with ``pyblp.options.dtype = numpy.longdouble``.
    On Windows, it is often easier to install Linux in a virtual machine than it is to build NumPy from source with a
    non-standard compiler.

finite_differences_epsilon : `float`
    Perturbation :math:`\epsilon` used to numerically approximate derivatives with central finite differences:

    .. math:: f'(x) = \frac{f(x + \epsilon / 2) - f(x - \epsilon / 2)}{\epsilon}.

    By default, this is the square root of the machine epsilon: ``numpy.sqrt(numpy.finfo(options.dtype).eps)``. The
    typical example where this is used is when computing the Hessian, but it may also be used to compute Jacobians
    required for standard errors when analytic gradients are disabled.

pseudo_inverses : `bool`
    Whether to compute Moore-Penrose pseudo-inverses of matrices with :func:`scipy.linalg.pinv` instead of their classic
    inverses with :func:`scipy.linalg.inv`. This is by default ``True``, so pseudo-inverses will be used. Up to small
    numerical differences, the pseudo-inverse is identical to the classic inverse for invertible matrices. Using the
    pseudo-inverse by default can help alleviate problems from, for example, near-singular weighting matrices.

    To always attempt to compute classic inverses first, set ``pyblp.options.pseudo_inverses = False``. If a classic
    inverse cannot be computed, an error will be displayed, and a pseudo-inverse may be computed instead.

weights_tol : `float`
    Tolerance for detecting integration weights that do not sum to one, which is by default ``1e-10``. In most setups
    weights should essentially sum to one, but for example with importance sampling they may be slightly different.
    Warnings can be disabled by setting this to ``numpy.inf``.
singular_tol : `float`
    Tolerance for detecting singular matrices, which is by default ``1 / numpy.finfo(options.dtype).eps``. If a matrix
    has a condition number larger than this tolerance, a warning will be displayed. To disable singularity checks, set
    ``pyblp.options.singular_tol = numpy.inf``.
collinear_atol : `float`
    Absolute tolerance for detecting collinear columns in each matrix of product characteristics and instruments:
    :math:`X_1`, :math:`X_2`, :math:`X_3`, :math:`Z_D`, and :math:`Z_S`. Micro moments can also be checked; see
    ``detect_micro_collinearity``.

    Each matrix is decomposed into a :math:`QR` decomposition and an error is raised for any column whose diagonal
    element in :math:`R` has a magnitude less than ``collinear_atol + collinear_rtol * sd`` where ``sd`` is the column's
    standard deviation.

    The default absolute tolerance is ``1e-10``. To disable collinearity checks, set
    ``pyblp.options.collinear_atol = pyblp.options.collinear_rtol = 0``.

collinear_rtol : `float`
    Relative tolerance for detecting collinear columns, which is by default also ``1e-10``.
psd_atol : `float`
    Absolute tolerance for detecting non-positive semidefinite matrices. For example, this check is applied to any
    custom weighting matrix, :math:`W`.

    Singular value decomposition factorizes the matrix into :math:`U \Sigma V` and an error is raised if any element in
    the original matrix differs in absolute value from :math:`V' \Sigma V` by more than ``psd_atol + psd_rtol * abs``
    where ``abs`` is the element's absolute value.

    The default tolerance is ``1e-8``. To disable positive semidefinite checks, set
    ``pyblp.options.psd_atol = pyblp.options.psd_rtol = numpy.inf``.

psd_rtol : `float`
    Relative tolerance for detecting non-positive definite matrices, which is by default also ``1e-8``.
detect_micro_collinearity : `bool`
    Whether to check if micro values :math:`v_{pijt}` (or :math:`v_{pijkt}` with second choices) are collinear with one
    another by computing these values once, stacking them, and using ``pyblp.options.collinear_atol` and
    ``pyblp.options.collinear_rtol``.

    By default, micro values are not checked for collinearity because this procedure can require a large amount of
    memory. To enable this check, set ``pyblp.options.detect_micro_collinearity = True``. If this uses a large amount
    of memory, one option is to temporarily reduce the number of markets, observations, or agents to cut down on memory
    while debugging one's code to see which micro moments are collinear with one another.

micro_computation_chunks : `int or dict`
    How finely to break up micro moment computation within market. Computation is broken up by groups of agents within
    market. This can help reduce the amount of memory being used by micro moments when there are a large number of
    agents and products, and especially when second choice micro moments are being used.

    By default, micro moment computation is done in one chunk for each market. To reduce memory usage without changing
    any estimation results, for example by splitting up computation into 10 chunks, use
    ``pyblp.options.micro_computation_chunks = 10``.

    If a dictionary, this should map market IDs to the number of chunks to use. For example, to only chunk computation
    in market ID ``'big market'``, use ``pyblp.options.micro_computation_chunks = {'big_market': 10}``.

drop_product_fields : `bool`
    Whether to conserve memory by dropping product data fields that are not needed for market-level computation when
    initializing a market. By default, these fields are not dropped. Setting
    ``pyblp.options.drop_product_fields = True`` may reduce memory usage, especially if there are many instruments,
    at the cost of extra time needed to drop these fields.

"""

import numpy as _np


digits = 7
verbose = True
verbose_tracebacks = False
verbose_output = print
flush_output = False
dtype = _np.float64
finite_differences_epsilon = _np.sqrt(_np.finfo(dtype).eps)
pseudo_inverses = True
weights_tol = 1e-10
singular_tol = 1 / _np.finfo(dtype).eps
collinear_atol = collinear_rtol = 1e-10
psd_atol = psd_rtol = 1e-8
detect_micro_collinearity = False
micro_computation_chunks = 1
drop_product_fields = False
