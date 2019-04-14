"""Global options.

Attributes
----------
digits : `int`
    Number of digits displayed by status updates. The default number of digits is ``10``. The number of digits can be
    changed to, for example, ``20``, with ``pyblp.options.digits = 20``.
verbose : `bool`
    Whether to output status updates. By default, verbosity is turned on. Verbosity can be turned off with
    ``pyblp.options.verbose = False``.
verbose_output : `callable`
    Function used to output status updates. The default function is simply ``print``. The function can be changed, for
    example, to include an indicator that statuses are from this package, with
    ``pyblp.verbose_output = lambda x: print(f"pyblp: {x}")``.
dtype : `dtype`
    The data type used for internal calculations, which is by default ``numpy.float64``. The other recommended option is
    ``numpy.longdouble``, which is the only extended precision floating point type currently supported by NumPy.
    Although this data type will be used internally, ``numpy.float64`` will be used when passing arrays to optimization
    and fixed point routines, which may not support extended precision. The library underlying :mod:`scipy.linalg`,
    which is used for matrix inversion, may also use ``numpy.float64``.

    One instance in which extended precision can be helpful in the BLP problem is when there are a large number of near
    zero choice probabilities with small integration weights, which, under standard precision are called zeros when in
    aggregate they are nonzero. For example, :ref:`references:Skrainka (2012)` finds that using long doubles is
    sufficient to solve many utility floating point problems.

    The precision of ``numpy.longdouble`` depends on the platform on which NumPy is installed. If the platform in use
    does not support extended precision, using ``numpy.longdouble`` may lead to unreliably results. For example, on
    Windows, NumPy is usually compiled such that ``numpy.longdouble`` often behaves like ``numpy.float64``. Precisions
    can be compared with :class:`numpy.finfo` by running ``np.finfo(np.float64)`` and ``np.finfo(np.longdouble)``. For
    more information, refer to
    `this discussion <https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html#extended-precision>`_.

    If extended precisions is supported, the data type can be switched with ``pyblp.options.dtype = np.longdouble``. On
    Windows, it is often easier to install Linux in a virtual machine than it is to build NumPy from source with a
    non-standard compiler.

collinear_atol : `float`
    Absolute tolerance for detecting collinear columns in each matrix of product characteristics and instruments:
    :math:`X_1`, :math:`X_2`, :math:`X_3`, :math:`Z_D`, and :math:`Z_S`.

    Each matrix is decomposed into a :math:`QR` decomposition and an error is raised for any column whose diagonal
    element in :math:`R` has a magnitude less than ``collinear_atol + collinear_rtol * sd`` where ``sd`` is the column's
    standard deviation.

    The default absolute tolerance is ``1e-14``. To disable collinearity checks, set
    ``pyblp.options.collinear_atol = pyblp.options.collinear_rtol = 0``.

collinear_rtol : `float`
    Relative tolerance for detecting collinear columns, which is by default also ``1e-14``.

"""

import numpy as _np


digits = 10
verbose = True
verbose_output = print
dtype = _np.float64
collinear_atol = collinear_rtol = 1e-14
