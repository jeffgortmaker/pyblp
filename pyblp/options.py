"""Global options.

Attributes
----------
digits : `int`
    Number of digits displayed by status updates. The default number of digits is ``10``. The number of digits can be
    changed to, for example, ``20``, with ``pyblp.options.digits = 20``.

verbose : `bool`
    Whether to output information and status updates. By default, verbosity is turned on. Verbosity can be turned off
    with ``pyblp.options.verbose = False``.

dtype : `dtype`
    The data type used for internal calculations, which is by default ``numpy.float64``. The other recommended option is
    ``numpy.longdouble``, which is the only extended precision floating point type currently supported by NumPy.
    Although this data type will be used internally, it will be converted to ``numpy.float64`` when passing arrays to
    optimization and fixed point routines, which may not support extended precision.

    One instance in which extended precision can be helpful in the BLP problem is when there are a large number of near
    zero choice probabilities with small integration weights, which, under standard precision are called zeros when in
    aggregate they are nonzero. For example, :ref:`Skrainka (2012) <s12>` finds that using long doubles is sufficient
    to solve many utility floating point problems.

    The precision of ``numpy.longdouble`` depends on the platform on which NumPy is installed. For example, on Windows,
    NumPy is usually compiled such that ``numpy.longdouble`` is identical to ``numpy.float64``. Precisions can be
    compared with :class:`numpy.finfo`.

    .. ipython:: python

       np.finfo(np.float64)
       np.finfo(np.longdouble)

    For more information, refer to
    `this discussion <https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html#extended-precision>`_.

    If extended precisions is supported, the data type can be switched with ``pyblp.options.dtype = np.longdouble``. On
    Windows, it is often easier to install Linux in a virtual machine than it is to build NumPy from source with a
    non-standard compiler.

"""

import numpy as np


digits = 10
verbose = True
dtype = np.float64
