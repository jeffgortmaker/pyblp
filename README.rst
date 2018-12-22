pyblp
=====

|docs-badge|_ |pypi-badge|_ |downloads-badge|_ |python-badge|_ |license-badge|_

.. |docs-badge| image:: https://img.shields.io/readthedocs/pyblp/stable.svg
.. _docs-badge: https://pyblp.readthedocs.io/en/stable/

.. |pypi-badge| image:: https://img.shields.io/pypi/v/pyblp.svg
.. _pypi-badge: https://pypi.org/project/pyblp/

.. |downloads-badge| image:: https://pepy.tech/badge/pyblp
.. _downloads-badge: https://pepy.tech/project/pyblp

.. |python-badge| image:: https://img.shields.io/pypi/pyversions/pyblp.svg
.. _python-badge: https://pypi.org/project/pyblp/

.. |license-badge| image:: https://img.shields.io/pypi/l/pyblp.svg
.. _license-badge: https://pypi.org/project/pyblp/

.. description-start

An overview of the model, examples, references, and other documentation can be found on `Read the Docs <https://pyblp.readthedocs.io/en/stable/>`_.

.. docs-start

The pyblp package is a Python 3 implementation of routines for estimating demand with BLP-type random coefficients logit models. The author of this package is `Jeff Gortmaker <https://jeffgortmaker.com/>`_. At the moment, the only other contributer is `Chris Conlon <https://chrisconlon.github.io/>`_. Development of the package has been guided by code made publicly available by many researchers and practitioners. Views expressed in the documentation of this package are those of the contributers and do not necessarily reflect the views of any institution at which they are employed.


Installation
------------

The pyblp package has been tested on `Python <https://www.python.org/downloads/>`_ versions 3.6 and 3.7. The `SciPy instructions <https://scipy.org/install.html>`_ for installing related packages is a good guide for how to install a scientific Python environment. A good choice is the `Anaconda Distribution <https://www.anaconda.com/download/>`_, since, along with many other packages that are useful for scientific computing, it comes packaged with pyblp's only required dependencies: `NumPy <https://www.numpy.org/>`_, `SciPy <https://www.scipy.org/>`_, `SymPy <https://www.sympy.org/en/index.html>`_, and `Patsy <https://patsy.readthedocs.io/en/latest/>`_.

You can install the current release of pyblp with `pip <https://pip.pypa.io/en/latest/>`_::

    pip install pyblp

You can upgrade to a newer release with the ``--upgrade`` flag::

    pip install --upgrade pyblp

If you lack permissions, you can install pyblp in your user directory with the ``--user`` flag::

    pip install --user pyblp

Alternatively, you can download a wheel or source archive from `PyPI <https://pypi.org/project/pyblp/>`_. You can find the latest development code on `GitHub <https://github.com/jeffgortmaker/pyblp/>`_ and the latest development documentation `here <https://pyblp.readthedocs.io/en/latest/>`_.


Features
--------

- R-style formula interface
- Bertrand-Nash supply-side moments
- Multiple equation GMM
- Demographic interactions
- Fixed effect absorption
- Nonlinear functions of product characteristics
- Concentrating out of linear parameters
- Parameter bounds and constraints
- Random coefficients nested logit (RCNL)
- Varying nesting parameters across groups
- Logit and nested logit benchmarks
- Classic BLP instruments
- Optimal instruments
- Elasticities and diversion ratios
- Marginal costs and markups
- Profits and consumer surplus
- Merger simulation
- Parametric boostrapping of post-estimation outputs
- Synthetic data construction
- SciPy or Artleys Knitro optimization
- Fixed point acceleration
- Monte Carlo, product rule, or sparse grid integration
- Custom optimization and iteration routines
- Robust and clustered errors
- Linear or log-linear marginal costs
- Partial ownership matrices
- Analytic gradients
- Finite difference Hessians
- Market-by-market parallelization
- Extended floating point precision
- Robust error handling


Features Slated for Future Versions
-----------------------------------

- Analytic Hessian computation
- Mathematical Program with Equilibrium Constraints (MPEC)
- Generalized Empirical Likelihood (GEL)
- Micro moments
- Agent type mixtures
- More optimization and iteration routines


Bugs and Requests
-----------------

Please use the `GitHub issue tracker <https://github.com/jeffgortmaker/pyblp/issues>`_ to submit bugs or to request features.
