pyblp
=====

|docs-badge|_ |pypi-badge|_ |python-badge|_ |license-badge|_

.. |docs-badge| image:: https://img.shields.io/readthedocs/pyblp/stable.svg
.. _docs-badge: https://pyblp.readthedocs.io/en/stable/

.. |pypi-badge| image:: https://img.shields.io/pypi/v/pyblp.svg
.. _pypi-badge: https://pypi.org/project/pyblp/

.. |python-badge| image::  https://img.shields.io/pypi/pyversions/pyblp.svg
.. _python-badge: https://pypi.org/project/pyblp/

.. |license-badge| image:: https://img.shields.io/pypi/l/pyblp.svg
.. _license-badge: https://pypi.org/project/pyblp/

.. description-start

An overview of the model, examples, and other documentation can be found on `Read the Docs <https://pyblp.readthedocs.io/en/stable/>`_.

.. docs-start

The pyblp package is a Python 3 implementation of the nested fixed point algorithm for BLP demand estimation. The author of this package is `Jeff Gortmaker <http://jeffgortmaker.com/>`_. At the moment, the only other contributer is `Chris Conlon <https://chrisconlon.github.io/>`_. Development of the package has been guided by the BLP code made publicly available by many researchers and practitioners. Views expressed in the documentation of this package are those of the contributers and do not necessarily reflect the views of any institution to which they belong.


Installation
------------

The pyblp package has been tested on `Python 3.6 <https://www.python.org/downloads/>`_. The `SciPy instructions <https://scipy.org/install.html>`_ for installing related packages is a good guide for how to install a scientific Python environment. A good choice is the `Anaconda Distribution <https://www.anaconda.com/download/>`_, since, along with many other packages that are useful for scientific computing, it comes packaged with pyblp's only required dependencies: `NumPy <https://www.numpy.org/>`_, `SciPy <https://www.scipy.org/>`_, `SymPy <https://www.sympy.org/en/index.html>`_, and `Patsy <https://patsy.readthedocs.io/en/latest/>`_.

You can install the current release of pyblp with `pip <https://pip.pypa.io/en/latest/>`_::

    pip install pyblp

You can upgrade to a newer release with the ``--upgrade`` flag::

    pip install --upgrade pyblp

If you lack permissions, you can install pyblp in your user directory with the ``--user`` flag::

    pip install --user pyblp

Alternatively, you can download a wheel or source archive from `PyPI <https://pypi.org/project/pyblp/>`_. You can find the latest development code on `GitHub <https://github.com/jeffgortmaker/pyblp/>`_ and the latest development documentation `here <http://pyblp.readthedocs.io/en/latest/>`_.


Features
--------

- Straightforward interface for configuring and solving BLP problems with R-style formulas.
- Support for absorption of arbitrary fixed effects with iterative demeaning.
- Support for robust and clustered standard errors.
- Support for demographics and supply-side moments.
- Support for nonlinear functions and interactions of product characteristics.
- Customizable parameter matrices and bounds.
- Estimation of a Logit bechmark model.
- Post-estimation functions for computing elasticities, diversion ratios, marginal costs, markups, profits, HHI, and consumer surplus.
- Post-estimation merger simulation.
- Flexible interface for simulating synthetic data under Bertrand-Nash competition.
- Optimization with Artleys Knitro, SciPy, or custom routines.
- Fixed point iteration with SQUAREM acceleration or custom routines.
- Integration with Monte Carlo, Gauss-Hermite/Kronrod-Patterson product rules, sparse grids, or custom specifications.
- One-step, two-step, or iterated GMM.  
- Control over weighting matrix and standard error computation.
- Linear or log-linear marginal cost specifications.
- Support for non-standard ownership matrices.
- Functions for building ownership matrices and BLP instruments.
- Support for market-by-market parallelization.
- Support for extended floating point precision.
- Robust handling of computational errors.
- Informative and configurable progress updates.


Features Slated for Future Versions
-----------------------------------

In no particular order, listed below are major features that will hopefully be added to the package in future versions:

- Support for nesting parameters.
- Parametric bootstrap computation of post-estimation standard errors.
- Built-in IPOPT support.
- Built-in support for more fixed point routines.
- Nonlinear optimization alternatives to fixed point iteration when computing Bertrand-Nash prices and shares.
- Hessian computation and tests for local minima.
- Mathematical Program with Equilibrium Constraints (MPEC) formulation of the GMM objective function.
- Generalized Empirical Likelihood (GEL) formulation of the problem.


Bugs and Requests
-----------------

Please use the `GitHub issue tracker <https://github.com/jeffgortmaker/pyblp/issues>`_ to submit bugs or to request features.
