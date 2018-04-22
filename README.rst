pyblp
=====

.. image:: https://readthedocs.org/projects/pyblp/badge/?version=latest
   :alt: Documentation Status
   :scale: 100%
   :target: https://pyblp.readthedocs.io/en/latest/?badge=latest

An overview of the model, examples, and other documentation can be found at `Read the Docs <http://pyblp.readthedocs.io/en/latest/>`_.

The pyblp package is a Python 3 implementation of the nested fixed point algorithm for BLP demand estimation. The author of this package is `Jeff Gortmaker <http://jeffgortmaker.com/>`_. At the moment, the only other contributer is `Chris Conlon <http://www.chrisconlon.org/>`_. Development of the package has been guided by the BLP code made publicly available by many researchers and practitioners.


Installation
------------

The pyblp package has been tested on `Python 3.6 <https://www.python.org/downloads/>`_. The `SciPy instructions <https://scipy.org/install.html>`_ for installing related packages is a good guide for how to install a scientific Python environment. A good choice is the `Anaconda Distribution <https://www.anaconda.com/download/>`_, since, along with many other packages that are useful for scientific computing, it comes packaged with pyblp's only required dependencies: `NumPy <http://www.numpy.org/>`_ and `SciPy <https://www.scipy.org/>`_.

You can install the current release of pyblp with `pip <http://www.pip-installer.org/en/latest/>`_::

    pip install pyblp

You can upgrade to a newer release with the ``--upgrade`` flag::

    pip install --upgrade pyblp

If you lack permissions, you can install pyblp in your user directory with the ``--user`` flag::

    pip install --user pyblp

Alternatively, you can download a wheel or source archive from `PyPI <https://pypi.python.org/pypi/pyblp>`_. You can find the latest development code on `GitHub <https://github.com/jeffgortmaker/pyblp>`_.


Features
--------

- Straightforward interface for configuring and solving BLP problems.
- Support for demographics and supply-side moments.
- Customizable parameter matrices and bounds.
- Post-estimation functions for computing elasticities, diversion ratios, marginal costs, markups, profits, HHI, and consumer surplus.
- Post-estimation merger simulation.
- Flexible interface for simulating synthetic data under Bertrand-Nash competition.
- Optimization with Artleys Knitro, SciPy, or custom routines.
- Fixed point iteration with SQUAREM acceleration or custom routines.
- Integration with Monte Carlo, Gauss-Hermite/Kronrod-Patterson product rules, sparse grids, or custom specifications.
- One-step, two-step, or iterated GMM.  
- Control over weighting matrix and standard error computation.
- Linear or log-linear marginal cost specifications.
- Functions for building indicators and BLP instruments.
- Support for market-by-market parallelization.
- Support for extended floating point precision.
- Robust handling of computational errors.
- Informative and configurable progress updates.


Features Slated for Future Versions
-----------------------------------

In no particular order, listed below are major features that will hopefully be added to the package in future versions:

- Solving with logit and nested logit.
- Tests for identification and overidentifying restrictions.
- Clustered standard errors.
- Fixed effect absorption.
- Parametric bootstrap computation of post-estimation standard errors.
- Built-in IPOPT support.
- Built-in support for more fixed point routines.
- Nonlinear optimization alternatives to fixed point iteration when simulation synthetic data or mergers.
- Hessian computation and tests for local minima.
- Mathematical Program with Equilibrium Constraints (MPEC) formulation of the GMM objective function.
- Generalized Empirical Likelihood (GEL) formulation of the problem.


Bugs and Requests
-----------------

Please use the `GitHub issue tracker <https://github.com/jeffgortmaker/pyblp/issues>`_ to submit bugs or to request features.
