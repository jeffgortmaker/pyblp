PyBLP
=====

|docs-badge|_ |pypi-badge|_ |downloads-badge|_ |python-badge|_ |license-badge|_

.. |docs-badge| image:: https://img.shields.io/readthedocs/pyblp/stable.svg
.. _docs-badge: https://pyblp.readthedocs.io/en/stable/

.. |pypi-badge| image:: https://img.shields.io/pypi/v/pyblp.svg
.. _pypi-badge: https://pypi.org/project/pyblp/

.. |downloads-badge| image:: https://img.shields.io/pypi/dm/pyblp.svg
.. _downloads-badge: https://pypistats.org/packages/pyblp

.. |python-badge| image:: https://img.shields.io/pypi/pyversions/pyblp.svg
.. _python-badge: https://pypi.org/project/pyblp/

.. |license-badge| image:: https://img.shields.io/pypi/l/pyblp.svg
.. _license-badge: https://pypi.org/project/pyblp/

.. description-start

An overview of the model, examples, references, and other documentation can be found on `Read the Docs <https://pyblp.readthedocs.io/en/stable/>`_.

.. docs-start

PyBLP is a Python 3 implementation of routines for estimating the demand for differentiated products with BLP-type random coefficients logit models. This package was created by `Jeff Gortmaker <https://jeffgortmaker.com/>`_ in collaboration with `Chris Conlon <https://chrisconlon.github.io/>`_.

Development of the package has been guided by the work of many researchers and practitioners. For a full list of references, including the original work of `Berry, Levinsohn, and Pakes (1995) <https://ideas.repec.org/a/ecm/emetrp/v63y1995i4p841-90.html>`_, refer to the `references <https://pyblp.readthedocs.io/en/stable/references.html>`_ section of the documentation.


Citation
--------

If you use PyBLP in your research, we ask that you also cite `Conlon and Gortmaker (2020) <https://jeffgortmaker.com/files/pyblp.pdf>`_, which describes the advances implemented in the package.
::

    @article{PyBLP,
        author = {Conlon, Christopher and Gortmaker, Jeff},
        title = {Best practices for differentiated products demand estimation with {PyBLP}},
        journal = {The RAND Journal of Economics},
        volume = {51},
        number = {4},
        pages = {1108-1161},
        doi = {https://doi.org/10.1111/1756-2171.12352},
        url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/1756-2171.12352},
        eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/1756-2171.12352},
        year = {2020}
    }


If you use PyBLP's micro moments functionality, we ask that you also cite `Conlon and Gortmaker (2023) <https://jeffgortmaker.com/files/micro.pdf>`_, which describes the standardized framework implemented by PyBLP for incorporating micro data into BLP-style estimation.
::

    @misc{MicroPyBLP,
        author = {Conlon, Christopher and Gortmaker, Jeff},
        title = {Incorporating micro data into differentiated products demand estimation with {PyBLP}},
        note = {Working paper},
        year = {2023}
    }


Installation
------------

The PyBLP package has been tested on `Python <https://www.python.org/downloads/>`_ versions 3.6 through 3.9. The `SciPy instructions <https://scipy.org/install/>`_ for installing related packages is a good guide for how to install a scientific Python environment. A good choice is the `Anaconda Distribution <https://www.anaconda.com/download>`_, since it comes packaged with the following PyBLP dependencies: `NumPy <https://numpy.org/>`_, `SciPy <https://scipy.org/>`_, `SymPy <https://www.sympy.org/en/index.html>`_, and `Patsy <https://patsy.readthedocs.io/en/latest/>`_. For absorption of high dimension fixed effects, PyBLP also depends on its companion package `PyHDFE <https://github.com/jeffgortmaker/pyhdfe>`_, which will be installed when PyBLP is installed.

However, PyBLP may not work with old versions of its dependencies. You can update PyBLP's Anaconda dependencies with::

    conda update numpy scipy sympy patsy

You can update PyHDFE with::

    pip install --upgrade pyhdfe

You can install the current release of PyBLP with `pip <https://pip.pypa.io/en/latest/>`_::

    pip install pyblp

You can upgrade to a newer release with the ``--upgrade`` flag::

    pip install --upgrade pyblp

If you lack permissions, you can install PyBLP in your user directory with the ``--user`` flag::

    pip install --user pyblp

Alternatively, you can download a wheel or source archive from `PyPI <https://pypi.org/project/pyblp/>`_. You can find the latest development code on `GitHub <https://github.com/jeffgortmaker/pyblp/>`_ and the latest development documentation `here <https://pyblp.readthedocs.io/en/latest/>`_.


Other Languages
---------------

Once installed, PyBLP can be incorporated into projects written in many other languages with the help of various tools that enable interoperability with Python.

For example, the `reticulate <https://github.com/rstudio/reticulate>`_ package makes interacting with PyBLP in R straightforward (when supported, Python objects can be converted to their R counterparts with the ``py_to_r`` function, which needs to be used manually because we set ``convert=FALSE`` to get rid of errors about trying to automatically convert unsupported objects)::

    library(reticulate)
    pyblp <- import("pyblp", convert=FALSE)
    pyblp$options$flush_output <- TRUE

Similarly, `PyCall <https://github.com/JuliaPy/PyCall.jl>`_ can be used to incorporate PyBLP into a Julia workflow::

    using PyCall
    pyblp = pyimport("pyblp")

The `py command <https://www.mathworks.com/help/matlab/call-python-libraries.html>`_ serves a similar purpose in MATLAB::

   py.pyblp


Features
--------

- R-style formula interface
- Bertrand-Nash supply-side moments
- Multiple equation GMM
- Demographic interactions
- Product-specific demographics
- Consumer-specific product availability
- Flexible micro moments that can match statistics based on survey data
- Support for micro moments based on second choice data
- Support for optimal micro moments that match micro data scores
- Fixed effect absorption
- Nonlinear functions of product characteristics
- Concentrating out linear parameters
- Flexible random coefficient distributions
- Parameter bounds and constraints
- Random coefficients nested logit (RCNL)
- Approximation to the pure characteristics model
- Varying nesting parameters across groups
- Logit and nested logit benchmarks
- Classic BLP instruments
- Differentiation instruments
- Optimal instruments
- Covariance restrictions
- Adjustments for simulation error
- Tests of overidentifying and model restrictions
- Parametric boostrapping post-estimation outputs
- Elasticities and diversion ratios
- Marginal costs and markups
- Passthrough calculations
- Profits and consumer surplus
- Newton and fixed point methods for computing pricing equilibria
- Merger simulation
- Custom counterfactual simulation
- Synthetic data construction
- SciPy or Artleys Knitro optimization
- Fixed point acceleration
- Monte Carlo, quasi-random sequences, quadrature, and sparse grids
- Importance sampling
- Custom optimization and iteration routines
- Robust and clustered errors
- Linear or log-linear marginal costs
- Partial ownership matrices
- Analytic gradients
- Finite difference Hessians
- Market-by-market parallelization
- Extended floating point precision
- Robust error handling


Bugs and Requests
-----------------

Please use the `GitHub issue tracker <https://github.com/jeffgortmaker/pyblp/issues>`_ to submit bugs or to request features.
