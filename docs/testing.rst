Testing
=======

Testing is done with the `tox <https://tox.wiki/en/latest/>`_ automation tool, which runs a `pytest <https://docs.pytest.org/en/latest/>`_-backed test suite in the ``tests/`` directory.


Testing Requirements
--------------------

In addition to the installation requirements for the package itself, running tests and building documentation requires additional packages specified by the ``tests`` and ``docs`` extras in ``setup.py``, along with any other explicitly specified ``deps`` in ``tox.ini``.

The full suite of tests also requires installation of the following software:

- `Artleys Knitro <https://www.artelys.com/solvers/knitro/>`_ version 10.3 or newer: testing optimization routines.
- `MATLAB <https://www.mathworks.com/products/matlab.html>`_: comparing sparse grids with those created by the function `nwspgr <http://www.sparse-grids.de/>`_ created by Florian Heiss and Viktor Winschel, which must be included in a directory on the MATLAB path.
- `R <https://www.r-project.org/>`_: simulating nested logit errors created by the package `evd <https://cran.r-project.org/web/packages/evd/index.html>`_ created by Alec Stephenson, which must be installed.

If software is not installed, its associated tests will be skipped. Additionally, some tests that require support for extended precision will be skipped if on the platform running the tests, ``numpy.longdouble`` has the same precision as ``numpy.float64``. This tends to be the case on Windows.


Running Tests
-------------

Defined in ``tox.ini`` are environments that test the package under different python versions, check types, enforce style guidelines, verify the integrity of the documentation, and release the package. First, `tox <https://tox.wiki/en/latest/>`_ should be installed on top of an Anaconda installation. The following command can be run in the top-level ``pyblp`` directory to run all testing environments::

    tox

You can choose to run only one environment, such as the one that builds the documentation, with the ``-e`` flag::

    tox -e docs


Test Organization
-----------------

Fixtures, which are defined in ``tests.conftest``, configure the testing environment and simulate problems according to a range of specifications.

Most BLP-specific tests in ``tests.test_blp`` verify properties about results obtained by solving the simulated problems under various parameterizations. Examples include:

- Reasonable formulations of problems should give rise to estimated parameters that are close to their true values.
- Cosmetic changes such as the number of processes should not change estimates.
- Post-estimation outputs should satisfy certain properties.
- Optimization routines should behave as expected.
- Derivatives computed with finite differences should approach analytic derivatives.

Tests of generic utilities in ``tests.test_formulation``, ``tests.test_integration``, ``tests.test_iteration``, and ``tests.test_optimization`` verify that matrix formulation, integral approximation, fixed point iteration, and nonlinear optimization all work as expected. Example include:

- Nonlinear formulas give rise to expected matrices and derivatives.
- Gauss-Hermite integrals are better approximated with quadrature based on Gauss-Hermite rules than with Monte Carlo integration.
- To solve a fixed point iteration problem for which it was developed, SQUAREM requires fewer fixed point evaluations than does simple iteration.
- All optimization routines manage to solve a well-known optimization problem under different parameterizations.
