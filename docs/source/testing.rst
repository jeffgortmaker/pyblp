Testing
=======

Testing is done with the `tox <https://tox.readthedocs.io/en/latest/>`_ automation tool, which runs a `pytest <https://docs.pytest.org/en/latest/>`_-backed test suite in the ``tests/`` directory.


Running Tests
-------------

Defined in ``tox.ini`` are environments that test the package, as well as the integrity of the documentation. The following command can be run in the top-level pyblp directory to run all tests in both the ``py36`` and ``docs`` environments::

    tox

You can run tests with multiple processes by passing along the ``-n`` flag to ``pytest-xdist``::

    tox -- -n 3

You can choose to only compile and test the documentation with the ``-e`` flag::

    tox -e docs


Test Organization
-----------------

Fixtures, which are defined in ``tests.confest``, load example problems and simulate other problems according to a range of specifications. Other replication data are loaded from ``tests/data/``, which are compared with code output.

Most BLP-specific tests in ``tests.test_blp`` verify properties about results obtained by solving the simulated problems under various parameterizations. Examples include:

- Reasonable formulations of the BLP problem should give rise to estimated parameters that are on average close to their true values.
- Cosmetic changes such as the number of processes should not change estimates.
- Post-estimation outputs should satisfy proven mathematical properties.
- Optimization routines should behave as expected.
- Gradients computed with finite differences should approach analytic gradients.
- Results from example problems should line up with results reported in the literature.

Tests of generic utilities in ``tests.test_integration``, ``tests.test_iteration``, and ``tests.test_optimization`` verify that integral approximation, fixed point iteration, and nonlinear optimization all work as expected. Example include:

- Gauss-Hermite integrals are better approximated with quadrature based on Gauss-Hermite rules than with Monte Carlo integration.
- To solve a fixed point iteration problem for which it was developed, SQUAREM requires fewer fixed point evaluations than does simple iteration.
- All optimization routines manage to solve a well-known optimization problem under different parameterizations.
