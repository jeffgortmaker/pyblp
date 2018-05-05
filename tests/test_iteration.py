"""Tests of fixed point iteration routines."""

import pytest
import numpy as np
import scipy.optimize

from pyblp.utilities import Iteration


@pytest.mark.parametrize(['method', 'options'], [
    pytest.param('simple', {}, id="simple"),
    pytest.param('simple', {'norm': lambda x: np.linalg.norm(x, np.inf)}, id="simple with infinity norm"),
    pytest.param('squarem', {'scheme': 1, 'step_min': 0.9, 'step_max': 1.1, 'step_factor': 3.0}, id="SQUAREM S1"),
    pytest.param('squarem', {'scheme': 2, 'step_min': 0.8, 'step_max': 1.2, 'step_factor': 4.0}, id="SQUAREM S2"),
    pytest.param('squarem', {'scheme': 3, 'step_min': 0.7, 'step_max': 1.3, 'step_factor': 5.0}, id="SQUAREM S3"),
    pytest.param(lambda c, x, _, tol: (scipy.optimize.fixed_point(c, x, xtol=tol), True), {}, id="custom")
])
@pytest.mark.parametrize('tol', [
    pytest.param(1e-2, id="large"),
    pytest.param(1e-4, id="medium"),
    pytest.param(1e-8, id="small")
])
def test_scipy(method, options, tol):
    """Test that the solution to the example fixed point problem from scipy.optimize.fixed_point is reasonably close to
    the exact solution.
    """
    options['tol'] = tol
    contraction = lambda x: np.sqrt(np.array([10, 12]) / (x + np.array([3, 5])))
    exact = [1.4920333, 1.37228132]
    computed = Iteration(method, options)._iterate(contraction, np.ones(2))[0]
    np.testing.assert_allclose(exact, computed, rtol=0, atol=10 * tol)


@pytest.mark.parametrize('scheme', [pytest.param(1, id="S1"), pytest.param(2, id="S2"), pytest.param(3, id="S3")])
def test_hasselblad(scheme):
    """Test that the solution to the fixed point problem from Hasselblad (1969) is reasonably close to the exact
    solution and that SQUAREM takes at least an order of magnitude fewer fixed point evaluations than does simple
    iteration. This same problem is used in an original SQUAREM unit test and is the first one discussed in Varadhan and
    Roland (2008).
    """
    options = {
        'tol': 1e-8,
        'max_evaluations': 100,
        'scheme': scheme
    }

    # define the frequency data and the contraction mapping
    y = np.array([162, 267, 271, 185, 111, 61, 27, 8, 3, 1])
    def contraction(x):
        i = np.arange(y.size)
        z = np.divide(
            x[0] * np.exp(-x[1]) * (x[1] ** i),
            x[0] * np.exp(-x[1]) * (x[1] ** i) + (1 - x[0]) * np.exp(-x[2]) * (x[2] ** i)
        )
        return np.array([
            (y * z).sum() / y.sum(),
            (y * i * z).sum() / (y * z).sum(),
            (y * i * (1 - z)).sum() / (y * (1 - z)).sum()
        ])

    # solve the problem with SQUAREM and verify that the solution is reasonably close to the true solution
    start_values = np.array([0.2, 2.5, 1.5])
    exact = np.array([0.6401146029910, 2.6634043566619, 1.2560951012662])
    computed = Iteration('squarem', options)._iterate(contraction, start_values)[0]
    np.testing.assert_allclose(exact, computed, rtol=0, atol=1e-5)

    # verify that many more iterations would be needed to solve the problem with simple iteration
    del options['scheme']
    options['max_evaluations'] *= 10
    converged = Iteration('simple', options)._iterate(contraction, start_values)[1]
    assert not converged
