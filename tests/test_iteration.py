"""Tests of fixed point iteration routines."""

import numpy as np
import pytest

from pyblp import Iteration
from pyblp.configurations.iteration import ContractionResults
from pyblp.utilities.basics import Array, Options


@pytest.mark.parametrize(['method', 'method_options'], [
    pytest.param('simple', {}, id="Simple"),
    pytest.param('simple', {'norm': lambda x: np.linalg.norm(x, np.inf)}, id="simple with infinity norm"),
    pytest.param('broyden1', {}, id="Broyden 1"),
    pytest.param('broyden2', {}, id="Broyden 2"),
    pytest.param('anderson', {}, id="Anderson"),
    pytest.param('diagbroyden', {}, id="diagonal Broyden"),
    pytest.param('krylov', {}, id="Krylov"),
    pytest.param('df-sane', {}, id="DF-SANE"),
    pytest.param('squarem', {'scheme': 1, 'step_min': 0.9, 'step_max': 1.1, 'step_factor': 3.0}, id="SQUAREM S1"),
    pytest.param('squarem', {'scheme': 2, 'step_min': 0.8, 'step_max': 1.2, 'step_factor': 4.0}, id="SQUAREM S2"),
    pytest.param('squarem', {'scheme': 3, 'step_min': 0.7, 'step_max': 1.3, 'step_factor': 5.0}, id="SQUAREM S3"),
    pytest.param('hybr', {}, id="Powell hybrid method"),
    pytest.param('lm', {}, id="Levenberg-Marquardt"),
    pytest.param('return', {}, id="Return")
])
@pytest.mark.parametrize('compute_jacobian', [
    pytest.param(True, id="analytic Jacobian"),
    pytest.param(False, id="no analytic Jacobian")
])
@pytest.mark.parametrize('universal_display', [
    pytest.param(True, id="universal display"),
    pytest.param(False, id="no universal display")
])
@pytest.mark.parametrize('use_weights', [
    pytest.param(True, id="weights"),
    pytest.param(False, id="no weights")
])
def test_scipy(
        method: str, method_options: Options, compute_jacobian: bool, universal_display: bool,
        use_weights: bool) -> None:
    """Test that the solution to the example fixed point problem from scipy.optimize.fixed_point is reasonably close to
    the exact solution. Also verify that the configuration can be formatted.
    """
    def contraction(x: Array, iterations: int, evaluations: int) -> ContractionResults:
        """Evaluate the contraction."""
        assert evaluations >= iterations >= 0
        c1 = np.array([10, 12])
        c2 = np.array([3, 5])
        x0, x = x, np.sqrt(c1 / (x + c2))
        weights = np.ones_like(x) if use_weights else None
        jacobian = -0.5 * np.eye(2) * x / (x0 + c2) if compute_jacobian else None
        return x, weights, jacobian

    # simple methods do not accept an analytic Jacobian
    if compute_jacobian and method not in {'hybr', 'lm'}:
        return pytest.skip("This method does not accept an analytic Jacobian.")

    # initialize the configuration and test that it can be formatted
    iteration = Iteration(method, method_options, compute_jacobian, universal_display)
    assert str(iteration)

    # define the exact solution
    exact_values = np.array([1.4920333, 1.37228132])

    # test that the solution is reasonably close (use the exact values if the iteration routine will just return them)
    start_values = exact_values if method == 'return' else np.ones_like(exact_values)
    computed_values, stats = iteration._iterate(start_values, contraction)
    assert stats.converged
    np.testing.assert_allclose(exact_values, computed_values, rtol=0, atol=1e-5)


@pytest.mark.parametrize('scheme', [pytest.param(1, id="S1"), pytest.param(2, id="S2"), pytest.param(3, id="S3")])
def test_hasselblad(scheme: int) -> None:
    """Test that the solution to the fixed point problem from Hasselblad (1969) is reasonably close to the exact
    solution and that SQUAREM takes at least an order of magnitude fewer fixed point evaluations than does simple
    iteration. This same problem is used in an original SQUAREM unit test and is the first one discussed in Varadhan and
    Roland (2008).
    """
    def contraction(x: Array, iterations: int, evaluations: int) -> ContractionResults:
        """Evaluate the contraction."""
        assert evaluations >= iterations >= 0
        y = np.array([162, 267, 271, 185, 111, 61, 27, 8, 3, 1])
        i = np.arange(y.size)
        z = np.divide(
            x[0] * np.exp(-x[1]) * x[1]**i,
            x[0] * np.exp(-x[1]) * x[1]**i + (1 - x[0]) * np.exp(-x[2]) * x[2]**i
        )
        x = np.array([
            (y * z).sum() / y.sum(),
            (y * i * z).sum() / (y * z).sum(),
            (y * i * (1 - z)).sum() / (y * (1 - z)).sum()
        ])
        return x, None, None

    # solve the problem with SQUAREM and verify that the solution is reasonably close to the true solution
    method_options = {
        'atol': 1e-8,
        'max_evaluations': 100,
        'scheme': scheme,
    }
    initial_values = np.array([0.2, 2.5, 1.5])
    exact_values = np.array([0.6401146029910, 2.6634043566619, 1.2560951012662])
    computed_values = Iteration('squarem', method_options)._iterate(initial_values, contraction)[0]
    np.testing.assert_allclose(exact_values, computed_values, rtol=0, atol=1e-5)

    # verify that many more iterations would be needed to solve the problem with simple iteration
    del method_options['scheme']
    method_options['max_evaluations'] *= 10
    stats = Iteration('simple', method_options)._iterate(initial_values, contraction)[1]
    assert not stats.converged
