"""Tests of optimization routines."""

from typing import Callable, Tuple, Union

import numpy as np
import pytest
import scipy.optimize

from pyblp import Optimization, options
from pyblp.utilities.basics import Array, Options


@pytest.mark.parametrize(['lb', 'ub'], [
    pytest.param(-np.inf, np.inf, id="unbounded"),
    pytest.param(-np.inf, 1, id="bounded above"),
    pytest.param(-1, np.inf, id="bounded below"),
    pytest.param(-1, 1, id="bounded above and below")
])
@pytest.mark.parametrize(['method', 'method_options'], [
    pytest.param('knitro', {'algorithm': 1}, id="Knitro algorithm 1"),
    pytest.param('knitro', {'algorithm': 2}, id="Knitro algorithm 2"),
    pytest.param('knitro', {'algorithm': 3}, id="Knitro algorithm 3"),
    pytest.param('knitro', {'algorithm': 4}, id="Knitro algorithm 4"),
    pytest.param('knitro', {'hessopt': 2}, id="Knitro hessopt 2"),
    pytest.param('knitro', {'hessopt': 3}, id="Knitro hessopt 3"),
    pytest.param('knitro', {'hessopt': 4}, id="Knitro hessopt 4"),
    pytest.param('slsqp', {}, id="SLSQP"),
    pytest.param('l-bfgs-b', {}, id="L-BFGS-B"),
    pytest.param('tnc', {}, id="TNC"),
    pytest.param('nelder-mead', {}, id="Nelder-Mead"),
    pytest.param('powell', {}, id="Powell"),
    pytest.param('cg', {}, id="CG"),
    pytest.param('bfgs', {}, id="BFGS"),
    pytest.param('newton-cg', {}, id="Newton-CG"),
    pytest.param('return', {}, id="Return"),
    pytest.param(lambda i, b, f, _, **o: (scipy.optimize.minimize(f, i, bounds=b, **o).x, True), {}, id="custom")
])
@pytest.mark.parametrize('compute_gradient', [
    pytest.param(True, id="analytic gradient"),
    pytest.param(False, id="no analytic gradient")
])
@pytest.mark.parametrize('universal_display', [
    pytest.param(True, id="universal display"),
    pytest.param(False, id="default display")
])
def test_entropy(
        lb: float, ub: float, method: Union[str, Callable], method_options: Options, compute_gradient: bool,
        universal_display: bool) -> None:
    """Test that solutions to the entropy maximization problem from Berger, Pietra, and Pietra (1996) are reasonably
    close to the exact solution (this is based on a subset of testing methods from scipy.optimize.tests.test_optimize).
    """
    def objective_function(x: Array) -> Union[Array, Tuple[Array, Array]]:
        """Evaluate the objective."""
        K = np.array([1, 0.3, 0.5])
        F = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0]])
        log_Z = np.log(np.exp(F @ x).sum())
        p = np.exp(F @ x - log_Z)
        objective = log_Z - K @ x
        gradient = F.T @ p - K
        return (objective, gradient) if compute_gradient else objective

    # simple methods do not accept an analytic gradient
    if compute_gradient is True and method in {'nelder-mead', 'powell'}:
        return

    # Newton CG requires an analytic gradient
    if compute_gradient is False and method == 'newton-cg':
        return

    # the custom method needs to know if an analytic gradient will be computed
    if callable(method):
        method_options = method_options.copy()
        method_options['jac'] = compute_gradient

    # skip optimization methods that haven't been configured properly
    try:
        optimization = Optimization(method, method_options, compute_gradient, universal_display)
    except OSError as exception:
        return pytest.skip(f"Failed to use the {method} method in this environment: {exception}.")

    # test that the configuration can be formatted
    assert str(optimization)

    # define the exact solution
    exact_values = np.array([0, -0.524869316, 0.487525860], options.dtype)

    # estimate the solution (use the exact values if the optimization routine will just return them)
    start_values = exact_values if method == 'return' else np.array([0, 0, 0], options.dtype)
    bounds = 3 * [(lb, ub)]
    estimated_values, converged = optimization._optimize(start_values, bounds, lambda x, *_: objective_function(x))[:2]
    assert converged

    # test that the estimated objective is reasonably close to the exact objective
    exact_results = objective_function(exact_values)
    estimated_results = objective_function(estimated_values)
    exact_objective = exact_results[0] if compute_gradient else exact_results
    estimated_objective = estimated_results[0] if compute_gradient else estimated_results
    np.testing.assert_allclose(estimated_objective, exact_objective, rtol=1e-5, atol=0)
