"""Tests of optimization routines."""

import pytest
import numpy as np
import scipy.optimize

from pyblp import options
from pyblp.utilities import Optimization


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
def test_entropy(lb, ub, method, method_options, compute_gradient, universal_display):
    """Test that solutions to the entropy maximization problem from Berger, Pietra, and Pietra (1996) are reasonably
    close to the exact solution. Based on a subset of testing methods from scipy.optimize.tests.test_optimize.
    """

    # simple methods do not accept an analytic gradient
    if compute_gradient is True and method in {'nelder-mead', 'powell'}:
        return

    # Newton CG requires an analytic gradient
    if compute_gradient is False and method == 'newton-cg':
        return

    # the custom method needs to know if an analytic gradient will be computed
    if callable(method):
        method_options['jac'] = compute_gradient

    # define the objective function
    K = np.array([1, 0.3, 0.5])
    F = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0]])
    def objective_function(x):
        log_Z = np.log(np.exp(F @ x).sum())
        p = np.exp(F @ x - log_Z)
        objective = log_Z - K @ x
        gradient = F.T @ p - K
        return (objective, gradient) if compute_gradient else objective

    # estimate the solution
    optimization = Optimization(method, method_options, compute_gradient, universal_display)
    start_values = np.array([0, 0, 0], dtype=options.dtype)
    bounds = 3 * [(lb, ub)]
    estimated_values, converged = optimization._optimize(start_values, bounds, lambda x, *_: objective_function(x))[:2]
    assert converged

    # test that the estimated objective is reasonably close to the exact objective
    exact_values = np.array([0, -0.524869316, 0.487525860], dtype=options.dtype)
    exact_results = objective_function(exact_values)
    estimated_results = objective_function(estimated_values)
    exact_objective = exact_results[0] if compute_gradient else exact_results
    estimated_objective = estimated_results[0] if compute_gradient else estimated_results
    np.testing.assert_allclose(estimated_objective, exact_objective, rtol=1e-5, atol=0)
