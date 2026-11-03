"""Regression tests for fixed-point dispatch and contraction Jacobians.

Run against an installed PyBLP checkout; these are not mocked unit tests.
"""
import numpy as np
import pytest
import pyblp


def demand_fixture(nested=False, epsilon=1.0):
    rng = np.random.default_rng(90210)
    T, J, I = 8, 4, 30
    N, sigma, rho = T * J, 0.7, 0.35
    ids = np.repeat(np.arange(T), J)
    prices = rng.uniform(0.3, 2.2, N)
    X = np.column_stack([np.ones(N), prices])
    delta = X @ np.array([-1.0, -1.2]) + 0.15 * rng.standard_normal(N)
    nodes = rng.normal(size=(T * I, 1))
    agent_data = {
        'market_ids': np.repeat(np.arange(T), I),
        'nodes': nodes,
        'weights': np.full(T * I, 1 / I),
    }
    shares = np.empty(N)
    nests = np.arange(N) % 2
    for t in range(T):
        products = slice(t * J, (t + 1) * J)
        agents = slice(t * I, (t + 1) * I)
        utility = (
            delta[products, None] +
            sigma * prices[products, None] * nodes[agents].T
        ) / epsilon
        if not nested:
            exponentials = np.exp(utility)
            probabilities = exponentials / (1 + exponentials.sum(axis=0))
        else:
            exponentials = np.exp(utility / (1 - rho))
            inclusive = np.vstack([
                exponentials[nests[products] == h].sum(axis=0)
                for h in [0, 1]
            ])
            conditionals = exponentials / inclusive[nests[products]]
            marginals = inclusive**(1 - rho)
            marginals /= 1 + marginals.sum(axis=0)
            probabilities = conditionals * marginals[nests[products]]
        shares[products] = probabilities.mean(axis=1)

    product_data = {
        'market_ids': ids, 'prices': prices, 'shares': shares,
        'demand_instruments0': prices,
    }
    if nested:
        product_data['nesting_ids'] = nests
    problem = pyblp.Problem(
        [pyblp.Formulation('1 + prices'), pyblp.Formulation('0 + prices')],
        product_data, agent_data=agent_data, epsilon_scale=epsilon,
    )
    solve_options = dict(
        sigma=np.array([[sigma]]),
        sigma_bounds=(np.array([[sigma]]), np.array([[sigma]])),
        method='1s',
        check_optimality='gradient',
        initial_update=False,
        error_behavior='raise',
        iteration=pyblp.Iteration('squarem', {'atol': 1e-13}),
    )
    if nested:
        solve_options.update(rho=rho, rho_bounds=(rho, rho))
    return problem, solve_options, delta[:, None]


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('fp_type', [
    'linear', 'safe_linear', 'nonlinear', 'safe_nonlinear',
])
def test_share_inversion(nested, fp_type):
    problem, options, truth = demand_fixture(nested=nested)
    results = problem.solve(**options, fp_type=fp_type)
    np.testing.assert_allclose(results.delta, truth, rtol=0, atol=2e-9)
    np.testing.assert_allclose(
        results.compute_shares(), problem.products.shares,
        rtol=2e-9, atol=2e-13,
    )


@pytest.mark.parametrize('epsilon', [0.2, 0.5, 2.0])
def test_scaled_utility_analytic_jacobian(epsilon):
    problem, options, truth = demand_fixture(epsilon=epsilon)
    options['iteration'] = pyblp.Iteration(
        'lm', {'xtol': 1e-12, 'ftol': 1e-12, 'gtol': 1e-12},
        compute_jacobian=True,
    )
    results = problem.solve(**options)
    np.testing.assert_allclose(results.delta, truth, rtol=0, atol=3e-8)
    np.testing.assert_allclose(
        np.log(results.compute_shares()),
        np.log(problem.products.shares), rtol=0, atol=2e-9,
    )
