"""Tests of construction of nodes and weights for integration."""

import pytest
import numpy as np

from pyblp import Integration


@pytest.mark.parametrize(['dimensions', 'specification', 'size', 'naive_specification'], [
    pytest.param(1, 'monte_carlo', 100000, None, id="1D Monte Carlo"),
    pytest.param(2, 'monte_carlo', 150000, None, id="2D Monte Carlo"),
    pytest.param(1, 'product', 5, 'monte_carlo', id="1D product rule and Monte Carlo"),
    pytest.param(5, 'product', 2, 'monte_carlo', id="5D product rule and Monte Carlo"),
    pytest.param(1, 'nested_product', 6, 'monte_carlo', id="1D nested product rule and Monte Carlo"),
    pytest.param(7, 'nested_product', 7, 'monte_carlo', id="7D nested product rule and Monte Carlo"),
    pytest.param(1, 'grid', 4, 'monte_carlo', id="1D sparse grid and Monte Carlo"),
    pytest.param(6, 'grid', 8, 'monte_carlo', id="6D sparse grid and Monte Carlo"),
    pytest.param(1, 'nested_grid', 3, 'monte_carlo', id="1D nested sparse grid and Monte Carlo"),
    pytest.param(4, 'nested_grid', 9, 'monte_carlo', id="4D nested sparse grid and Monte Carlo")
])
def test_hermite_integral(dimensions, specification, size, naive_specification):
    """Test if approximations of a simple Gauss-Hermite integral (product of squared variables of integration with
    respect to the standard normal density) are reasonably correct. Then, if a naive specification is given, tests that
    it performs worse, even with an order of magnitude more nodes.
    """
    integral = lambda n, w: w.T @ (n ** 2).prod(axis=1)
    nodes, weights = Integration(specification, size, seed=0)._build(dimensions)
    simulated = integral(nodes, weights)
    np.testing.assert_allclose(simulated, 1, rtol=0, atol=0.01)
    if naive_specification:
        naive = integral(*Integration(naive_specification, 10 * weights.size, seed=0)._build(dimensions))
        np.testing.assert_array_less(np.linalg.norm(simulated - 1), np.linalg.norm(naive - 1))


@pytest.mark.parametrize('dimensions', [pytest.param(1, id="1D"), pytest.param(3, id="3D"), pytest.param(6, id="6D")])
@pytest.mark.parametrize(['specification', 'size'], [
    pytest.param('monte_carlo', 100, id="small Monte Carlo"),
    pytest.param('monte_carlo', 300, id="large Monte Carlo"),
    pytest.param('product', 1, id="small product rule"),
    pytest.param('product', 7, id="large product rule"),
    pytest.param('nested_product', 1, id="small nested product rule"),
    pytest.param('nested_product', 4, id="large nested product rule"),
    pytest.param('grid', 1, id="small sparse grid"),
    pytest.param('grid', 8, id="large sparse grid"),
    pytest.param('nested_grid', 1, id="small nested sparse grid"),
    pytest.param('nested_grid', 6, id="large nested sparse grid")
])
def test_weight_sums(dimensions, specification, size):
    """Test that weights sum to one."""
    weights = Integration(specification, size)._build(dimensions)[1]
    np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-12)


@pytest.mark.usefixtures('nwspgr')
def test_nwspgr(nwspgr):
    """Replicate output from the Matlab function nwspgr by Florian Heiss and Viktor Winschel."""
    dimensions, level, nested, expected_nodes, expected_weights = nwspgr
    nodes, weights = Integration('nested_grid' if nested else 'grid', level)._build(dimensions)
    np.testing.assert_allclose(nodes, expected_nodes, rtol=0, atol=1e-9)
    np.testing.assert_allclose(weights, expected_weights, rtol=0, atol=1e-9)
