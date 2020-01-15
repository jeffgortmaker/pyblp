"""Tests of construction of nodes and weights for integration."""

import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import scipy.io

from pyblp import Integration


@pytest.mark.parametrize(['dimensions', 'specification', 'size', 'naive_specification'], [
    pytest.param(1, 'monte_carlo', 100000, None, id="1D Monte Carlo"),
    pytest.param(2, 'monte_carlo', 150000, None, id="2D Monte Carlo"),
    pytest.param(1, 'halton', 50000, None, id="1D Halton"),
    pytest.param(2, 'halton', 100000, None, id="2D Halton"),
    pytest.param(1, 'lhs', 100000, None, id="1D LHS"),
    pytest.param(3, 'lhs', 200000, None, id="3D LHS"),
    pytest.param(1, 'mlhs', 50000, None, id="1D MLHS"),
    pytest.param(3, 'mlhs', 100000, None, id="3D MLHS"),
    pytest.param(1, 'product', 6, 'monte_carlo', id="1D product rule and Monte Carlo"),
    pytest.param(3, 'product', 3, 'monte_carlo', id="3D product rule and Monte Carlo"),
    pytest.param(5, 'product', 4, 'monte_carlo', id="5D product rule and Monte Carlo"),
    pytest.param(1, 'nested_product', 7, 'monte_carlo', id="1D nested product rule and Monte Carlo"),
    pytest.param(2, 'nested_product', 3, 'monte_carlo', id="2D nested product rule and Monte Carlo"),
    pytest.param(6, 'nested_product', 2, 'monte_carlo', id="6D nested product rule and Monte Carlo"),
    pytest.param(1, 'grid', 4, 'monte_carlo', id="1D sparse grid and Monte Carlo"),
    pytest.param(6, 'grid', 8, 'monte_carlo', id="6D sparse grid and Monte Carlo"),
    pytest.param(1, 'nested_grid', 3, 'monte_carlo', id="1D nested sparse grid and Monte Carlo"),
    pytest.param(4, 'nested_grid', 9, 'monte_carlo', id="4D nested sparse grid and Monte Carlo"),
])
def test_hermite_integral(dimensions: int, specification: str, size: int, naive_specification: str) -> None:
    """Test if approximations of a simple Gauss-Hermite integral (product of squared variables of integration with
    respect to the standard normal density) are reasonably correct. Then, if a naive specification is given, tests that
    it performs worse, even with an order of magnitude more nodes.
    """
    integral = lambda n, w: w.T @ (n**2).prod(axis=1)
    specification_options = {'seed': 0}
    nodes, weights = Integration(specification, size, specification_options)._build(dimensions)
    simulated = integral(nodes, weights)
    np.testing.assert_allclose(simulated, 1, rtol=0, atol=0.01)
    if naive_specification:
        naive = integral(*Integration(naive_specification, 10 * weights.size, specification_options)._build(dimensions))
        np.testing.assert_array_less(np.linalg.norm(simulated - 1), np.linalg.norm(naive - 1))


@pytest.mark.parametrize('dimensions', [
    pytest.param(1, id="1D"),
    pytest.param(2, id="2D"),
    pytest.param(3, id="3D"),
    pytest.param(6, id="6D"),
])
@pytest.mark.parametrize(['specification', 'size'], [
    pytest.param('monte_carlo', 100, id="small Monte Carlo"),
    pytest.param('monte_carlo', 300, id="large Monte Carlo"),
    pytest.param('halton', 10, id="small Monte Carlo"),
    pytest.param('halton', 20, id="large Monte Carlo"),
    pytest.param('product', 1, id="small product rule"),
    pytest.param('product', 7, id="large product rule"),
    pytest.param('nested_product', 1, id="small nested product rule"),
    pytest.param('nested_product', 4, id="large nested product rule"),
    pytest.param('grid', 1, id="small sparse grid"),
    pytest.param('grid', 8, id="large sparse grid"),
    pytest.param('nested_grid', 1, id="small nested sparse grid"),
    pytest.param('nested_grid', 6, id="large nested sparse grid"),
])
def test_weights_and_formatting(dimensions: int, specification: str, size: int) -> None:
    """Test that weights sum to one and that the configurations can be formatted."""
    integration = Integration(specification, size)
    assert str(integration)
    weights = integration._build(dimensions)[1]
    np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-12)


@pytest.mark.parametrize('dimensions', [
    pytest.param(1, id="1D"),
    pytest.param(3, id="3D"),
])
@pytest.mark.parametrize('level', [
    pytest.param(1, id="L1"),
    pytest.param(3, id="L3"),
    pytest.param(4, id="L4"),
])
@pytest.mark.parametrize('nested', [
    pytest.param(True, id="nested"),
    pytest.param(False, id="not nested"),
])
def test_nwspgr(dimensions: int, level: int, nested: bool) -> None:
    """Compare with output from the Matlab function nwspgr by Florian Heiss and Viktor Winschel. The weights differ
    by floating point error because different sorting algorithms are used by Matlab and NumPy.
    """
    if shutil.which('matlab') is None:
        return pytest.skip("Failed to find a MATLAB executable in this environment.")
    with tempfile.NamedTemporaryFile() as handle:
        command = ';'.join([
            f"[nodes, weights] = nwspgr('{'KPN' if nested else 'GQN'}', {dimensions}, {level})",
            f"save('{handle.name}', 'nodes', 'weights')",
            'exit'
        ])
        subprocess.run(['matlab', '-nodesktop', '-nosplash', '-minimize', '-wait', '-r', f'"{command}"'])
        nwspgr = scipy.io.loadmat(handle.name)
    nodes, weights = Integration('nested_grid' if nested else 'grid', level)._build(dimensions)
    np.testing.assert_allclose(nwspgr['nodes'], np.c_[nodes], rtol=0, atol=0)
    np.testing.assert_allclose(nwspgr['weights'], np.c_[weights], rtol=0, atol=1e-14)
