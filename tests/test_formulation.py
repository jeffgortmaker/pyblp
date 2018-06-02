"""Tests of formulation of data matrices."""

import pytest
import numpy as np

from pyblp.utilities import Formulation


@pytest.mark.usefixtures('formula_data')
@pytest.mark.parametrize(['formulas', 'build_columns', 'build_derivatives'], [
    pytest.param(
        ['', '1', '0 + I(1)', '-1 + I(1)', 'I(1) - 1'],
        lambda d: [d['1']],
        lambda d: [d['0']],
        id="intercept"
    ),
    pytest.param(
        ['0 + x', 'I(x) - 1'],
        lambda d: [d['x']],
        lambda d: [d['1']],
        id="continuous variable"
    ),
    pytest.param(
        ['0 + x + C(x)'],
        lambda d: [d['x']] + [d[f'x[{v}]'] for v in np.unique(d['x'])],
        lambda d: [d['1']] + [d['0']] * d['x'].size,
        id="discretized continuous variable"
    ),
    pytest.param(
        ['0 + a', 'C(a) - 1'],
        lambda d: [d["a['a1']"], d["a['a2']"]],
        lambda d: [d['0'], d['0']],
        id="full-rank coding of categorical variable"
    ),
    pytest.param(
        ['a', '1 + C(a)'],
        lambda d: [d['1'], d["a['a2']"]],
        lambda d: [d['0'], d['0']],
        id="reduced-rank coding of categorical variable"
    ),
    pytest.param(
        ['0 + log(2 * x * y) + I(1 + x ** -0.5 * exp(y))'],
        lambda d: [np.log(2 * d['x'] * d['y']), 1 + d['x'] ** -0.5 * np.exp(d['y'])],
        lambda d: [1 / d['x'], -0.5 * d['x'] ** -1.5 * np.exp(d['y'])],
        id="functions"
    ),
    pytest.param(
        ['0 + x * y', '0 + (x + y) ** 2', '0 + x + y + x:y'],
        lambda d: [d['x'], d['y'], d['x'] * d['y']],
        lambda d: [d['1'], d['0'], d['y']],
        id="product short-hands"
    ),
    pytest.param(
        ['0 + x / y', '0 + x * y - y', '0 + x + x:y'],
        lambda d: [d['x'], d['x'] * d['y']],
        lambda d: [d['1'], d['y']],
        id="division short-hand"
    ),
    pytest.param(
        ['0 + a:b + a:x + x:y', '0 + C(a):C(b) + a:x + x:y'],
        lambda d: [
            d["a['a1']"] * d["b['b1']"], d["a['a2']"] * d["b['b1']"], d["a['a1']"] * d["b['b2']"],
            d["a['a2']"] * d["b['b2']"], d["a['a1']"] * d['x'], d["a['a2']"] * d['x'], d['x'] * d['y']
        ],
        lambda d: [d['0']] * 4 + [d["a['a1']"], d["a['a2']"], d['y']],
        id="interactions"
    )
])
def test_valid_formulas(formula_data, formulas, build_columns, build_derivatives):
    """Test that equivalent formulas build columns and derivatives as expected. Take derivatives with respect to x."""

    # construct convenience columns of ones and zeros
    ones = np.ones_like(formula_data['x'])
    zeros = np.zeros_like(formula_data['x'])

    # build columns and derivatives for each formula
    for formula in formulas:
        matrix, column_formulations, underlying_data = Formulation(formula)._build(formula_data)
        derivatives = np.column_stack([ones * f.evaluate_derivative('x', underlying_data) for f in column_formulations])

        # build expected columns and derivatives
        supplemented_data = {'1': ones, '0': zeros, **underlying_data}
        expected_matrix = np.column_stack(build_columns(supplemented_data))
        expected_derivatives = np.column_stack(build_derivatives(supplemented_data))

        # compare columns and derivatives
        np.testing.assert_allclose(matrix, expected_matrix, rtol=0, atol=1e-14, err_msg=formula)
        np.testing.assert_allclose(derivatives, expected_derivatives, rtol=0, atol=1e-14, err_msg=formula)
