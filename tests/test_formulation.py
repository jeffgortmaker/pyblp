"""Tests of formulation of data matrices."""

import copy
import itertools
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence, Type

import numpy as np
import patsy
import pytest

from pyblp import Formulation
from pyblp.utilities.basics import Array, Data


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
        ['0 + x + I(1)', 'x + I(1) - 1'],
        lambda d: [d['x'], d['1']],
        lambda d: [d['1'], d['0']],
        id="continuous variable before intercept"
    ),
    pytest.param(
        ['0 + x + C(x)'],
        lambda d: [d['x']] + [d[f'x[{v}]'] for v in np.unique(d['x'])],
        lambda d: [d['1']] + [d['0']] * d['x'].size,
        id="discretized continuous variable"
    ),
    pytest.param(
        ['0 + a', '0 + C(a)', 'C(a) - 1'],
        lambda d: [d["a['a1']"], d["a['a2']"]],
        lambda d: [d['0'], d['0']],
        id="full-rank coding of categorical variable"
    ),
    pytest.param(
        ['a', 'C(a)', '1 + C(a)'],
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
def test_matrices(
        formula_data: Data, formulas: Iterable[str], build_columns: Callable[[Mapping[str, Array]], Sequence[Array]],
        build_derivatives: Callable[[Mapping[str, Array]], Sequence[Array]]) -> None:
    """Test that equivalent formulas build columns and derivatives as expected. Take derivatives with respect to x."""

    # construct convenience columns of ones and zeros
    ones = np.ones_like(formula_data['x'])
    zeros = np.zeros_like(formula_data['x'])

    # build columns and derivatives for each formula, making sure that it can be formatted
    for formula in formulas:
        formulation = Formulation(formula)
        assert str(formulation)
        matrix, column_formulations, underlying_data = formulation._build_matrix(formula_data)
        evaluated_matrix = np.column_stack([ones * f.evaluate(underlying_data) for f in column_formulations])
        derivatives = np.column_stack([ones * f.evaluate_derivative('x', underlying_data) for f in column_formulations])

        # build expected columns and derivatives
        supplemented_data = {'1': ones, '0': zeros, **underlying_data}
        expected_matrix = np.column_stack(build_columns(supplemented_data))
        expected_derivatives = np.column_stack(build_derivatives(supplemented_data))

        # compare columns and derivatives
        np.testing.assert_allclose(matrix, expected_matrix, rtol=0, atol=1e-14, err_msg=formula)
        np.testing.assert_allclose(matrix, evaluated_matrix, rtol=0, atol=1e-14, err_msg=formula)
        np.testing.assert_allclose(derivatives, expected_derivatives, rtol=0, atol=1e-14, err_msg=formula)


@pytest.mark.usefixtures('formula_data')
@pytest.mark.parametrize(['formulas', 'build_columns'], [
    pytest.param(
        ['a', 'C(a)', '1 + a', '0 + a', 'a - 1', 'a + 1 + 1'],
        lambda d: [d['a']],
        id="categorical variable"
    ),
    pytest.param(
        ['C(x)'],
        lambda d: [d['x']],
        id="discretized continuous variable"
    ),
    pytest.param(
        ['a * b', '(a + b) ** 2', 'a + b + a:b'],
        lambda d: [d['a'], d['b'], d['ab']],
        id="product short-hands"
    ),
    pytest.param(
        ['a / b', 'a * b - b', 'a + a:b'],
        lambda d: [d['a'], d['ab']],
        id="division short-hand"
    ),
    pytest.param(
        ['a + a:b:c + a:C(x):C(y)'],
        lambda d: [d['a'], d['abc'], d['axy']],
        id="interactions"
    )
])
def test_ids(
        formula_data: Data, formulas: Iterable[str],
        build_columns: Callable[[Mapping[str, Array]], Sequence[Array]]) -> None:
    """Test that equivalent formulas build IDs as expected."""

    # create convenience columns of tuples of categorical variables
    formula_data = copy.deepcopy(formula_data)
    for (key1, values1), (key2, values2), (key3, values3) in itertools.product(formula_data.items(), repeat=3):
        key12 = f'{key1}{key2}'
        key123 = f'{key1}{key2}{key3}'
        if key12 not in formula_data:
            values12 = np.empty_like(values1, np.object_)
            values12[:] = list(zip(values1, values2))
            formula_data[key12] = values12
        if key123 not in formula_data:
            values123 = np.empty_like(values1, np.object_)
            values123[:] = list(zip(values1, values2, values3))
            formula_data[key123] = values123

    # build and compare columns for each formula, making sure that it can be formatted
    for absorb in formulas:
        formulation = Formulation('x', absorb)
        assert str(formulation)
        ids = formulation._build_ids(formula_data)
        expected_ids = np.column_stack(build_columns(formula_data))
        np.testing.assert_array_equal(ids, expected_ids, err_msg=absorb)


@pytest.mark.usefixtures('formula_data')
@pytest.mark.parametrize(['exception', 'formula', 'absorb'], [
    pytest.param(TypeError, None, None, id="None"),
    pytest.param(TypeError, 1, None, id="integer"),
    pytest.param(TypeError, '', 1, id="absorbed integer"),
    pytest.param(patsy.PatsyError, 'x ~ y', None, id="left-hand side"),
    pytest.param(patsy.PatsyError, '', 'a ~ b', id="absorbed left-hand side"),
    pytest.param(patsy.PatsyError, 'I(1)', None, id="two intercepts"),
    pytest.param(patsy.PatsyError, '', 'a + I(1)', id="absorbed intercept"),
    pytest.param(patsy.PatsyError, '', 'x', id="absorbed continuous variable"),
    pytest.param(patsy.PatsyError, '', 'x:a', id="absorbed continuous-categorical interaction"),
    pytest.param(patsy.PatsyError, 'C(a, levels=["a1", "a2"])', None, id="categorical marker arguments"),
    pytest.param(patsy.PatsyError, '', 'C(a, levels=["a1", "a2"])', id="absorbed categorical marker arguments"),
    pytest.param(patsy.PatsyError, 'I(x + a)', None, id="categorical variable inside identity function"),
    pytest.param(patsy.PatsyError, 'I(C(a))', None, id="categorical marker inside identity function"),
    pytest.param(patsy.PatsyError, 'C(C(a))', None, id="nested categorical marker"),
    pytest.param(patsy.PatsyError, '', 'C(C(a))', id="absorbed nested categorical marker"),
    pytest.param(patsy.PatsyError, 'log(a)', None, id="log of categorical variable"),
    pytest.param(patsy.PatsyError, 'abs(x)', None, id="unsupported function"),
    pytest.param(patsy.PatsyError, '', 'log(x)', id="absorbed function"),
    pytest.param(patsy.PatsyError, 'g', None, id="bad variable name"),
    pytest.param(patsy.PatsyError, '', 'g', id="bad absorbed variable name"),
    pytest.param(patsy.PatsyError, 'x ^ y', None, id="unsupported patsy operator"),
    pytest.param(patsy.PatsyError, '', 'x ^ y', id="absorbed unsupported patsy operator"),
    pytest.param(patsy.PatsyError, 'I(x | y)', None, id="unsupported SymPy operator"),
    pytest.param(patsy.PatsyError, 'log(-x)', None, id="logarithm of negative values"),
    pytest.param(patsy.PatsyError, 'Q("a")', None, id="unsupported patsy quoting"),
    pytest.param(patsy.PatsyError, '', 'Q("a")', id="absorbed unsupported patsy quoting")
])
def test_invalid_formulas(formula_data: Data, exception: Type[Exception], formula: Any, absorb: Any) -> None:
    """Test that an invalid formula gives rise to an exception."""
    try:
        formulation = Formulation(formula, absorb)
        formulation._build_matrix(formula_data)
        if absorb is not None:
            formulation._build_ids(formula_data)
    except exception:
        print(traceback.format_exc())
        return
    raise RuntimeError(f"Successful formulation: {formulation}.")
