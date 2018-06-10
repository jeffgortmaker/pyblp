"""Fixtures used by tests."""

import os
import re
from pathlib import Path

import patsy
import pytest
import scipy.io
import numpy as np
import numpy.lib.recfunctions

from .data import TEST_DATA_PATH
from pyblp.data import BLP_PRODUCTS_LOCATION, BLP_AGENTS_LOCATION
from pyblp import (
    Problem, Simulation, Formulation, Integration, options, build_id_data, build_ownership, build_blp_instruments
)


@pytest.fixture(autouse=True)
def configure():
    """Configure NumPy so that it raises all warnings as exceptions, and, if a DTYPE environment variable is set in this
    testing environment that is different from the default data type, use it for all numeric calculations.
    """
    old_error = np.seterr(all='raise')
    old_dtype = options.dtype
    dtype_string = os.environ.get('DTYPE')
    if dtype_string:
        options.dtype = np.dtype(dtype_string)
        if np.finfo(options.dtype).dtype == old_dtype:
            pytest.skip(f"The {dtype_string} data type is the same as the default one in this environment.")
    yield
    options.dtype = old_dtype
    np.seterr(**old_error)


@pytest.fixture
def small_simulation():
    """Solve a simulation with two markets, linear prices, a nonlinear characteristic, a cost characteristic, and an
    acquisition.
    """
    simulation = Simulation(
        product_formulations=(
            Formulation('0 + prices'),
            Formulation('0 + x'),
            Formulation('0 + a')
        ),
        beta=-5,
        sigma=1,
        gamma=2,
        product_data=build_id_data(T=2, J=18, F=3, mergers=[{1: 0}]),
        integration=Integration('product', 3),
        xi_variance=0.001,
        omega_variance=0.001,
        correlation=0.7,
        seed=0
    )
    return simulation, simulation.solve()


@pytest.fixture
def medium_simulation():
    """Solve a simulation with four markets, a nonlinear/cost constant, two linear characteristics, two cost
    characteristics, a demographic interacted with prices, a double acquisition, and a non-standard ownership structure.
    """
    id_data = build_id_data(T=4, J=25, F=6, mergers=[{f: 2 for f in range(2)}])
    simulation = Simulation(
        product_formulations=(
            Formulation('0 + x + y'),
            Formulation('1 + prices'),
            Formulation('1 + a + b')
        ),
        beta=[2, 1],
        sigma=[
            [0.5, 0],
            [0,   0],
        ],
        gamma=[1, 1, 2],
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'ownership': build_ownership(id_data, lambda f, g: 1 if f == g else (0.1 if f > 3 and g > 3 else 0))
        },
        agent_formulation=Formulation('0 + f'),
        pi=[
            [ 0],
            [-3]
        ],
        integration=Integration('product', 4),
        xi_variance=0.0001,
        omega_variance=0.0001,
        correlation=0.8,
        seed=1
    )
    return simulation, simulation.solve()


@pytest.fixture
def large_simulation():
    """Solve a simulation with ten markets, linear/nonlinear prices, a linear constant, a cost/linear/nonlinear
    characteristic, another three cost characteristics, another two linear characteristics, demographics interacted with
    prices and the cost/linear/nonlinear characteristic, dense parameter matrices, an acquisition, a triple acquisition,
    and a log-linear cost specification.
    """
    simulation = Simulation(
        product_formulations=(
            Formulation('1 + prices + x + y + z'),
            Formulation('0 + prices + x'),
            Formulation('0 + x + a + b + c')
        ),
        beta=[1, -6, 1, 2, 3],
        sigma=[
            [1, -0.1],
            [0,  2  ]
        ],
        gamma=[0.1, 0.2, 0.3, 0.5],
        product_data=build_id_data(T=10, J=20, F=9, mergers=[{f: 4 + int(f > 0) for f in range(4)}]),
        agent_formulation=Formulation('0 + f + g'),
        pi=[
            [1, 0],
            [0, 2]
        ],
        integration=Integration('product', 4),
        xi_variance=0.00001,
        omega_variance=0.00001,
        correlation=0.9,
        linear_costs=False,
        seed=2
    )
    return simulation, simulation.solve()


@pytest.fixture(params=[
    pytest.param(['small', False], id="small without supply"),
    pytest.param(['small', True], id="small with supply"),
    pytest.param(['medium', False], id="medium without supply"),
    pytest.param(['medium', True], id="medium with supply"),
    pytest.param(['large', False], id="large without supply"),
    pytest.param(['large', True], id="large with supply")
])
def simulated_problem(request):
    """Configure and solve a simulated problem, either with or without supply-side data."""
    name, supply = request.param
    simulation, product_data = request.getfixturevalue(f'{name}_simulation')
    product_formulations = simulation.product_formulations
    if not supply:
        product_data = np.lib.recfunctions.drop_fields(product_data, 'supply_instruments')
        product_formulations = product_formulations[:2]
    problem = Problem(product_formulations, product_data, simulation.agent_formulation, simulation.agent_data)
    results = problem.solve(simulation.sigma, simulation.pi, steps=1, linear_costs=simulation.linear_costs)
    return simulation, product_data, problem, results


@pytest.fixture
def knittel_metaxoglou_2014():
    """Configure the example automobile problem from Knittel and Metaxoglou (2014) and load initial parameter values and
    estimates created by replication code.

    The replication code was modified to output a Matlab data file for the automobile dataset, which contains the
    results of one round of Knitro optimization and post-estimation calculations. The replication code was kept mostly
    intact, but was modified slightly in the following ways:

        - Tolerance parameters, Knitro optimization parameters, and starting values for sigma were all configured.
        - A bug in the code's computation of the BLP instruments was fixed. When creating a vector of "other" and
          "rival" sums, the code did not specify a dimension over which to sum, which created problems with one-
          dimensional vectors. A dimension of 1 was added to both sum commands.
        - Delta was initialized as the solution to the Logit model.
        - After estimation, the objective was called again at the optimal parameters to re-load globals at the optimal
          parameter values.
        - Before being saved to a Matlab data file, matrices were renamed and reshaped.

    """
    product_data = np.recfromcsv(BLP_PRODUCTS_LOCATION)
    product_data = {n: product_data[n] for n in product_data.dtype.names}
    product_data['demand_instruments'] = build_blp_instruments(Formulation('hpwt + air + mpg + space'), product_data)
    problem = Problem(
        product_formulations=(
            Formulation('0 + prices + I(1) + hpwt + air + mpg + space'),
            Formulation('0 + prices + I(1) + hpwt + air + mpg')
        ),
        product_data=product_data,
        agent_data=np.recfromcsv(BLP_AGENTS_LOCATION)
    )
    return scipy.io.loadmat(str(TEST_DATA_PATH / 'knittel_metaxoglou_2014.mat'), {'problem': problem})


@pytest.fixture(params=[pytest.param(p, id=p.name) for p in Path(TEST_DATA_PATH / 'nwspgr').iterdir()])
def nwspgr(request):
    """Load a sample of sparse grids of nodes and weights constructed according to the Gauss-Hermite quadrature rule and
    its nested analog, which were computed by the Matlab function nwspgr by Florian Heiss and Viktor Winschel.
    """
    rule, dimensions, level = re.search(r'(GQN|KPN)_d(\d+)_l(\d+)', request.param.name).groups()
    matrix = np.atleast_2d(np.genfromtxt(request.param, delimiter=','))
    nested = rule == 'KPN'
    nodes = matrix[:, :-1]
    weights = matrix[:, -1]
    return int(dimensions), int(level), nested, nodes, weights


@pytest.fixture(params=[pytest.param(1, id="1 observation"), pytest.param(10, id="10 observations")])
def formula_data(request):
    """Simulate patsy demo data with two-level categorical variables and varying numbers of observations."""
    raw_data = patsy.user_util.demo_data('a', 'b', 'c', 'x', 'y', 'z', nlevels=2, min_rows=request.param)
    return {k: np.array(v) if isinstance(v[0], str) else np.abs(v) for k, v in raw_data.items()}
