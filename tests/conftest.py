"""Fixtures used by tests."""

import os
import re
from pathlib import Path

import pytest
import scipy.io
import numpy as np
import numpy.lib.recfunctions

from .data import TEST_DATA_PATH
from pyblp.utilities import extract_matrix
from pyblp.data import BLP_PRODUCTS_LOCATION, BLP_AGENTS_LOCATION, NEVO_PRODUCTS_LOCATION, NEVO_AGENTS_LOCATION
from pyblp import Problem, Simulation, Integration, options, build_id_data, build_indicators, build_blp_instruments


@pytest.fixture(autouse=True)
def configure():
    """Configure NumPy so that it raises all warnings as exceptions, and, if a DTYPE environment variable is set in this
    testing environment that is different from the default data type, use it for all numeric calculations.
    """
    np.seterr(all='raise')
    dtype_string = os.environ.get('DTYPE')
    if dtype_string:
        dtype = np.dtype(dtype_string)
        if np.finfo(dtype).dtype == options.dtype:
            pytest.skip(f"The {dtype_string} data type is the same as the default one in this environment.")
        options.dtype = dtype


@pytest.fixture
def small_simulation():
    """Solve a simulation with two markets, a cost characteristic, a nonlinear characteristic, and an acquisition."""
    simulation = Simulation(
        build_id_data(T=2, J=18, F=3, mergers=[{1: 0}]),
        Integration('product', 3),
        gamma=[0, 2, 0],
        beta=[-5, 0, 0, 0],
        sigma=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ],
        xi_variance=0.001,
        omega_variance=0.001,
        correlation=0.7,
        seed=0
    )
    return simulation, simulation.solve()


@pytest.fixture
def medium_simulation():
    """Solve a simulation with four markets, nonlinear prices, a cost/nonlinear constant, two cost characteristics, two
    linear characteristics, a demographic, and a double acquisition.
    """
    simulation = Simulation(
        build_id_data(T=4, J=25, F=6, mergers=[{f: 2 for f in range(2)}]),
        Integration('product', 4),
        gamma=[1, 1, 2, 0, 0],
        beta=[-5, 0, 0, 0, 2, 1],
        sigma=[
            [1, 0,   0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0],
            [0, 0,   0, 0, 0, 0],
            [0, 0,   0, 0, 0, 0],
            [0, 0,   0, 0, 0, 0],
            [0, 0,   0, 0, 0, 0]
        ],
        pi=[
            [1],
            [0],
            [0],
            [0],
            [0],
            [0]
        ],
        xi_variance=0.0001,
        omega_variance=0.0001,
        correlation=0.8,
        seed=1
    )
    return simulation, simulation.solve()


@pytest.fixture
def large_simulation():
    """Solve a simulation with ten markets, nonlinear prices, a cost/linear constant, a cost/linear/nonlinear
    characteristic, a cost characteristic, a linear characteristic, two demographics, dense parameter matrices, an
    acquisition, a triple acquisition, and a log-linear cost specification.
    """
    simulation = Simulation(
        build_id_data(T=10, J=20, F=9, mergers=[{f: 4 + int(f > 0) for f in range(4)}]),
        Integration('product', 4),
        gamma=[0, 0.1, 0.2, 0.3, 0.5, 0],
        beta=[-6, 1, 1, 0, 0, 0, 2],
        sigma=[
            [1, 0, -0.1, 0, 0, 0, 0],
            [0, 0,  0,   0, 0, 0, 0],
            [0, 0,  2,   0, 0, 0, 0],
            [0, 0,  0,   0, 0, 0, 0],
            [0, 0,  0,   0, 0, 0, 0],
            [0, 0,  0,   0, 0, 0, 0],
            [0, 0,  0,   0, 0, 0, 0]
        ],
        pi=[
            [1, 0],
            [0, 0],
            [0, 2],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ],
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
    if not supply:
        product_data = np.lib.recfunctions.drop_fields(product_data, ['cost_characteristics', 'supply_instruments'])
    problem = Problem(product_data, simulation.agent_data, nonlinear_prices=simulation.nonlinear_prices)
    results = problem.solve(simulation.sigma, simulation.pi, steps=1, linear_costs=simulation.linear_costs)
    return simulation, problem, results


@pytest.fixture
def blp_problem():
    """Configure the example automobile problem."""
    data = np.recfromcsv(BLP_PRODUCTS_LOCATION)
    linear_characteristics = np.c_[np.ones(data.size), data['hpwt'], data['air'], data['mpg'], data['space']]
    product_data = {k: data[k] for k in data.dtype.names}
    product_data.update({
        'firm_ids': np.c_[data['firm_ids'], data['changed_firm_ids']],
        'linear_characteristics': linear_characteristics,
        'nonlinear_characteristics': linear_characteristics[:, :-1],
        'demand_instruments': np.c_[linear_characteristics, build_blp_instruments({
            'market_ids': data['market_ids'],
            'firm_ids': data['firm_ids'],
            'characteristics': linear_characteristics
        })]
    })
    return Problem(product_data, np.recfromcsv(BLP_AGENTS_LOCATION))


@pytest.fixture
def nevo_problem():
    """Configure the example fake cereal problem."""
    data = np.recfromcsv(NEVO_PRODUCTS_LOCATION)
    indicators = build_indicators(data['product_ids'])
    product_data = {k: data[k] for k in data.dtype.names}
    product_data.update({
        'firm_ids': np.c_[data['firm_ids'], data['changed_firm_ids']],
        'linear_characteristics': indicators,
        'nonlinear_characteristics': np.c_[np.ones(data.size), data['sugar'], data['mushy']],
        'demand_instruments': np.c_[indicators, extract_matrix(data, 'instruments')]
    })
    return Problem(product_data, np.recfromcsv(NEVO_AGENTS_LOCATION))


@pytest.fixture(params=[pytest.param('blp_problem', id="BLP"), pytest.param('nevo_problem', id="Nevo")])
def knittel_metaxoglou_2014(request):
    """Load initial parameter values and estimates from replication code for Knittel and Metaxoglou (2014).

    The replication code was modified to output one Matlab data file for each dataset (BLP automobile data and Nevo's
    cereal data), each containing the results of one round of Knitro optimization and post-estimation calculations. The
    replication code was kept mostly intact, but was modified slightly in the following ways:

        - Tolerance parameters, Knitro optimization parameters, and starting values for sigma were all configured.
        - A bug in the code's computation of the BLP instruments was fixed. When creating a vector of "other" and
          "rival" sums, the code did not specify a dimension over which to sum, which created problems with one-
          dimensional vectors. A dimension of 1 was added to both sum commands.
        - For the Nevo data, the constant column and the column of prices were swapped in X2. Parameter matrices were
          changed accordingly. For example, before elasticities were computed, the order of coefficients was changed to
          reflect the new position of prices.
        - Delta was initialized as the solution to the Logit model.
        - After estimation, the objective was called again at the optimal parameters to re-load globals at the optimal
          parameter values.
        - Before being saved to a Matlab data file, matrices were renamed and reshaped.

    """
    problem = request.getfixturevalue(request.param)
    return scipy.io.loadmat(str(TEST_DATA_PATH / 'knittel_metaxoglou_2014' / request.param), {'problem': problem})


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
