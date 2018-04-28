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
    testing environment, use the specified data type for all numeric calculations.
    """
    np.seterr(all='raise')
    options.dtype = np.dtype(os.environ.get('DTYPE') or options.dtype)


@pytest.fixture
def small_simulations():
    """Simulations with one market, a cost/nonlinear characteristic, and a single acquisition."""
    simulations = []
    simulation_solutions = []
    for seed in range(3):
        simulation = Simulation(
            build_id_data(T=1, J=20, F=10, mergers=[{1: 0}]),
            Integration('product', 3),
            gamma=[0, 1],
            beta=[-5, 0, 0],
            sigma=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]
            ],
            xi_variance=0.001,
            omega_variance=0.001,
            correlation=0.7,
            seed=seed
        )
        simulations.append(simulation)
        simulation_solutions.append(simulation.solve())
    return simulations, simulation_solutions


@pytest.fixture
def medium_simulations():
    """Simulations with two markets, nonlinear prices, a nonlinear constant, a cost/linear characteristic, another cost
    characteristic, a demographic, sparse parameter matrices, and an acquisition of four firms.
    """
    simulations = []
    simulation_solutions = []
    for seed in range(5):
        simulation = Simulation(
            build_id_data(T=2, J=25, F=11, mergers=[{f: 4 for f in range(4)}]),
            Integration('nested_product', 4),
            gamma=[0, 1, 2],
            beta=[-10, 0, 1, 0],
            sigma=[
                [1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            pi=[
                [0],
                [1],
                [0],
                [0]
            ],
            xi_variance=0.0001,
            omega_variance=0.0001,
            correlation=0.8,
            seed=seed
        )
        simulations.append(simulation)
        simulation_solutions.append(simulation.solve())
    return simulations, simulation_solutions


@pytest.fixture
def large_simulations():
    """Simulations with three markets, nonlinear prices, a cost constant, two linear/cost characteristics, another
    nonlinear/cost characteristic, two other cost characteristics, two demographics, dense parameter matrices, an
    acquisition of two firms, another acquisition of five firms, and a log-linear cost specification.
    """
    simulations = []
    simulation_solutions = []
    for seed in range(15):
        simulation = Simulation(
            build_id_data(T=3, J=30, F=15, mergers=[{f: 7 + int(f > 1) for f in range(7)}]),
            Integration('product', 4),
            gamma=[0.1, 0.1, 0.2, 0.3, 0.1, 0.2],
            beta=[-4, 0, 0, 10, 6, 0, 0],
            sigma=[
                [0.1, 0, -0.1, 0, 0, 0, 0],
                [0,   0,  0,   0, 0, 0, 0],
                [0,   0,  1,   0, 0, 0, 0],
                [0,   0,  0,   0, 0, 0, 0],
                [0,   0,  0,   0, 0, 0, 0],
                [0,   0,  0,   0, 0, 0, 0],
                [0,   0,  0,   0, 0, 0, 0]
            ],
            pi=[
                [0, 0.1],
                [0, 0  ],
                [1, 0  ],
                [0, 0  ],
                [0, 0  ],
                [0, 0  ],
                [0, 0  ]
            ],
            xi_variance=0.000001,
            omega_variance=0.000001,
            correlation=0.9,
            linear_costs=False,
            seed=seed
        )
        simulations.append(simulation)
        simulation_solutions.append(simulation.solve())
    return simulations, simulation_solutions


@pytest.fixture(params=[
    pytest.param(['small', False], id="small without supply"),
    pytest.param(['small', True], id="small with supply"),
    pytest.param(['medium', False], id="medium without supply"),
    pytest.param(['medium', True], id="medium with supply"),
    pytest.param(['large', False], id="large without supply"),
    pytest.param(['large', True], id="large with supply")
])
def simulated_problems(request):
    """Problems configured with product and agent data from solved simulations. Parametrized to either include or not
    include supply-side data.
    """
    problems = []
    name, supply = request.param
    simulations, simulation_solutions = request.getfixturevalue(f'{name}_simulations')
    for simulation, product_data in zip(simulations, simulation_solutions):
        if not supply:
            product_data = np.lib.recfunctions.drop_fields(product_data, ['cost_characteristics', 'supply_instruments'])
        problems.append(Problem(product_data, simulation.agent_data, nonlinear_prices=simulation.nonlinear_prices))
    return simulations, problems


@pytest.fixture
def blp_problem():
    """Example problem using the BLP automobile data."""
    raw = np.recfromcsv(BLP_PRODUCTS_LOCATION)
    linear_characteristics = np.c_[np.ones(raw.size), raw['hpwt'], raw['air'], raw['mpg'], raw['space']]
    product_data = {k: raw[k] for k in raw.dtype.names}
    product_data.update({
        'firm_ids': np.c_[raw['firm_ids'], raw['changed_firm_ids']],
        'linear_characteristics': linear_characteristics,
        'nonlinear_characteristics': linear_characteristics[:, :-1],
        'demand_instruments': np.c_[linear_characteristics, build_blp_instruments({
            'market_ids': raw['market_ids'],
            'firm_ids': raw['firm_ids'],
            'characteristics': linear_characteristics
        })]
    })
    return Problem(product_data, np.recfromcsv(BLP_AGENTS_LOCATION))


@pytest.fixture
def nevo_problem():
    """Example problem using the Nevo fake cereal data."""
    raw = np.recfromcsv(NEVO_PRODUCTS_LOCATION)
    indicators = build_indicators(raw['product_ids'])
    product_data = {k: raw[k] for k in raw.dtype.names}
    product_data.update({
        'firm_ids': np.c_[raw['firm_ids'], raw['changed_firm_ids']],
        'linear_characteristics': indicators,
        'nonlinear_characteristics': np.c_[np.ones(raw.size), raw['sugar'], raw['mushy']],
        'demand_instruments': np.c_[indicators, extract_matrix(raw, 'instruments')]
    })
    return Problem(product_data, np.recfromcsv(NEVO_AGENTS_LOCATION))


@pytest.fixture(params=[pytest.param('blp_problem', id="BLP"), pytest.param('nevo_problem', id="Nevo")])
def knittel_metaxoglou_2014(request):
    """Initial parameter values for and estimates created by replication code for Knittel and Metaxoglou (2014).

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
    """Sample of pre-computed sparse grids of nodes and weights computed according to the Gauss-Hermite quadrature rule
    and its nested analog, which were computed by the Matlab function nwspgr by Florian Heiss and Viktor Winschel.
    """
    rule, dimensions, level = re.search(r'(GQN|KPN)_d(\d+)_l(\d+)', request.param.name).groups()
    matrix = np.atleast_2d(np.genfromtxt(request.param, delimiter=','))
    nested = rule == 'KPN'
    nodes = matrix[:, :-1]
    weights = matrix[:, -1]
    return int(dimensions), int(level), nested, nodes, weights
