"""Fixtures used by tests."""

import hashlib
import os
from typing import Any, Callable, Dict, Hashable, Iterator, List, Tuple

import numpy as np
import patsy
import pytest
import scipy.linalg

from pyblp import (
    Formulation, Integration, Optimization, Problem, ProblemResults, ProductsAgentsCovarianceMoment, Simulation,
    SimulationResults, build_differentiation_instruments, build_id_data, build_matrix, build_ownership, options
)
from pyblp.utilities.basics import update_matrices, Array, Data, Options


# define common types
SimulationFixture = Tuple[Simulation, SimulationResults, List[ProductsAgentsCovarianceMoment]]
SimulatedProblemFixture = Tuple[Simulation, SimulationResults, Problem, Options, ProblemResults]


@pytest.fixture(scope='session', autouse=True)
def configure() -> Iterator[None]:
    """Configure NumPy so that it raises all warnings as exceptions. Next, if a DTYPE environment variable is set in
    this testing environment that is different from the default data type, use it for all numeric calculations. Finally,
    cache results for SciPy linear algebra inversion routines. This is very memory inefficient but guarantees that
    matrix inversion will always give rise to the same deterministic result, which is important for precise testing of
    equality.
    """

    # configure NumPy so that it raises all warnings as exceptions
    old_error = np.seterr(all='raise')

    # use any different data type for all numeric calculations
    old_dtype = options.dtype
    dtype_string = os.environ.get('DTYPE')
    if dtype_string:
        options.dtype = np.dtype(dtype_string)
        if np.finfo(options.dtype).dtype == old_dtype:
            pytest.skip(f"The {dtype_string} data type is the same as the default one in this environment.")

    # define a patch for SciPy functions
    def patch(uncached: Callable) -> Tuple[Callable, Callable]:
        """Patch a function by caching its array arguments."""
        mapping: Dict[Hashable, Array] = {}

        # define the cached function
        def cached(*args: Array) -> Array:
            """Replicate the function, caching its results."""
            nonlocal mapping
            key = tuple(hashlib.sha1(a.data.tobytes()).digest() for a in args)
            if key not in mapping:
                mapping[key] = uncached(*args)
            return mapping[key]
        return uncached, cached

    # monkey patch the functions
    old_inv, scipy.linalg.inv = patch(scipy.linalg.inv)
    old_solve, scipy.linalg.solve = patch(scipy.linalg.solve)

    # run tests before reverting all changes
    yield
    scipy.linalg.inv = old_inv
    scipy.linalg.solve = old_solve
    options.dtype = old_dtype
    np.seterr(**old_error)


@pytest.fixture(scope='session')
def small_logit_simulation() -> SimulationFixture:
    """Solve a simulation with two markets, a linear constant, linear prices, a linear characteristic, and a cost
    characteristic
    """
    id_data = build_id_data(T=2, J=18, F=3)
    simulation = Simulation(
        product_formulations=(
            Formulation('1 + prices + x'),
            None,
            Formulation('0 + a')
        ),
        beta=[1, -5, 1],
        sigma=None,
        gamma=2,
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'clustering_ids': np.random.RandomState(0).choice(range(10), id_data.size)
        },
        xi_variance=0.001,
        omega_variance=0.001,
        correlation=0.7,
        seed=0
    )
    simulation_results = simulation.solve()
    return simulation, simulation_results, []


@pytest.fixture(scope='session')
def large_logit_simulation() -> SimulationFixture:
    """Solve a simulation with ten markets, a linear constant, linear prices, a linear/cost characteristic, another two
    linear characteristics, another three cost characteristics, and a log-linear cost specification.
    """
    id_data = build_id_data(T=10, J=20, F=9)
    simulation = Simulation(
        product_formulations=(
            Formulation('1 + prices + x + y + z'),
            None,
            Formulation('0 + log(x) + a + b + c')
        ),
        beta=[1, -6, 1, 2, 3],
        sigma=None,
        gamma=[0.1, 0.2, 0.3, 0.5],
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'clustering_ids': np.random.RandomState(2).choice(range(30), id_data.size)
        },
        xi_variance=0.00001,
        omega_variance=0.00001,
        correlation=0.1,
        costs_type='log',
        seed=2
    )
    simulation_results = simulation.solve()
    return simulation, simulation_results, []


@pytest.fixture(scope='session')
def small_nested_logit_simulation() -> SimulationFixture:
    """Solve a simulation with four markets, linear prices, two linear characteristics, two cost characteristics, and
    two nesting groups with different nesting parameters
    """
    id_data = build_id_data(T=4, J=18, F=3)
    simulation = Simulation(
        product_formulations=(
            Formulation('0 + prices + x + y'),
            None,
            Formulation('0 + a + b')
        ),
        beta=[-5, 1, 1],
        sigma=None,
        gamma=[2, 1],
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'nesting_ids': np.random.RandomState(0).choice(['f', 'g'], id_data.size),
            'clustering_ids': np.random.RandomState(0).choice(range(10), id_data.size)
        },
        rho=[0.1, 0.2],
        xi_variance=0.001,
        omega_variance=0.001,
        correlation=0.7,
        seed=0
    )
    simulation_results = simulation.solve()
    return simulation, simulation_results, []


@pytest.fixture(scope='session')
def large_nested_logit_simulation() -> SimulationFixture:
    """Solve a simulation with ten markets, a linear constant, linear prices, a linear/cost characteristic, another two
    linear characteristics, another three cost characteristics, three nesting groups with the same nesting
    parameter, and a log-linear cost specification.
    """
    id_data = build_id_data(T=10, J=20, F=9)
    simulation = Simulation(
        product_formulations=(
            Formulation('1 + prices + x + y + z'),
            None,
            Formulation('0 + log(x) + a + b + c')
        ),
        beta=[1, -6, 1, 2, 3],
        sigma=None,
        gamma=[0.1, 0.2, 0.3, 0.5],
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'nesting_ids': np.random.RandomState(2).choice(['f', 'g', 'h'], id_data.size),
            'clustering_ids': np.random.RandomState(2).choice(range(30), id_data.size)
        },
        rho=0.1,
        xi_variance=0.00001,
        omega_variance=0.00001,
        correlation=0.9,
        costs_type='log',
        seed=2
    )
    simulation_results = simulation.solve()
    return simulation, simulation_results, []


@pytest.fixture(scope='session')
def small_blp_simulation() -> SimulationFixture:
    """Solve a simulation with three markets, linear prices, a linear/nonlinear characteristic, two cost
    characteristics, and uniform unobserved product characteristics.
    """
    id_data = build_id_data(T=3, J=18, F=3)
    uniform = 0.001 * np.random.RandomState(0).uniform(size=(id_data.size, 3))
    simulation = Simulation(
        product_formulations=(
            Formulation('0 + prices + x'),
            Formulation('0 + x'),
            Formulation('0 + a + b')
        ),
        beta=[-5, 1],
        sigma=2,
        gamma=[2, 1],
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'clustering_ids': np.random.RandomState(0).choice(range(10), id_data.size)
        },
        integration=Integration('product', 3),
        xi=uniform[:, 0] + uniform[:, 1],
        omega=uniform[:, 0] + uniform[:, 2],
        seed=0
    )
    simulation_results = simulation.solve()
    return simulation, simulation_results, []


@pytest.fixture(scope='session')
def medium_blp_simulation() -> SimulationFixture:
    """Solve a simulation with four markets, linear/nonlinear/cost constants, two linear characteristics, two cost
    characteristics, a demographic interacted with second-degree prices, and an alternative ownership structure.
    """
    id_data = build_id_data(T=4, J=25, F=6)
    simulation = Simulation(
        product_formulations=(
            Formulation('1 + x + y'),
            Formulation('1 + I(prices ** 2)'),
            Formulation('1 + a + b')
        ),
        beta=[1, 2, 1],
        sigma=[
            [0.5, 0],
            [0.0, 0],
        ],
        gamma=[1, 1, 2],
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'clustering_ids': np.random.RandomState(1).choice(range(20), id_data.size),
            'ownership': build_ownership(id_data, lambda f, g: 1 if f == g else (0.1 if f > 3 and g > 3 else 0))
        },
        agent_formulation=Formulation('0 + f'),
        pi=[
            [+0],
            [-3]
        ],
        integration=Integration('product', 4),
        xi_variance=0.0001,
        omega_variance=0.0001,
        correlation=0.8,
        seed=1
    )
    simulation_results = simulation.solve()
    simulated_micro_moments = [ProductsAgentsCovarianceMoment(X2_index=1, demographics_index=0, value=0)]
    return simulation, simulation_results, simulated_micro_moments


@pytest.fixture(scope='session')
def large_blp_simulation() -> SimulationFixture:
    """Solve a simulation with 20 markets, varying numbers of products per market, a linear constant, linear/nonlinear
    prices, a linear/nonlinear/cost characteristic, another three linear characteristics, another two cost
    characteristics, demographics interacted with prices and the linear/nonlinear/cost characteristic, dense parameter
    matrices, a log-linear cost specification, and local differentiation instruments on the demand side.
    """
    id_data = build_id_data(T=20, J=20, F=9)
    keep = np.arange(id_data.size)
    np.random.RandomState(0).shuffle(keep)
    id_data = id_data[keep[:int(0.5 * id_data.size)]]
    simulation = Simulation(
        product_formulations=(
            Formulation('1 + prices + x + y + z + q'),
            Formulation('0 + prices + x'),
            Formulation('0 + log(x) + log(a) + log(b)')
        ),
        beta=[1, -10, 1, 2, 3, 1],
        sigma=[
            [1, -0.1],
            [0, +2.0]
        ],
        gamma=[0.1, 0.2, 0.3],
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'clustering_ids': np.random.RandomState(2).choice(range(30), id_data.size)
        },
        agent_formulation=Formulation('0 + f + g'),
        pi=[
            [1, 0],
            [0, 2]
        ],
        integration=Integration('product', 4),
        xi_variance=0.00001,
        omega_variance=0.00001,
        correlation=0.9,
        costs_type='log',
        seed=2
    )
    simulation_results = simulation.solve()
    differentiation_instruments = np.c_[
        build_differentiation_instruments(Formulation('0 + x + y + z + q'), simulation_results.product_data),
        build_matrix(Formulation('0 + a + b'), simulation_results.product_data)
    ]
    simulation_results.product_data = update_matrices(simulation_results.product_data, {
        'demand_instruments': (differentiation_instruments, simulation_results.product_data.demand_instruments.dtype)
    })
    simulated_micro_moments = [
        ProductsAgentsCovarianceMoment(X2_index=0, demographics_index=0, value=0),
        ProductsAgentsCovarianceMoment(X2_index=1, demographics_index=1, value=0),
        ProductsAgentsCovarianceMoment(
            X2_index=0, demographics_index=1, value=0, market_ids=simulation.unique_market_ids[:5]
        )
    ]
    return simulation, simulation_results, simulated_micro_moments


@pytest.fixture(scope='session')
def small_nested_blp_simulation() -> SimulationFixture:
    """Solve a simulation with five markets, linear prices, a linear/nonlinear characteristic, another linear
    characteristic, three cost characteristics, and two nesting groups with different nesting parameters.
    """
    id_data = build_id_data(T=5, J=18, F=3)
    simulation = Simulation(
        product_formulations=(
            Formulation('0 + prices + x + z'),
            Formulation('0 + x'),
            Formulation('0 + a + b + c')
        ),
        beta=[-5, 1, 2],
        sigma=2,
        gamma=[2, 1, 1],
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'nesting_ids': np.random.RandomState(0).choice(['f', 'g'], id_data.size),
            'clustering_ids': np.random.RandomState(0).choice(range(10), id_data.size)
        },
        integration=Integration('product', 3),
        rho=[0.1, 0.2],
        xi_variance=0.001,
        omega_variance=0.001,
        correlation=0.7,
        seed=0
    )
    simulation_results = simulation.solve()
    return simulation, simulation_results, []


@pytest.fixture(scope='session')
def large_nested_blp_simulation() -> SimulationFixture:
    """Solve a simulation with 20 markets, varying numbers of products per market, a linear constant, linear/nonlinear
    prices, a linear/nonlinear/cost characteristic, another three linear characteristics, another two cost
    characteristics, demographics interacted with prices and the linear/nonlinear/cost characteristic, three nesting
    groups with the same nesting parameter, and a log-linear cost specification.
    """
    id_data = build_id_data(T=20, J=20, F=9)
    keep = np.arange(id_data.size)
    np.random.RandomState(0).shuffle(keep)
    id_data = id_data[keep[:int(0.5 * id_data.size)]]
    simulation = Simulation(
        product_formulations=(
            Formulation('1 + prices + x + y + z + q'),
            Formulation('0 + prices + x'),
            Formulation('0 + log(x) + log(a) + log(b)')
        ),
        beta=[1, -10, 1, 2, 3, 1],
        sigma=[
            [1, 0],
            [0, 2]
        ],
        gamma=[0.1, 0.2, 0.3],
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'nesting_ids': np.random.RandomState(2).choice(['f', 'g', 'h'], id_data.size),
            'clustering_ids': np.random.RandomState(2).choice(range(30), id_data.size)
        },
        agent_formulation=Formulation('0 + f + g'),
        pi=[
            [1, 0],
            [0, 2]
        ],
        integration=Integration('product', 4),
        rho=0.1,
        xi_variance=0.00001,
        omega_variance=0.00001,
        correlation=0.9,
        costs_type='log',
        seed=2
    )
    simulation_results = simulation.solve()
    simulated_micro_moments = [
        ProductsAgentsCovarianceMoment(X2_index=0, demographics_index=0, value=0),
        ProductsAgentsCovarianceMoment(X2_index=1, demographics_index=1, value=0)
    ]
    return simulation, simulation_results, simulated_micro_moments


@pytest.fixture(scope='session', params=[
    pytest.param(['small_logit', False], id="small Logit simulation without supply"),
    pytest.param(['small_logit', True], id="small Logit simulation with supply"),
    pytest.param(['large_logit', False], id="large Logit simulation without supply"),
    pytest.param(['large_logit', True], id="large Logit simulation with supply"),
    pytest.param(['small_nested_logit', False], id="small nested Logit simulation without supply"),
    pytest.param(['small_nested_logit', True], id="small nested Logit simulation with supply"),
    pytest.param(['large_nested_logit', False], id="large nested Logit simulation without supply"),
    pytest.param(['large_nested_logit', True], id="large nested Logit simulation with supply"),
    pytest.param(['small_blp', False], id="small BLP simulation without supply"),
    pytest.param(['small_blp', True], id="small BLP simulation with supply"),
    pytest.param(['medium_blp', False], id="medium BLP simulation without supply"),
    pytest.param(['medium_blp', True], id="medium BLP simulation with supply"),
    pytest.param(['large_blp', False], id="large BLP simulation without supply"),
    pytest.param(['large_blp', True], id="large BLP simulation with supply"),
    pytest.param(['small_nested_blp', False], id="small nested BLP simulation without supply"),
    pytest.param(['small_nested_blp', True], id="small nested BLP simulation with supply"),
    pytest.param(['large_nested_blp', False], id="large nested BLP simulation without supply"),
    pytest.param(['large_nested_blp', True], id="large nested BLP simulation with supply")
])
def simulated_problem(request: Any) -> SimulatedProblemFixture:
    """Configure and solve a simulated problem, either with or without supply-side data. Preclude overflow with rho
    bounds that are more conservative than the default ones.
    """
    name, supply = request.param
    simulation, simulation_results, simulated_micro_moments = request.getfixturevalue(f'{name}_simulation')
    micro_moments = []
    if simulated_micro_moments:
        micro_values = simulation_results.compute_micro(simulated_micro_moments)
        for moment, value in zip(simulated_micro_moments, micro_values):
            micro_moments.append(ProductsAgentsCovarianceMoment(
                moment.X2_index, moment.demographics_index, value, moment.market_ids
            ))
    problem = simulation_results.to_problem(simulation.product_formulations[:2 + int(supply)])
    solve_options = {
        'sigma': simulation.sigma,
        'pi': simulation.pi,
        'rho': simulation.rho,
        'beta': np.where(simulation._parameters.alpha_index, simulation.beta if supply else np.nan, np.nan),
        'rho_bounds': (np.zeros_like(simulation.rho), np.minimum(0.9, 1.5 * simulation.rho)),
        'costs_type': simulation.costs_type,
        'method': '1s',
        'check_optimality': 'gradient',
        'optimization': Optimization('slsqp', {'ftol': 1e-10}),
        'micro_moments': micro_moments,
        'micro_covariances': lambda m: np.eye(m.size)
    }
    problem_results = problem.solve(**solve_options)
    return simulation, simulation_results, problem, solve_options, problem_results


@pytest.fixture(scope='session', params=[pytest.param(1, id="1 observation"), pytest.param(10, id="10 observations")])
def formula_data(request: Any) -> Data:
    """Simulate patsy demo data with two-level categorical variables and varying numbers of observations."""
    raw_data = patsy.user_util.demo_data('a', 'b', 'c', 'x', 'y', 'z', nlevels=2, min_rows=request.param)
    return {k: np.array(v) if isinstance(v[0], str) else np.abs(v) for k, v in raw_data.items()}
