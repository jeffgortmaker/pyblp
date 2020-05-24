"""Fixtures used by tests."""

import hashlib
import os
from typing import Any, Callable, Dict, Hashable, Iterator, List, Tuple

import numpy as np
import patsy
import pytest
import scipy.linalg

from pyblp import (
    Formulation, Integration, Problem, ProblemResults, DemographicExpectationMoment, DemographicCovarianceMoment,
    DiversionProbabilityMoment, DiversionCovarianceMoment, Simulation, SimulationResults,
    build_differentiation_instruments, build_id_data, build_matrix, build_ownership, options
)
from pyblp.utilities.basics import update_matrices, Array, Data, Options


# define common types
SimulationFixture = Tuple[Simulation, SimulationResults, Dict[str, Array], List[Any]]
SimulatedProblemFixture = Tuple[Simulation, SimulationResults, Problem, Options, ProblemResults]


@pytest.fixture(scope='session', autouse=True)
def configure() -> Iterator[None]:
    """Configure NumPy so that it raises all warnings as exceptions. Next, if a DTYPE environment variable is set in
    this testing environment that is different from the default data type, use it for all numeric calculations. Finally,
    cache results for SciPy linear algebra routines. This is very memory inefficient but guarantees that linear algebra
    will always give rise to the same deterministic result, which is important for precise testing of equality.
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

    def patch(uncached: Callable) -> Callable:
        """Patch a function by caching its array arguments."""
        mapping: Dict[Hashable, Array] = {}

        def cached(*args: Array, **kwargs: Any) -> Array:
            """Replicate the function, caching its results."""
            nonlocal mapping
            key = tuple(hashlib.sha1(a.data.tobytes()).digest() for a in args)
            if key not in mapping:
                mapping[key] = uncached(*args, **kwargs)
            return mapping[key]

        return cached

    # patch the functions
    old = {}
    for name in ['inv', 'solve', 'svd', 'eigvalsh', 'pinv', 'qr']:
        old[name] = getattr(scipy.linalg, name)
        setattr(scipy.linalg, name, patch(old[name]))

    # run tests before reverting all changes
    yield
    for name, old in old.items():
        setattr(scipy.linalg, name, old)
    options.dtype = old_dtype
    np.seterr(**old_error)


@pytest.fixture(scope='session')
def small_logit_simulation() -> SimulationFixture:
    """Solve a simulation with two markets, a linear constant, linear prices, a linear characteristic, a cost
    characteristic, and a scaled epsilon.
    """
    id_data = build_id_data(T=2, J=18, F=3)
    simulation = Simulation(
        product_formulations=(
            Formulation('1 + prices + x'),
            None,
            Formulation('0 + a')
        ),
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'clustering_ids': np.random.RandomState(0).choice(range(10), id_data.size)
        },
        beta=[1, -5, 1],
        gamma=2,
        xi_variance=0.001,
        omega_variance=0.001,
        correlation=0.7,
        epsilon_scale=0.5,
        seed=0,
    )
    simulation_results = simulation.replace_exogenous('x', 'a')
    return simulation, simulation_results, {}, []


@pytest.fixture(scope='session')
def large_logit_simulation() -> SimulationFixture:
    """Solve a simulation with ten markets, a linear constant, linear prices, a linear/cost characteristic, another two
    linear characteristics, another two cost characteristics, and a quantity-dependent, log-linear cost specification.
    """
    id_data = build_id_data(T=10, J=20, F=9)
    simulation = Simulation(
        product_formulations=(
            Formulation('1 + prices + x + y + z'),
            None,
            Formulation('0 + log(x) + a + b + shares')
        ),
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'clustering_ids': np.random.RandomState(2).choice(range(30), id_data.size)
        },
        beta=[1, -6, 1, 2, 3],
        gamma=[0.1, 0.2, 0.3, -0.1],
        xi_variance=0.00001,
        omega_variance=0.00001,
        correlation=0.1,
        costs_type='log',
        seed=2
    )
    simulation_results = simulation.replace_endogenous()
    return simulation, simulation_results, {}, []


@pytest.fixture(scope='session')
def small_nested_logit_simulation() -> SimulationFixture:
    """Solve a simulation with four markets, linear prices, two linear characteristics, two cost characteristics, and
    two nesting groups with different nesting parameters.
    """
    id_data = build_id_data(T=4, J=18, F=3)
    simulation = Simulation(
        product_formulations=(
            Formulation('0 + prices + x + y'),
            None,
            Formulation('0 + a + b')
        ),
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'nesting_ids': np.random.RandomState(0).choice(['f', 'g'], id_data.size),
            'clustering_ids': np.random.RandomState(0).choice(range(10), id_data.size)
        },
        beta=[-5, 1, 1],
        gamma=[2, 1],
        rho=[0.1, 0.2],
        xi_variance=0.001,
        omega_variance=0.001,
        correlation=0.7,
        seed=0
    )
    simulation_results = simulation.replace_endogenous()
    return simulation, simulation_results, {}, []


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
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'nesting_ids': np.random.RandomState(2).choice(['f', 'g', 'h'], id_data.size),
            'clustering_ids': np.random.RandomState(2).choice(range(30), id_data.size)
        },
        beta=[1, -6, 1, 2, 3],
        gamma=[0.1, 0.2, 0.3, 0.5],
        rho=0.1,
        xi_variance=0.00001,
        omega_variance=0.00001,
        correlation=0.9,
        costs_type='log',
        seed=2
    )
    simulation_results = simulation.replace_endogenous()
    return simulation, simulation_results, {}, []


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
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'clustering_ids': np.random.RandomState(0).choice(range(10), id_data.size)
        },
        beta=[-5, 1],
        sigma=2,
        gamma=[2, 1],
        integration=Integration('product', 3),
        xi=uniform[:, 0] + uniform[:, 1],
        omega=uniform[:, 0] + uniform[:, 2],
        seed=0
    )
    simulation_results = simulation.replace_endogenous()
    return simulation, simulation_results, {}, []


@pytest.fixture(scope='session')
def medium_blp_simulation() -> SimulationFixture:
    """Solve a simulation with four markets, linear/nonlinear/cost constants, two linear characteristics, two cost
    characteristics, a demographic interacted with second-degree prices, an alternative ownership structure, and a
    scaled epsilon.
    """
    id_data = build_id_data(T=4, J=25, F=6)
    simulation = Simulation(
        product_formulations=(
            Formulation('1 + x + y'),
            Formulation('1 + I(prices**2)'),
            Formulation('1 + a + b')
        ),
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'clustering_ids': np.random.RandomState(1).choice(range(20), id_data.size),
            'ownership': build_ownership(id_data, lambda f, g: 1 if f == g else (0.1 if f > 3 and g > 3 else 0))
        },
        beta=[1, 2, 1],
        sigma=[
            [0.5, 0],
            [0.0, 0],
        ],
        pi=[
            [+0],
            [-3]
        ],
        gamma=[1, 1, 2],
        agent_formulation=Formulation('0 + f'),
        integration=Integration('product', 4),
        xi_variance=0.0001,
        omega_variance=0.0001,
        correlation=0.8,
        epsilon_scale=0.7,
        seed=1,
    )
    simulation_results = simulation.replace_endogenous()
    simulated_micro_moments = [DemographicCovarianceMoment(
        X2_index=0, demographics_index=0, value=0, market_ids=[simulation.unique_market_ids[2]]
    )]
    return simulation, simulation_results, {}, simulated_micro_moments


@pytest.fixture(scope='session')
def large_blp_simulation() -> SimulationFixture:
    """Solve a simulation with 20 markets, varying numbers of products per market, a linear constant, log-linear
    coefficients on prices, a linear/nonlinear/cost characteristic, another three linear characteristics, another two
    cost characteristics, demographics interacted with prices and the linear/nonlinear/cost characteristic, dense
    parameter matrices, a log-linear cost specification, and local differentiation instruments.
    """
    id_data = build_id_data(T=20, J=20, F=9)

    keep = np.arange(id_data.size)
    np.random.RandomState(0).shuffle(keep)
    id_data = id_data[keep[:int(0.5 * id_data.size)]]

    product_ids = id_data.market_ids.copy()
    for t in np.unique(id_data.market_ids):
        product_ids[id_data.market_ids == t] = np.arange((id_data.market_ids == t).sum())

    simulation = Simulation(
        product_formulations=(
            Formulation('1 + x + y + z + q'),
            Formulation('1 + I(-prices) + x'),
            Formulation('0 + log(x) + log(a) + log(b)')
        ),
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'product_ids': product_ids,
            'clustering_ids': np.random.RandomState(2).choice(range(30), id_data.size)
        },
        beta=[1, 1, 2, 3, 1],
        sigma=[
            [0, +0.0, 0],
            [0, +0.5, 0],
            [0, -0.2, 2]
        ],
        pi=[
            [0, 0, 0],
            [2, 1, 0],
            [0, 0, 2]
        ],
        gamma=[0.1, 0.2, 0.3],
        agent_formulation=Formulation('1 + f + g'),
        integration=Integration('product', 4),
        xi_variance=0.00001,
        omega_variance=0.00001,
        correlation=0.9,
        distributions=['normal', 'lognormal', 'normal'],
        costs_type='log',
        seed=2
    )
    simulation_results = simulation.replace_endogenous()
    simulated_data_override = {
        'demand_instruments': np.c_[
            build_differentiation_instruments(Formulation('0 + x + y + z + q'), simulation_results.product_data),
            build_matrix(Formulation('0 + a + b'), simulation_results.product_data)
        ],
        'supply_instruments': np.c_[
            build_differentiation_instruments(Formulation('0 + x + a + b'), simulation_results.product_data),
            build_matrix(Formulation('0 + y + z + q'), simulation_results.product_data)
        ]
    }
    simulated_micro_moments = [
        DemographicExpectationMoment(product_id=0, demographics_index=1, value=0),
        DemographicExpectationMoment(
            product_id=None, demographics_index=1, value=0, market_ids=simulation.unique_market_ids[1:4]
        ),
        DemographicCovarianceMoment(
            X2_index=0, demographics_index=2, value=0, market_ids=simulation.unique_market_ids[3:5]
        ),
        DiversionProbabilityMoment(
            product_id1=1, product_id2=0, value=0, market_ids=simulation.unique_market_ids[6:10]
        ),
        DiversionProbabilityMoment(
            product_id1=None, product_id2=1, value=0, market_ids=[simulation.unique_market_ids[8]]
        ),
        DiversionProbabilityMoment(
            product_id1=1, product_id2=None, value=0, market_ids=[simulation.unique_market_ids[9]]
        ),
        DiversionCovarianceMoment(
            X2_index1=1, X2_index2=1, value=0, market_ids=[simulation.unique_market_ids[12]]
        ),
    ]
    return simulation, simulation_results, simulated_data_override, simulated_micro_moments


@pytest.fixture(scope='session')
def small_nested_blp_simulation() -> SimulationFixture:
    """Solve a simulation with eight markets, linear prices, a linear/nonlinear characteristic, another linear
    characteristic, three cost characteristics, and two nesting groups with different nesting parameters.
    """
    id_data = build_id_data(T=8, J=18, F=3)
    simulation = Simulation(
        product_formulations=(
            Formulation('0 + prices + x + z'),
            Formulation('0 + x'),
            Formulation('0 + a + b + c')
        ),
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'nesting_ids': np.random.RandomState(0).choice(['f', 'g'], id_data.size),
            'clustering_ids': np.random.RandomState(0).choice(range(10), id_data.size)
        },
        beta=[-5, 1, 2],
        sigma=2,
        gamma=[2, 1, 1],
        rho=[0.1, 0.2],
        integration=Integration('product', 3),
        xi_variance=0.001,
        omega_variance=0.001,
        correlation=0.7,
        seed=0
    )
    simulation_results = simulation.replace_endogenous()
    return simulation, simulation_results, {}, []


@pytest.fixture(scope='session')
def large_nested_blp_simulation() -> SimulationFixture:
    """Solve a simulation with 20 markets, varying numbers of products per market, a linear constant, log-normal
    coefficients on prices, a linear/nonlinear/cost characteristic, another three linear characteristics, another two
    cost characteristics, demographics interacted with prices and the linear/nonlinear/cost characteristic, three
    nesting groups with the same nesting parameter, and a log-linear cost specification.
    """
    id_data = build_id_data(T=20, J=20, F=9)

    keep = np.arange(id_data.size)
    np.random.RandomState(0).shuffle(keep)
    id_data = id_data[keep[:int(0.5 * id_data.size)]]

    simulation = Simulation(
        product_formulations=(
            Formulation('1 + x + y + z + q'),
            Formulation('0 + I(-prices) + x'),
            Formulation('0 + log(x) + log(a) + log(b)')
        ),
        product_data={
            'market_ids': id_data.market_ids,
            'firm_ids': id_data.firm_ids,
            'nesting_ids': np.random.RandomState(2).choice(['f', 'g', 'h'], id_data.size),
            'clustering_ids': np.random.RandomState(2).choice(range(30), id_data.size)
        },
        beta=[1, 1, 2, 3, 1],
        sigma=[
            [0.5, 0],
            [0.0, 2]
        ],
        pi=[
            [2, 1, 0],
            [0, 0, 2]
        ],
        gamma=[0.1, 0.2, 0.3],
        rho=0.1,
        agent_formulation=Formulation('1 + f + g'),
        integration=Integration('product', 4),
        xi_variance=0.00001,
        omega_variance=0.00001,
        correlation=0.9,
        distributions=['lognormal', 'normal'],
        costs_type='log',
        seed=2,
    )
    simulation_results = simulation.replace_endogenous()
    simulated_micro_moments = [DemographicExpectationMoment(
        product_id=None, demographics_index=1, value=0, market_ids=simulation.unique_market_ids[3:5]
    )]
    return simulation, simulation_results, {}, simulated_micro_moments


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
    pytest.param(['large_nested_blp', True], id="large nested BLP simulation with supply"),
])
def simulated_problem(request: Any) -> SimulatedProblemFixture:
    """Configure and solve a simulated problem, either with or without supply-side data. Preclude overflow with rho
    bounds that are more conservative than the default ones.
    """
    name, supply = request.param
    simulation, simulation_results, simulated_data_override, simulated_micro_moments = (
        request.getfixturevalue(f'{name}_simulation')
    )

    # override the simulated data
    product_data = None
    if simulated_data_override:
        product_data = update_matrices(
            simulation_results.product_data,
            {k: (v, v.dtype) for k, v in simulated_data_override.items()}
        )

    # compute micro moments
    micro_moments: List[Any] = []
    if simulated_micro_moments:
        micro_values = simulation_results.compute_micro(simulated_micro_moments)
        for moment, value in zip(simulated_micro_moments, micro_values):
            if isinstance(moment, DemographicExpectationMoment):
                micro_moments.append(DemographicExpectationMoment(
                    moment.product_id, moment.demographics_index, value, moment.market_ids
                ))
            elif isinstance(moment, DemographicCovarianceMoment):
                micro_moments.append(DemographicCovarianceMoment(
                    moment.X2_index, moment.demographics_index, value, moment.market_ids
                ))
            elif isinstance(moment, DiversionProbabilityMoment):
                micro_moments.append(DiversionProbabilityMoment(
                    moment.product_id1, moment.product_id2, value, moment.market_ids
                ))
            else:
                assert isinstance(moment, DiversionCovarianceMoment)
                micro_moments.append(DiversionCovarianceMoment(
                    moment.X2_index1, moment.X2_index2, value, moment.market_ids
                ))

    # initialize and solve the problem
    problem = simulation_results.to_problem(simulation.product_formulations[:2 + int(supply)], product_data)
    solve_options = {
        'sigma': simulation.sigma,
        'pi': simulation.pi,
        'rho': simulation.rho,
        'beta': np.where(simulation._parameters.alpha_index, simulation.beta if supply else np.nan, np.nan),
        'rho_bounds': (np.zeros_like(simulation.rho), np.minimum(0.9, 1.5 * simulation.rho)),
        'method': '1s',
        'check_optimality': 'gradient',
        'micro_moments': micro_moments
    }
    problem_results = problem.solve(**solve_options)
    return simulation, simulation_results, problem, solve_options, problem_results


@pytest.fixture(scope='session', params=[pytest.param(1, id="1 observation"), pytest.param(10, id="10 observations")])
def formula_data(request: Any) -> Data:
    """Simulate patsy demo data with two-level categorical variables and varying numbers of observations."""
    raw_data = patsy.user_util.demo_data('a', 'b', 'c', 'x', 'y', 'z', nlevels=2, min_rows=request.param)
    return {k: np.array(v) if isinstance(v[0], str) else np.abs(v) for k, v in raw_data.items()}
