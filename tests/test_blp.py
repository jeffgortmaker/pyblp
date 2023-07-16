"""Primary tests."""

import copy
import itertools
import os
import pickle
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pytest
import scipy.optimize
import scipy.special

import pyblp.exceptions
from pyblp import (
    Formulation, Integration, Iteration, MicroDataset, MicroPart, MicroMoment, Optimization, Problem, Simulation,
    build_ownership, data_to_dict, parallel
)
from pyblp.utilities.basics import (
    Array, Options, RecArray, update_matrices, compute_finite_differences, compute_second_finite_differences
)
from .conftest import SimulatedProblemFixture


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('solve_options_update', [
    pytest.param({'method': '2s'}, id="two-step"),
    pytest.param({'scale_objective': True}, id="scaled objective"),
    pytest.param({'covariance_moments_mean': 1e-14}, id="tiny covariance moments mean"),
    pytest.param({'center_moments': False, 'W_type': 'unadjusted', 'se_type': 'clustered'}, id="complex covariances"),
    pytest.param({'delta_behavior': 'last'}, id="faster starting delta values"),
    pytest.param({'delta_behavior': 'logit'}, id="logit starting delta values"),
    pytest.param({'fp_type': 'linear'}, id="non-safe linear fixed point"),
    pytest.param({'fp_type': 'safe_nonlinear'}, id="nonlinear fixed point"),
    pytest.param({'fp_type': 'nonlinear'}, id="non-safe nonlinear fixed point"),
    pytest.param(
        {'iteration': Iteration('hybr', {'xtol': 1e-12}, compute_jacobian=True, universal_display=True)},
        id="linear Newton fixed point"
    ),
    pytest.param(
        {'fp_type': 'safe_nonlinear', 'iteration': Iteration('hybr', {'xtol': 1e-12}, compute_jacobian=True)},
        id="nonlinear Newton fixed point"
    )
])
def test_accuracy(simulated_problem: SimulatedProblemFixture, solve_options_update: Options) -> None:
    """Test that starting parameters that are half their true values give rise to errors of less than 10%."""
    simulation, _, problem, solve_options, _ = simulated_problem

    # skip different iteration configurations when they won't matter
    if simulation.K2 == 0 and {'delta_behavior', 'fp_type', 'iteration'} & set(solve_options_update):
        return pytest.skip("A different iteration configuration has no impact when there is no heterogeneity.")
    if simulation.epsilon_scale != 1 and 'nonlinear' in solve_options_update.get('fp_type', 'safe_linear'):
        return pytest.skip("Nonlinear fixed point configurations are not supported when epsilon is scaled.")

    # skip the covariance moments value difference when it doesn't matter
    if 'covariance_moments_mean' in solve_options_update and (problem.K3 == 0 or problem.MC == 0):
        return pytest.skip("There is nothing to test with the covariance moments mean with no covariance moments.")

    # update the default options and solve the problem
    updated_solve_options = copy.deepcopy(solve_options)
    updated_solve_options.update(solve_options_update)
    updated_solve_options.update({k: 0.5 * solve_options[k] for k in ['sigma', 'pi', 'rho', 'beta']})
    results = problem.solve(**updated_solve_options)

    # test the accuracy of the estimated parameters
    keys = ['sigma', 'pi', 'rho', 'beta']
    if problem.K3 > 0:
        keys.append('gamma')
    for key in keys:
        np.testing.assert_allclose(getattr(simulation, key), getattr(results, key), atol=0, rtol=0.1, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('compute_options', [
    pytest.param({'method': 'approximate'}, id="approximation"),
    pytest.param({'method': 'normal'}, id="normal distribution"),
    pytest.param({'method': 'empirical'}, id="empirical distribution")
])
def test_optimal_instruments(simulated_problem: SimulatedProblemFixture, compute_options: Options) -> None:
    """Test that starting parameters that are half their true values also give rise to errors of less than 10% under
    optimal instruments.
    """
    simulation, _, problem, solve_options, problem_results = simulated_problem

    # compute optimal instruments and update the problem (only use a few draws to speed up the test)
    compute_options = copy.deepcopy(compute_options)
    compute_options.update({
        'draws': 5,
        'seed': 0
    })
    new_problem = problem_results.compute_optimal_instruments(**compute_options).to_problem()

    # update the default options and solve the problem
    updated_solve_options = copy.deepcopy(solve_options)
    updated_solve_options.update({k: 0.5 * solve_options[k] for k in ['sigma', 'pi', 'rho', 'beta']})
    new_results = new_problem.solve(**updated_solve_options)

    # test the accuracy of the estimated parameters
    keys = ['beta', 'sigma', 'pi', 'rho']
    if problem.K3 > 0:
        keys.append('gamma')
    for key in keys:
        np.testing.assert_allclose(getattr(simulation, key), getattr(new_results, key), atol=0, rtol=0.1, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
def test_importance_sampling(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that starting parameters that are half their true values also give rise to errors of less than 20% under
    importance sampling.
    """
    simulation, _, problem, solve_options, problem_results = simulated_problem

    # importance sampling is only relevant when there are agent data
    if problem.K2 == 0:
        return pytest.skip("There are no agent data.")

    # it suffices to test importance sampling for problems without demographics
    if problem.D > 0:
        return pytest.skip("Testing importance sampling is hard with demographics.")

    # compute a more precise delta
    delta = problem_results.compute_delta(integration=simulation.integration)

    # do importance sampling and verify that the mean utility didn't change if precise integration isn't used
    sampling_results = problem_results.importance_sampling(
        draws=500,
        ar_constant=2,
        seed=0,
        delta=delta,
        integration=Integration('mlhs', 50000, {'seed': 0}),
    )

    # solve the new problem
    new_problem = sampling_results.to_problem()
    updated_solve_options = copy.deepcopy(solve_options)
    updated_solve_options.update({k: 0.5 * solve_options[k] for k in ['sigma', 'pi', 'rho', 'beta']})
    new_results = new_problem.solve(**updated_solve_options)

    # test the accuracy of the estimated parameters
    keys = ['beta', 'sigma', 'pi', 'rho']
    if problem.K3 > 0:
        keys.append('gamma')
    for key in keys:
        np.testing.assert_allclose(getattr(simulation, key), getattr(new_results, key), atol=0, rtol=0.2, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
def test_bootstrap(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that post-estimation output medians are within 5% parametric bootstrap confidence intervals."""
    _, _, problem, solve_options, problem_results = simulated_problem

    # create bootstrapped results (use only a few draws and don't iterate for speed)
    bootstrapped_results = problem_results.bootstrap(draws=100, seed=0, iteration=Iteration('return'))

    # test that post-estimation outputs are within 95% confidence intervals
    t = problem.products.market_ids[0]
    merger_ids = np.where(problem.products.firm_ids == 1, 0, problem.products.firm_ids)
    merger_ids_t = merger_ids[problem.products.market_ids == t]
    method_mapping = {
        "aggregate elasticities": lambda r: r.compute_aggregate_elasticities(),
        "consumer surpluses": lambda r: r.compute_consumer_surpluses(),
        "approximate prices": lambda r: r.compute_approximate_prices(merger_ids),
        "own elasticities": lambda r: r.extract_diagonals(r.compute_elasticities()),
        "own elasticities in t": lambda r: r.extract_diagonals(r.compute_elasticities(market_id=t), market_id=t),
        "aggregate elasticity in t": lambda r: r.compute_aggregate_elasticities(market_id=t),
        "consumer surplus in t": lambda r: r.compute_consumer_surpluses(market_id=t),
        "approximate prices in t": lambda r: r.compute_approximate_prices(merger_ids_t, market_id=t)
    }
    for name, method in method_mapping.items():
        values = method(problem_results)
        bootstrapped_values = method(bootstrapped_results)
        median = np.median(values)
        bootstrapped_medians = np.nanmedian(bootstrapped_values, axis=range(1, bootstrapped_values.ndim))
        lb = np.percentile(bootstrapped_medians, 2.5)
        ub = np.percentile(bootstrapped_medians, 97.5)
        np.testing.assert_array_less(np.squeeze(lb), np.squeeze(median) + 1e-14, err_msg=name)
        np.testing.assert_array_less(np.squeeze(median), np.squeeze(ub) + 1e-14, err_msg=name)


@pytest.mark.usefixtures('simulated_problem')
def test_bootstrap_se(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that bootstrapped SEs are close to analytic ones. Or at least the same order of magnitude -- especially for
    large numbers of RCs they may not necessarily be very close to each other.
    """
    _, _, _, _, problem_results = simulated_problem

    # compute bootstrapped results (ignore supply side iteration because we will only use the parameter draws)
    bootstrapped_results = problem_results.bootstrap(draws=1000, seed=0, iteration=Iteration('return'))

    # compare SEs
    for key in ['sigma', 'pi', 'rho', 'beta', 'gamma']:
        analytic_se = np.nan_to_num(getattr(problem_results, f'{key}_se'))
        bootstrapped_se = getattr(bootstrapped_results, f'bootstrapped_{key}').std(axis=0)
        np.testing.assert_allclose(analytic_se, bootstrapped_se, atol=0.001, rtol=0.5, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
def test_agent_resampling(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that estimating simulation error by resampling agents seems to work."""
    simulation, _, problem, solve_options, _ = simulated_problem

    # skip configurations when they won't matter
    if simulation.K2 == 0:
        return pytest.skip("Agent resampling does nothing when there is no heterogeneity.")

    # resample agent data but trivially to get all zeros
    updated_solve_options = copy.deepcopy(solve_options)
    updated_solve_options.update({
        'optimization': Optimization('return'),
        'resample_agent_data': lambda i: None if i > 3 else simulation.agent_data,
    })
    problem_results1 = problem.solve(**updated_solve_options)
    assert (problem_results1.simulation_covariances == 0).all()

    def resample_agent_data(index: int) -> Optional[RecArray]:
        """Non-trivially resample agent data."""
        if index > 3:
            return None

        assert simulation.agent_data is not None
        resampled_agent_data = simulation.agent_data.copy()
        resampled_weights = simulation.agent_data.weights.copy()
        np.random.default_rng(index).shuffle(resampled_weights)
        resampled_agent_data.weights[:] = resampled_weights
        return resampled_agent_data

    # resample agent data non-trivially
    updated_solve_options['resample_agent_data'] = resample_agent_data
    problem_results2 = problem.solve(**updated_solve_options)
    assert not (problem_results2.simulation_covariances == 0).all()


@pytest.mark.usefixtures('simulated_problem')
def test_micro_chunking(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that micro chunking doesn't substantially change anything."""
    simulation, simulation_results, problem, solve_options, _ = simulated_problem

    # skip configurations without micro moments
    if 'micro_moments' not in solve_options or not solve_options['micro_moments']:
        return pytest.skip("Micro chunking has no effect when there are no micro moments.")

    # get some baseline results
    updated_solve_options = copy.deepcopy(solve_options)
    updated_solve_options['optimization'] = Optimization('return')
    agent_scores1 = simulation_results.compute_agent_scores(
        solve_options['micro_moments'][0].parts[0].dataset, integration=simulation.integration
    )
    problem_results1 = problem.solve(**updated_solve_options)

    # get some results with chunking
    assert pyblp.options.micro_computation_chunks == 1
    for chunks in [2, 3]:
        pyblp.options.micro_computation_chunks = chunks
        agent_scores2 = simulation_results.compute_agent_scores(
            solve_options['micro_moments'][0].parts[0].dataset, integration=simulation.integration
        )
        problem_results2 = problem.solve(**updated_solve_options)
        pyblp.options.micro_computation_chunks = 1

        # verify that all results are very close
        np.testing.assert_allclose(
            list(agent_scores1[0].values())[0], list(agent_scores2[0].values())[0], atol=1e-12, rtol=1e-6
        )
        for key, result in problem_results1.__dict__.items():
            if isinstance(result, np.ndarray) and result.dtype != np.object_:
                np.testing.assert_allclose(result, getattr(problem_results2, key), atol=1e-12, rtol=1e-6, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
def test_profit_gradients(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that if equilibrium prices were computed, profit gradients are nearly zero."""
    _, simulation_results, _, _, _ = simulated_problem

    if simulation_results.profit_gradient_norms is None:
        return pytest.skip("Equilibrium prices were not computed.")

    for t, profit_gradient_norms_t in simulation_results.profit_gradient_norms.items():
        for f, profit_gradient_norm_ft in profit_gradient_norms_t.items():
            assert profit_gradient_norm_ft < 1e-12, (t, f)


@pytest.mark.usefixtures('simulated_problem')
def test_result_serialization(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that result objects can be serialized and that their string representations are the same when they are
    unpickled.
    """
    simulation, simulation_results, problem, solve_options, problem_results = simulated_problem
    originals = [
        Formulation('x + y', absorb='C(z)', absorb_method='lsmr', absorb_options={'tol': 1e-10}),
        Integration('halton', size=10, specification_options={'seed': 0, 'scramble': True}),
        Iteration('lm', method_options={'max_evaluations': 100}, compute_jacobian=True, universal_display=True),
        Optimization('nelder-mead', method_options={'xatol': 1e-5}, compute_gradient=False, universal_display=False),
        problem,
        simulation,
        simulation_results,
        problem_results,
        problem_results.compute_optimal_instruments(),
        problem_results.bootstrap(draws=1, seed=0),
        data_to_dict(simulation_results.product_data),
        solve_options['micro_moments'],
    ]
    for original in originals:
        try:
            unpickled = pickle.loads(pickle.dumps(original))
        except AttributeError as exception:
            # use dill to pickle lambda functions
            if "Can't pickle local object" not in str(exception) or "<lambda>" not in str(exception):
                raise
            import dill
            unpickled = dill.loads(dill.dumps(original))

        assert str(original) == str(unpickled), str(original)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('solve_options_update', [
    pytest.param({'costs_bounds': (-1e10, 1e10)}, id="non-binding costs bounds"),
    pytest.param({'check_optimality': 'both'}, id="Hessian computation")
])
def test_trivial_changes(simulated_problem: SimulatedProblemFixture, solve_options_update: Dict) -> None:
    """Test that solving a problem with arguments that shouldn't give rise to meaningful differences doesn't give rise
    to any differences.
    """
    simulation, _, problem, solve_options, results = simulated_problem

    # solve the problem with the updated options
    updated_solve_options = copy.deepcopy(solve_options)
    updated_solve_options.update(solve_options_update)
    updated_results = problem.solve(**updated_solve_options)

    # test that all arrays in the results are essentially identical
    for key, result in results.__dict__.items():
        if isinstance(result, np.ndarray) and result.dtype != np.object_:
            if 'hessian' not in key:
                np.testing.assert_allclose(result, getattr(updated_results, key), atol=1e-14, rtol=0, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
def test_parallel(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that solving problems and computing results in parallel gives rise to the same results as when using serial
    processing.
    """
    _, _, problem, solve_options, results = simulated_problem

    # compute marginal costs as a test of results (everything else has already been computed without parallelization)
    costs = results.compute_costs()

    # solve the problem and compute costs in parallel
    try:
        with parallel(2):
            parallel_results = problem.solve(**solve_options)
            parallel_costs = parallel_results.compute_costs()
    except RuntimeError as exception:
        # use dill via pathos to handle lambda function pickling
        if "multiprocessing module does not support lambda functions" not in str(exception):
            raise
        with parallel(2, use_pathos=True):
            parallel_results = problem.solve(**solve_options)
            parallel_costs = parallel_results.compute_costs()

    # test that all arrays in the results are essentially identical
    for key, result in results.__dict__.items():
        if isinstance(result, np.ndarray) and result.dtype != np.object_:
            np.testing.assert_allclose(result, getattr(parallel_results, key), atol=1e-14, rtol=0, err_msg=key)

    # test that marginal costs are essentially equal
    np.testing.assert_allclose(costs, parallel_costs, atol=1e-14, rtol=0)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize(['ED', 'ES', 'absorb_method', 'absorb_options'], [
    pytest.param(1, 0, None, None, id="1 demand FE, default method"),
    pytest.param(0, 1, None, None, id="1 supply FE, default method"),
    pytest.param(1, 1, None, None, id="1 demand- and 1 supply FE, default method"),
    pytest.param(2, 0, None, None, id="2 demand FEs, default method"),
    pytest.param(0, 2, 'sw', None, id="2 supply FEs, SW"),
    pytest.param(3, 1, 'lsmr', None, id="3 demand- and 1 supply FEs, LSMR"),
    pytest.param(1, 3, 'map', {'transform': 'cimmino', 'acceleration': 'cg'}, id="1 demand- and 3 supply FEs, MAP-CG"),
])
def test_fixed_effects(
        simulated_problem: SimulatedProblemFixture, ED: int, ES: int, absorb_method: Optional[str],
        absorb_options: Optional[dict]) -> None:
    """Test that absorbing different numbers of demand- and supply-side fixed effects gives rise to essentially
    identical first-stage results as does including indicator variables. Also test that optimal instruments results,
    marginal costs, and test statistics remain unchanged.
    """
    simulation, simulation_results, problem, solve_options, problem_results = simulated_problem

    # there cannot be supply-side fixed effects if there isn't a supply side
    if problem.K3 == 0:
        ES = 0
    if ED == ES == 0:
        return pytest.skip("There are no fixed effects to test.")

    # configure the optimization routine to only do a few iterations to save time and never get to the point where small
    #   numerical differences between methods build up into noticeable differences
    solve_options = copy.deepcopy(solve_options)
    solve_options['optimization'] = Optimization('l-bfgs-b', {'maxfun': 3})

    # make product data mutable and add instruments
    product_data = {k: simulation_results.product_data[k] for k in simulation_results.product_data.dtype.names}
    product_data.update({
        'demand_instruments': problem.products.ZD[:, :-problem.K1],
        'supply_instruments': problem.products.ZS[:, :-problem.K3],
        'covariance_instruments': problem.products.ZC,
    })

    # remove constants and delete associated elements in the initial beta
    product_formulations = list(problem.product_formulations).copy()
    if ED > 0:
        assert product_formulations[0] is not None
        constant_indices = [i for i, e in enumerate(product_formulations[0]._expressions) if not e.free_symbols]
        solve_options['beta'] = np.delete(solve_options['beta'], constant_indices, axis=0)
        product_formulations[0] = Formulation(f'{product_formulations[0]._formula} - 1')
    if ES > 0:
        assert product_formulations[2] is not None
        product_formulations[2] = Formulation(f'{product_formulations[2]._formula} - 1')

    # add fixed effect IDs to the data
    demand_id_names: List[str] = []
    supply_id_names: List[str] = []
    state = np.random.RandomState(seed=0)
    for side, count, names in [('demand', ED, demand_id_names), ('supply', ES, supply_id_names)]:
        for index in range(count):
            name = f'{side}_ids{index}'
            ids = state.choice(['a', 'b', 'c'], problem.N)
            product_data[name] = ids
            names.append(name)

    # split apart excluded demand-side instruments so they can be included in formulations
    instrument_names: List[str] = []
    for index, instrument in enumerate(product_data['demand_instruments'].T):
        name = f'demand_instrument{index}'
        product_data[name] = instrument
        instrument_names.append(name)

    # build formulas for the IDs
    demand_id_formula = ' + '.join(demand_id_names)
    supply_id_formula = ' + '.join(supply_id_names)

    # solve the first stage of a problem in which the fixed effects are absorbed
    solve_options1 = copy.deepcopy(solve_options)
    product_formulations1 = product_formulations.copy()
    if ED > 0:
        assert product_formulations[0] is not None
        product_formulations1[0] = Formulation(
            product_formulations[0]._formula, demand_id_formula, absorb_method, absorb_options
        )
    if ES > 0:
        assert product_formulations[2] is not None
        product_formulations1[2] = Formulation(
            product_formulations[2]._formula, supply_id_formula, absorb_method, absorb_options
        )
    problem1 = Problem(
        product_formulations1, product_data, problem.agent_formulation, simulation.agent_data,
        rc_types=simulation.rc_types, epsilon_scale=simulation.epsilon_scale, costs_type=simulation.costs_type
    )
    if solve_options1['micro_moments']:
        solve_options1['W'] = scipy.linalg.pinv(scipy.linalg.block_diag(
            problem1.products.ZD.T @ problem1.products.ZD,
            problem1.products.ZS.T @ problem1.products.ZS,
            problem1.products.ZC.T @ problem1.products.ZC,
            np.eye(len(solve_options1['micro_moments'])),
        ))
    problem_results1 = problem1.solve(**solve_options1)

    # solve the first stage of a problem in which fixed effects are included as indicator variables
    solve_options2 = copy.deepcopy(solve_options)
    product_formulations2 = product_formulations.copy()
    if ED > 0:
        assert product_formulations[0] is not None
        product_formulations2[0] = Formulation(f'{product_formulations[0]._formula} + {demand_id_formula}')
    if ES > 0:
        assert product_formulations[2] is not None
        product_formulations2[2] = Formulation(f'{product_formulations[2]._formula} + {supply_id_formula}')
    problem2 = Problem(
        product_formulations2, product_data, problem.agent_formulation, simulation.agent_data,
        rc_types=simulation.rc_types, epsilon_scale=simulation.epsilon_scale, costs_type=simulation.costs_type
    )
    solve_options2['beta'] = np.r_[
        solve_options2['beta'],
        np.full((problem2.K1 - solve_options2['beta'].size, 1), np.nan)
    ]
    if solve_options2['micro_moments']:
        solve_options2['W'] = scipy.linalg.pinv(scipy.linalg.block_diag(
            problem2.products.ZD.T @ problem2.products.ZD,
            problem2.products.ZS.T @ problem2.products.ZS,
            problem2.products.ZC.T @ problem2.products.ZC,
            np.eye(len(solve_options2['micro_moments'])),
        ))
    problem_results2 = problem2.solve(**solve_options2)

    # solve the first stage of a problem in which some fixed effects are absorbed and some are included as indicators
    if ED == ES == 0:
        problem_results3 = problem_results2
    else:
        solve_options3 = copy.deepcopy(solve_options)
        product_formulations3 = product_formulations.copy()
        if ED > 0:
            assert product_formulations[0] is not None
            product_formulations3[0] = Formulation(
                f'{product_formulations[0]._formula} + {demand_id_names[0]}', ' + '.join(demand_id_names[1:]) or None
            )
        if ES > 0:
            assert product_formulations[2] is not None
            product_formulations3[2] = Formulation(
                f'{product_formulations[2]._formula} + {supply_id_names[0]}', ' + '.join(supply_id_names[1:]) or None
            )
        problem3 = Problem(
            product_formulations3, product_data, problem.agent_formulation, simulation.agent_data,
            rc_types=simulation.rc_types, epsilon_scale=simulation.epsilon_scale, costs_type=simulation.costs_type
        )
        solve_options3['beta'] = np.r_[
            solve_options3['beta'],
            np.full((problem3.K1 - solve_options3['beta'].size, 1), np.nan)
        ]
        if solve_options3['micro_moments']:
            solve_options3['W'] = scipy.linalg.pinv(scipy.linalg.block_diag(
                problem3.products.ZD.T @ problem3.products.ZD,
                problem3.products.ZS.T @ problem3.products.ZS,
                problem3.products.ZC.T @ problem3.products.ZC,
                np.eye(len(solve_options3['micro_moments'])),
            ))
        problem_results3 = problem3.solve(**solve_options3)

    # compute optimal instruments (use only two draws for speed; accuracy is not a concern here)
    Z_results1 = problem_results1.compute_optimal_instruments(draws=2, seed=0)
    Z_results2 = problem_results2.compute_optimal_instruments(draws=2, seed=0)
    Z_results3 = problem_results3.compute_optimal_instruments(draws=2, seed=0)

    # compute marginal costs
    costs1 = problem_results1.compute_costs()
    costs2 = problem_results2.compute_costs()
    costs3 = problem_results3.compute_costs()
    J1 = problem_results1.run_hansen_test()
    J2 = problem_results2.run_hansen_test()
    J3 = problem_results3.run_hansen_test()
    LR1 = problem_results1.run_distance_test(problem_results)
    LR2 = problem_results2.run_distance_test(problem_results)
    LR3 = problem_results3.run_distance_test(problem_results)
    LM1 = problem_results1.run_lm_test()
    LM2 = problem_results2.run_lm_test()
    LM3 = problem_results3.run_lm_test()
    wald1 = problem_results1.run_wald_test(
        problem_results1.parameters[:2], np.eye(problem_results1.parameters.size)[:2]
    )
    wald2 = problem_results2.run_wald_test(
        problem_results2.parameters[:2], np.eye(problem_results2.parameters.size)[:2]
    )
    wald3 = problem_results3.run_wald_test(
        problem_results3.parameters[:2], np.eye(problem_results3.parameters.size)[:2]
    )

    # choose tolerances
    atol = 1e-8
    rtol = 1e-5

    # test that fixed effect estimates are essentially identical
    if ED == 0:
        assert (problem_results1.xi_fe == 0).all()
    else:
        xi_fe2 = problem2.products.X1[:, problem1.K1:] @ problem_results2.beta[problem1.K1:]
        np.testing.assert_allclose(problem_results1.xi_fe, xi_fe2, atol=atol, rtol=rtol)
    if ES == 0:
        assert (problem_results1.omega_fe == 0).all()
    else:
        omega_fe2 = problem2.products.X3[:, problem1.K3:] @ problem_results2.gamma[problem1.K3:]
        np.testing.assert_allclose(problem_results1.omega_fe, omega_fe2, atol=atol, rtol=rtol)

    # test that all problem results expected to be identical are essentially identical, except for standard errors under
    #   micro moments, which are expected to be slightly different
    problem_results_keys = [
        'theta', 'sigma', 'pi', 'rho', 'beta', 'gamma', 'sigma_se', 'pi_se', 'rho_se', 'beta_se', 'gamma_se',
        'delta', 'tilde_costs', 'xi', 'omega', 'xi_by_theta_jacobian', 'omega_by_theta_jacobian', 'objective',
        'gradient', 'projected_gradient'
    ]
    for key in problem_results_keys:
        if key.endswith('_se') and solve_options['micro_moments']:
            continue
        result1 = getattr(problem_results1, key)
        result2 = getattr(problem_results2, key)
        result3 = getattr(problem_results3, key)
        if key in {'beta', 'gamma', 'beta_se', 'gamma_se'}:
            result2 = result2[:result1.size]
            result3 = result3[:result1.size]
        np.testing.assert_allclose(result1, result2, atol=atol, rtol=rtol, err_msg=key, equal_nan=True)
        np.testing.assert_allclose(result1, result3, atol=atol, rtol=rtol, err_msg=key, equal_nan=True)

    # test that all optimal instrument results expected to be identical are essentially identical
    Z_results_keys = [
        'demand_instruments', 'supply_instruments', 'inverse_covariance_matrix', 'expected_xi_by_theta_jacobian',
        'expected_omega_by_theta_jacobian'
    ]
    for key in Z_results_keys:
        result1 = getattr(Z_results1, key)
        result2 = getattr(Z_results2, key)
        result3 = getattr(Z_results3, key)
        np.testing.assert_allclose(result1, result2, atol=atol, rtol=rtol, err_msg=key)
        np.testing.assert_allclose(result1, result3, atol=atol, rtol=rtol, err_msg=key)

    # test that marginal costs and test statistics are essentially identical
    np.testing.assert_allclose(costs1, costs2, atol=atol, rtol=rtol)
    np.testing.assert_allclose(costs1, costs3, atol=atol, rtol=rtol)
    np.testing.assert_allclose(J1, J2, atol=atol, rtol=rtol)
    np.testing.assert_allclose(J1, J3, atol=atol, rtol=rtol)
    np.testing.assert_allclose(LR1, LR2, atol=atol, rtol=rtol)
    np.testing.assert_allclose(LR1, LR3, atol=atol, rtol=rtol)
    np.testing.assert_allclose(LM1, LM2, atol=atol, rtol=rtol)
    np.testing.assert_allclose(LM1, LM3, atol=atol, rtol=rtol)
    np.testing.assert_allclose(wald1, wald2, atol=atol, rtol=rtol)
    np.testing.assert_allclose(wald1, wald3, atol=atol, rtol=rtol)


@pytest.mark.usefixtures('simulated_problem')
def test_special_ownership(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that ownership matrices constructed according to special cases take on expected forms."""
    simulation, simulation_results, _, _, _ = simulated_problem

    # test monopoly ownership matrices
    monopoly_ownership1 = build_ownership(simulation_results.product_data, 'monopoly')
    monopoly_ownership2 = build_ownership(simulation_results.product_data, lambda f, g: 1)
    np.testing.assert_equal(monopoly_ownership1, monopoly_ownership2)
    assert (monopoly_ownership1[~np.isnan(monopoly_ownership1)] == 1).all()

    # test single product ownership matrices
    single_ownership = build_ownership(simulation_results.product_data, 'single')
    assert np.nansum(single_ownership) == simulation.N


@pytest.mark.usefixtures('simulated_problem')
def test_specified_ownership(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that marginal costs computed under specified firm IDs and ownership are the same as costs computed when
    firm IDs and ownership are left unspecified.
    """
    _, simulation_results, _, _, results = simulated_problem

    # compute costs in the simplest way possible
    costs1 = results.compute_costs()

    # under custom ownership, just test that results are the same under ownership specification
    if simulation_results.product_data.ownership.size > 0:
        costs2 = results.compute_costs(ownership=simulation_results.product_data.ownership)
        np.testing.assert_equal(costs1, costs2)
        return

    # otherwise, also test that results are the same under a firm IDs specification
    costs2 = results.compute_costs(firm_ids=simulation_results.product_data.firm_ids)
    costs3 = results.compute_costs(ownership=build_ownership(simulation_results.product_data))
    np.testing.assert_equal(costs1, costs2)
    np.testing.assert_equal(costs1, costs3)


@pytest.mark.usefixtures('simulated_problem')
def test_costs(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that true marginal costs can be recovered after estimation at the true parameter values. This is essentially
    a joint test of the zeta-markup approach to computing equilibrium prices and the eta-markup expression.
    """
    _, simulation_results, _, _, _ = simulated_problem
    costs = simulation_results.compute_costs()
    np.testing.assert_allclose(costs, simulation_results.costs, atol=1e-10, rtol=1e-10)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('iteration', [
    pytest.param(Iteration('simple', {'atol': 1e-12}, universal_display=True), id="simple"),
    pytest.param(Iteration('hybr', {'xtol': 1e-12, 'ftol': 0}, compute_jacobian=True), id="Powell/LM"),
])
def test_prices(simulated_problem: SimulatedProblemFixture, iteration: Iteration) -> None:
    """Test that equilibrium prices computed with different methods are approximate the same.
    """
    simulation, simulation_results, _, _, _ = simulated_problem

    # skip simulations that didn't compute equilibrium prices
    if simulation_results.profit_gradient_norms is None:
        return pytest.skip("Equilibrium prices were not computed.")

    try:
        new_simulation_results = simulation.replace_endogenous(iteration=iteration, constant_costs=False)
    except (pyblp.exceptions.SyntheticPricesConvergenceError, pyblp.exceptions.MultipleErrors):
        # sometimes Powell won't converge at these starting values but LM will
        if 'hybr' not in str(iteration):
            raise
        iteration = Iteration('lm', {'xtol': 1e-12, 'ftol': 0}, compute_jacobian=True)
        new_simulation_results = simulation.replace_endogenous(iteration=iteration, constant_costs=False)

    np.testing.assert_allclose(
        simulation_results.product_data.prices,
        new_simulation_results.product_data.prices,
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('ownership', [
    pytest.param(False, id="firm IDs change"),
    pytest.param(True, id="ownership change")
])
@pytest.mark.parametrize('solve_options', [
    pytest.param({}, id="defaults"),
    pytest.param({'iteration': Iteration('simple')}, id="configured iteration"),
])
def test_merger(simulated_problem: SimulatedProblemFixture, ownership: bool, solve_options: Options) -> None:
    """Test that prices and shares simulated under changed firm IDs are reasonably close to prices and shares computed
    from the results of a solved problem. In particular, test that unchanged prices and shares are farther from their
    simulated counterparts than those computed by approximating a merger, which in turn are farther from their simulated
    counterparts than those computed by fully solving a merger. Also test that simple acquisitions increase HHI. These
    inequalities are only guaranteed because of the way in which the simulations are configured.
    """
    simulation, simulation_results, problem, _, results = simulated_problem

    # skip simulations that complicate the test
    if simulation.products.ownership.size > 0:
        return pytest.skip("Merger testing doesn't work with custom ownership.")
    if 'shares' in str(simulation.product_formulations[2]):
        return pytest.skip("Merger testing doesn't work with quantity-dependent costs.")

    # create changed ownership or firm IDs associated with a merger
    merger_product_data = copy.deepcopy(simulation_results.product_data)
    if ownership:
        merger_ids = None
        merger_ownership = build_ownership(merger_product_data, lambda f, g: 1 if f == g or (f < 2 and g < 2) else 0)
        merger_product_data = update_matrices(merger_product_data, {
            'ownership': (merger_ownership, merger_ownership.dtype)
        })
    else:
        merger_ownership = None
        merger_product_data.firm_ids[merger_product_data.firm_ids < 2] = 0
        merger_ids = merger_product_data.firm_ids

    # get actual prices and shares
    merger_simulation = Simulation(
        simulation.product_formulations, merger_product_data, simulation.beta, simulation.sigma, simulation.pi,
        simulation.gamma, simulation.rho, simulation.agent_formulation,
        simulation.agent_data, xi=simulation.xi, omega=simulation.omega, rc_types=simulation.rc_types,
        epsilon_scale=simulation.epsilon_scale, costs_type=simulation.costs_type
    )
    actual = merger_simulation.replace_endogenous(**solve_options)

    # compute marginal costs; get estimated prices and shares
    costs = results.compute_costs()
    results_simulation = Simulation(
        simulation.product_formulations[:2], merger_product_data, results.beta, results.sigma, results.pi,
        rho=results.rho, agent_formulation=simulation.agent_formulation, agent_data=simulation.agent_data,
        xi=results.xi, rc_types=simulation.rc_types, epsilon_scale=simulation.epsilon_scale
    )
    estimated = results_simulation.replace_endogenous(costs, problem.products.prices, **solve_options)
    estimated_prices = results.compute_prices(merger_ids, merger_ownership, costs, **solve_options)
    approximated_prices = results.compute_approximate_prices(merger_ids, merger_ownership, costs)
    estimated_shares = results.compute_shares(estimated_prices)
    approximated_shares = results.compute_shares(approximated_prices)

    # test that we get the same results from solving the simulation
    np.testing.assert_allclose(estimated.product_data.prices, estimated_prices, atol=1e-14, rtol=0, verbose=True)
    np.testing.assert_allclose(estimated.product_data.shares, estimated_shares, atol=1e-14, rtol=0, verbose=True)

    # test that estimated prices are closer to changed prices than approximate prices
    approximated_prices_error = np.linalg.norm(actual.product_data.prices - approximated_prices)
    estimated_prices_error = np.linalg.norm(actual.product_data.prices - estimated_prices)
    np.testing.assert_array_less(estimated_prices_error, approximated_prices_error, verbose=True)

    # test that estimated shares are closer to changed shares than approximate shares
    approximated_shares_error = np.linalg.norm(actual.product_data.shares - approximated_shares)
    estimated_shares_error = np.linalg.norm(actual.product_data.shares - estimated_shares)
    np.testing.assert_array_less(estimated_shares_error, approximated_shares_error, verbose=True)


@pytest.mark.usefixtures('simulated_problem')
def test_shares(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that shares computed from estimated parameters are essentially equal to actual shares."""
    simulation, simulation_results, _, _, results = simulated_problem
    shares1 = results.compute_shares()
    shares2 = results.compute_shares(agent_data=simulation.agent_data, delta=results.delta)
    np.testing.assert_allclose(simulation_results.product_data.shares, shares1, atol=1e-14, rtol=0, verbose=True)
    np.testing.assert_allclose(simulation_results.product_data.shares, shares2, atol=1e-14, rtol=0, verbose=True)


@pytest.mark.usefixtures('simulated_problem')
def test_probabilities(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that integrating over choice probabilities computed from estimated parameters essentially gives actual
    shares.
    """
    _, simulation_results, problem, _, results = simulated_problem

    # only do the test for a single market
    t = problem.products.market_ids[0]
    shares = problem.products.shares[problem.products.market_ids.flat == t]
    prices = problem.products.prices[problem.products.market_ids.flat == t]
    delta = results.delta[problem.products.market_ids.flat == t]
    weights = problem.agents.weights[problem.agents.market_ids.flat == t]

    # compute and compare shares
    estimated_shares1 = results.compute_probabilities(market_id=t) @ weights
    estimated_shares2 = results.compute_probabilities(market_id=t, prices=prices, delta=delta) @ weights
    np.testing.assert_allclose(shares, estimated_shares1, atol=1e-14, rtol=0, verbose=True)
    np.testing.assert_allclose(shares, estimated_shares2, atol=1e-14, rtol=0, verbose=True)


@pytest.mark.usefixtures('simulated_problem')
def test_surplus(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that integrating over individual-level surpluses gives market-level surpluses."""
    _, _, problem, _, results = simulated_problem

    # compute surpluses for a single market
    t = problem.products.market_ids[0]
    surpluses = results.compute_consumer_surpluses(market_id=t, keep_all=True)
    surplus = results.compute_consumer_surpluses(market_id=t)

    # test that we get the same result when manually integrating over surpluses
    weights = problem.agents.weights[problem.agents.market_ids.flat == t]
    np.testing.assert_allclose(surpluses @ weights, surplus, atol=1e-14, rtol=0, verbose=True)


@pytest.mark.usefixtures('simulated_problem')
def test_demand_jacobian(simulated_problem: SimulatedProblemFixture) -> None:
    """Use central finite differences to test that analytic values in the Jacobian of shares with respect to prices are
    essentially correct. Also test that scaled elasticities are exactly equal to this Jacobian.
    """
    simulation, simulation_results, _, _, results = simulated_problem
    product_data = simulation_results.product_data

    # only do the test for a single market
    t = product_data.market_ids[0]
    shares = product_data.shares[product_data.market_ids.flat == t]
    prices = product_data.prices[product_data.market_ids.flat == t]

    # extract the Jacobian from the analytic expression for elasticities and approximate it with finite differences
    exact1 = results.compute_demand_jacobians(market_id=t)
    exact2 = results.compute_elasticities(market_id=t) * shares / prices.T
    approximate = compute_finite_differences(lambda p: results.compute_shares(p, market_id=t), prices)
    np.testing.assert_allclose(exact1, exact2, atol=1e-15, rtol=0)
    np.testing.assert_allclose(exact1, approximate, atol=1e-8, rtol=0)


@pytest.mark.usefixtures('simulated_problem')
def test_demand_hessian(simulated_problem: SimulatedProblemFixture) -> None:
    """Use central finite differences to test that analytic values in the Hessian of shares with respect to prices are
    essentially correct.
    """
    simulation, simulation_results, problem, _, results = simulated_problem
    product_data = simulation_results.product_data

    # only do the test for a single market
    t = product_data.market_ids[0]
    prices = product_data.prices[product_data.market_ids.flat == t]
    xi = results.xi[product_data.market_ids.flat == t]
    if simulation.agent_data is None:
        agent_data = None
    else:
        agent_data = simulation.agent_data[simulation.agent_data.market_ids.flat == t]

    def compute_perturbed_jacobian(perturbed_prices: Array) -> Array:
        """Compute the true Jacobian at perturbed prices."""
        perturbed_product_data = copy.deepcopy(product_data[product_data.market_ids.flat == t])
        perturbed_product_data.prices[:] = perturbed_prices
        perturbed_simulation = Simulation(
            problem.product_formulations[:2], perturbed_product_data, results.beta, results.sigma, results.pi,
            rho=results.rho, agent_formulation=simulation.agent_formulation, agent_data=agent_data,
            xi=xi, rc_types=problem.rc_types, epsilon_scale=problem.epsilon_scale, costs_type=problem.costs_type
        )
        perturbed_simulation_results = perturbed_simulation.replace_endogenous(
            costs=perturbed_prices, prices=perturbed_prices, iteration=Iteration('return')
        )
        return perturbed_simulation_results.compute_demand_jacobians()

    # compute analytic Hessians and approximate them with finite differences
    exact = results.compute_demand_hessians(market_id=t)
    approximate1 = compute_finite_differences(compute_perturbed_jacobian, prices)
    approximate2 = compute_second_finite_differences(lambda p: results.compute_shares(p, market_id=t), prices)
    np.testing.assert_allclose(exact, approximate1, atol=1e-7, rtol=0)
    np.testing.assert_allclose(exact, approximate2, atol=1e-5, rtol=0)


@pytest.mark.usefixtures('simulated_problem')
def test_profit_hessian(simulated_problem: SimulatedProblemFixture) -> None:
    """Use central finite differences to test that analytic values in the Hessian of profits with respect to prices are
    essentially correct.
    """
    simulation, simulation_results, _, _, results = simulated_problem
    product_data = simulation_results.product_data

    # only do the test for a single market
    t = product_data.market_ids[0]
    prices = product_data.prices[product_data.market_ids.flat == t]
    costs = results.compute_costs(market_id=t)

    # compute the exact Hessian and approximat it with finite differences
    exact = results.compute_profit_hessians(costs=costs, market_id=t)
    approximate = compute_second_finite_differences(
        lambda p: results.compute_profits(p, results.compute_shares(p, market_id=t), costs, market_id=t),
        prices,
    )
    np.testing.assert_allclose(exact, approximate, atol=1e-6, rtol=0)


@pytest.mark.usefixtures('simulated_problem')
def test_passthrough(simulated_problem: SimulatedProblemFixture) -> None:
    """Use central finite differences to test that analytic values in the passthrough matrix of prices with respect to
    marginal costs are essentially correct.
    """
    simulation, simulation_results, problem, _, _ = simulated_problem
    product_data = simulation_results.product_data

    # obtain results at the true parameter values so there aren't issues with FOCs not matching
    true_results = problem.solve(
        beta=simulation.beta, sigma=simulation.sigma, pi=simulation.pi, rho=simulation.rho,
        gamma=simulation.gamma if problem.K3 > 0 else None, delta=simulation_results.delta, method='1s',
        check_optimality='gradient', optimization=Optimization('return'), iteration=Iteration('return')
    )

    # only do the test for a single market
    t = int(product_data.market_ids[0])
    costs = true_results.compute_costs(market_id=t)

    # use a looser tolerance if prices enter nonlinearly into utility (this is harder to approximate)
    atol = 1e-6
    for key in ['X1', 'X2']:
        for formulation in problem.products.dtype.fields[key][2]:
            if any(s.name == 'prices' for s in formulation.differentiate('prices').free_symbols):
                atol = 2e-3

    # compute the exact passthrough matrix and approximate it with finite differences
    exact = true_results.compute_passthrough(market_id=t)
    approximate = compute_finite_differences(
        lambda c: true_results.compute_prices(costs=c, market_id=t, iteration=Iteration('simple', {'atol': 1e-16})),
        costs,
    )
    np.testing.assert_allclose(exact, approximate, atol=atol, rtol=0)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('factor', [pytest.param(0.01, id="large"), pytest.param(0.0001, id="small")])
def test_elasticity_aggregates_and_means(simulated_problem: SimulatedProblemFixture, factor: float) -> None:
    """Test that the magnitude of simulated aggregate elasticities is less than the magnitude of mean elasticities, both
    for prices and for other characteristics.
    """
    simulation, _, _, _, results = simulated_problem

    # test that demand for an entire product category is less elastic for prices than for individual products
    np.testing.assert_array_less(
        np.abs(results.compute_aggregate_elasticities(factor)),
        np.abs(results.extract_diagonal_means(results.compute_elasticities())),
        verbose=True
    )

    # test the same inequality but for all non-price variables (including the mean utility)
    names = {n for f in simulation._X1_formulations + simulation._X2_formulations for n in f.names}
    for name in names - {'prices'} | {None}:
        np.testing.assert_array_less(
            np.abs(results.compute_aggregate_elasticities(factor, name)),
            np.abs(results.extract_diagonal_means(results.compute_elasticities(name))),
            err_msg=str(name),
            verbose=True
        )


@pytest.mark.usefixtures('simulated_problem')
def test_diversion_ratios(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that simulated diversion ratio rows sum to one."""
    simulation, _, _, _, results = simulated_problem

    # only do the test for a single market
    t = simulation.products.market_ids[0]

    # test price-based ratios
    ratios = results.compute_diversion_ratios(market_id=t)
    long_run_ratios = results.compute_long_run_diversion_ratios(market_id=t)
    np.testing.assert_allclose(ratios.sum(axis=1), 1, atol=1e-14, rtol=0)
    np.testing.assert_allclose(long_run_ratios.sum(axis=1), 1, atol=1e-14, rtol=0)

    # test ratios based on other variables (including mean utilities)
    names = {n for f in simulation._X1_formulations + simulation._X2_formulations for n in f.names}
    for name in names - {'prices'} | {None}:
        ratios = results.compute_diversion_ratios(name, market_id=t)
        np.testing.assert_allclose(ratios.sum(axis=1), 1, atol=1e-14, rtol=0, err_msg=str(name))


@pytest.mark.usefixtures('simulated_problem')
def test_result_positivity(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that simulated markups, profits, consumer surpluses are positive, both before and after a merger."""
    simulation, _, _, _, results = simulated_problem

    # only do the test for a single market
    t = simulation.products.market_ids[0]

    # compute post-merger prices and shares
    changed_prices = results.compute_approximate_prices(market_id=t)
    changed_shares = results.compute_shares(changed_prices, market_id=t)

    # compute surpluses and test positivity
    test_positive = lambda x: np.testing.assert_array_less(-1e-14, x, verbose=True)
    test_positive(results.compute_markups(market_id=t))
    test_positive(results.compute_profits(market_id=t))
    test_positive(results.compute_consumer_surpluses(market_id=t))
    test_positive(results.compute_markups(changed_prices, market_id=t))
    test_positive(results.compute_profits(changed_prices, changed_shares, market_id=t))
    test_positive(results.compute_consumer_surpluses(changed_prices, market_id=t))

    # compute willingness to pay when the simulation has product IDs and test its positivity
    if simulation.products.product_ids.size > 0:
        unique_product_ids = np.unique(simulation.products.product_ids[simulation.products.market_ids.flat == t, 0])
        eliminate0 = results.compute_consumer_surpluses(market_id=t)
        eliminate1 = results.compute_consumer_surpluses(market_id=t, eliminate_product_ids=unique_product_ids[:1])
        eliminate2 = results.compute_consumer_surpluses(market_id=t, eliminate_product_ids=unique_product_ids[:2])
        test_positive(eliminate0 - eliminate1)
        test_positive(eliminate1 - eliminate2)


@pytest.mark.usefixtures('simulated_problem')
def test_second_step(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that results from two-step GMM on simulated data are identical to results from one-step GMM configured with
    results from a first step.
    """
    simulation, _, problem, solve_options, _ = simulated_problem

    # use two steps and remove sigma bounds so that it can't get stuck at zero (use a small number of optimization
    #   iterations to speed up the test)
    updated_solve_options = copy.deepcopy(solve_options)
    updated_solve_options.update({
        'method': '2s',
        'optimization': Optimization('l-bfgs-b', {'maxfun': 3}),
        'sigma_bounds': (np.full_like(simulation.sigma, -np.inf), np.full_like(simulation.sigma, +np.inf)),
    })

    # get two-step GMM results
    results12 = problem.solve(**updated_solve_options)
    assert results12.last_results is not None
    assert results12.last_results.last_results is None or results12.last_results.last_results.step == 0
    assert results12.step == 2 and results12.last_results.step == 1

    # get results from the first step
    updated_solve_options1 = copy.deepcopy(updated_solve_options)
    updated_solve_options1['method'] = '1s'
    results1 = problem.solve(**updated_solve_options1)

    # get results from the second step
    updated_solve_options2 = copy.deepcopy(updated_solve_options1)
    updated_solve_options2.update({
        'sigma': results1.sigma,
        'pi': results1.pi,
        'rho': results1.rho,
        'beta': np.where(np.isnan(solve_options['beta']), np.nan, results1.beta),
        'delta': results1.delta,
        'W': results1.updated_W,
    })
    results2 = problem.solve(**updated_solve_options2)
    assert results1.last_results is None or results1.last_results.step == 0
    assert results2.last_results is None or results2.last_results.step == 0
    assert results1.step == results2.step == 1

    # test that results are essentially identical
    for key, result12 in results12.__dict__.items():
        if 'cumulative' not in key and isinstance(result12, np.ndarray) and result12.dtype != np.object_:
            np.testing.assert_allclose(result12, getattr(results2, key), atol=1e-14, rtol=0, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
def test_return(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that using a trivial optimization and fixed point iteration routines that just return initial values yield
    results that are the same as the specified initial values.
    """
    simulation, simulation_results, problem, solve_options, _ = simulated_problem

    # specify initial values and the trivial routines
    initial_values = {
        'sigma': simulation.sigma,
        'pi': simulation.pi,
        'rho': simulation.rho,
        'beta': simulation.beta,
        'gamma': simulation.gamma if problem.K3 > 0 else None,
        'delta': problem.products.X1 @ simulation.beta + simulation.xi
    }
    updated_solve_options = copy.deepcopy(solve_options)
    updated_solve_options.update({
        'optimization': Optimization('return'),
        'iteration': Iteration('return'),
        **initial_values
    })

    # obtain problem results and test that initial values are the same
    results = problem.solve(**updated_solve_options)
    for key, initial in initial_values.items():
        if initial is not None:
            np.testing.assert_allclose(initial, getattr(results, key), atol=1e-14, rtol=1e-14, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('scipy_method', [
    pytest.param('l-bfgs-b', id="L-BFGS-B"),
    pytest.param('trust-constr', id="Trust Region")
])
def test_gradient_optionality(simulated_problem: SimulatedProblemFixture, scipy_method: str) -> None:
    """Test that the option of not computing the gradient for simulated data does not affect estimates when the gradient
    isn't used.
    """
    simulation, _, problem, solve_options, results = simulated_problem

    # this test doesn't work when there's a supply side because we use an inverse instead of solving a linear system
    #   directly to save on computation when computing gradients
    if problem.K3 > 0:
        return pytest.skip("Results are expected to change when there's a supply side.")

    # this test only requires a few optimization iterations (enough for gradient problems to be clear)
    method_options = {'maxiter': 2}

    def custom_method(
            initial: Array, bounds: List[Tuple[float, float]], objective_function: Callable, _: Any) -> (
            Tuple[Array, bool]):
        """Optimize without gradients."""
        optimize_results = scipy.optimize.minimize(
            lambda x: objective_function(x)[2].objective, initial, method=scipy_method, bounds=bounds,
            options=method_options
        )
        return optimize_results.x, optimize_results.success

    # solve the problem when not using gradients and when not computing them (use the identity weighting matrix to make
    #   tiny gradients with some initial weighting matrices less problematic when comparing values)
    updated_solve_options1 = copy.deepcopy(solve_options)
    updated_solve_options2 = copy.deepcopy(solve_options)
    updated_solve_options1.update({
        'optimization': Optimization(custom_method),
    })
    updated_solve_options2.update({
        'optimization': Optimization(scipy_method, method_options, compute_gradient=False),
        'finite_differences': True,
    })
    results1 = problem.solve(**updated_solve_options1)
    results2 = problem.solve(**updated_solve_options2)

    # test that all arrays close except for those created with finite differences after the fact
    for key, result1 in results1.__dict__.items():
        if isinstance(result1, np.ndarray) and result1.dtype != np.object_:
            if not any(s in key for s in ['gradient', '_jacobian', '_se', '_covariances']):
                np.testing.assert_allclose(result1, getattr(results2, key), atol=1e-14, rtol=0, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('method_info', [
    pytest.param(('l-bfgs-b', 'gtol'), id="L-BFGS-B"),
    pytest.param(('trust-constr', 'gtol'), id="trust-region"),
    pytest.param(('tnc', 'gtol'), id="TNC"),
    pytest.param(('slsqp', 'ftol'), id="SLSQP"),
    pytest.param(('knitro', 'ftol'), id="Knitro"),
    pytest.param(('cg', 'gtol'), id="CG"),
    pytest.param(('bfgs', 'gtol'), id="BFGS"),
    pytest.param(('newton-cg', 'xtol'), id="Newton-CG"),
    pytest.param(('nelder-mead', 'xatol'), id="Nelder-Mead"),
    pytest.param(('powell', 'xtol'), id="Powell")
])
def test_bounds(simulated_problem: SimulatedProblemFixture, method_info: Tuple[str, str]) -> None:
    """Test that non-binding bounds on parameters in simulated problems do not affect estimates and that binding bounds
    are respected. Forcing parameters to be far from their optimal values creates instability problems, so this is also
    a test of how well estimation handles unstable problems.
    """
    simulation, _, problem, solve_options, _ = simulated_problem
    method, tol_option = method_info

    # skip optimization methods that haven't been configured properly, using a loose tolerance to speed up the test
    updated_solve_options = copy.deepcopy(solve_options)
    try:
        updated_solve_options['optimization'] = Optimization(
            method,
            method_options={tol_option: 1e-8 if method == 'trust-constr' else 0.1},
            compute_gradient=method not in {'nelder-mead', 'powell'}
        )
    except OSError as exception:
        return pytest.skip(f"Failed to use the {method} method in this environment: {exception}.")

    # solve the problem when unbounded
    unbounded_solve_options = copy.deepcopy(updated_solve_options)
    unbounded_solve_options.update({
        'sigma_bounds': (np.full_like(simulation.sigma, -np.inf), np.full_like(simulation.sigma, +np.inf)),
        'pi_bounds': (np.full_like(simulation.pi, -np.inf), np.full_like(simulation.pi, +np.inf)),
        'rho_bounds': (np.full_like(simulation.rho, -np.inf), np.full_like(simulation.rho, +np.inf)),
        'beta_bounds': (np.full_like(simulation.beta, -np.inf), np.full_like(simulation.beta, +np.inf)),
        'gamma_bounds': (np.full_like(simulation.gamma, -np.inf), np.full_like(simulation.gamma, +np.inf))
    })
    unbounded_results = problem.solve(**unbounded_solve_options)

    # choose a parameter from each set and identify its estimated value
    sigma_index = pi_index = rho_index = beta_index = gamma_index = None
    sigma_value = pi_value = rho_value = beta_value = gamma_value = None
    if problem.K2 > 0:
        sigma_index = (simulation.sigma.nonzero()[0][0], simulation.sigma.nonzero()[1][0])
        sigma_value = unbounded_results.sigma[sigma_index]
    if problem.D > 0:
        pi_index = (simulation.pi.nonzero()[0][0], simulation.pi.nonzero()[1][0])
        pi_value = unbounded_results.pi[pi_index]
    if problem.H > 0:
        rho_index = (simulation.rho.nonzero()[0][0], simulation.rho.nonzero()[1][0])
        rho_value = unbounded_results.rho[rho_index]
    if problem.K1 > 0:
        beta_index = (simulation.beta.nonzero()[0][-1], simulation.beta.nonzero()[1][-1])
        beta_value = unbounded_results.beta[beta_index]
    if problem.K3 > 0:
        gamma_index = (simulation.gamma.nonzero()[0][-1], simulation.gamma.nonzero()[1][-1])
        gamma_value = unbounded_results.gamma[gamma_index]

    # only test non-fixed (but bounded) parameters for routines that support this
    bound_scales: List[Tuple[Any, Any]] = [(0, 0)]
    if method not in {'cg', 'bfgs', 'newton-cg', 'nelder-mead', 'powell'}:
        bound_scales.extend([(-0.1, +np.inf), (+1, -0.1)])

    # use different types of binding bounds
    for lb_scale, ub_scale in bound_scales:
        binding_sigma_bounds = (np.full_like(simulation.sigma, -np.inf), np.full_like(simulation.sigma, +np.inf))
        binding_pi_bounds = (np.full_like(simulation.pi, -np.inf), np.full_like(simulation.pi, +np.inf))
        binding_rho_bounds = (np.full_like(simulation.rho, -np.inf), np.full_like(simulation.rho, +np.inf))
        binding_beta_bounds = (np.full_like(simulation.beta, -np.inf), np.full_like(simulation.beta, +np.inf))
        binding_gamma_bounds = (np.full_like(simulation.gamma, -np.inf), np.full_like(simulation.gamma, +np.inf))
        if problem.K2 > 0:
            binding_sigma_bounds[0][sigma_index] = sigma_value - lb_scale * np.abs(sigma_value)
            binding_sigma_bounds[1][sigma_index] = sigma_value + ub_scale * np.abs(sigma_value)
        if problem.D > 0:
            binding_pi_bounds[0][pi_index] = pi_value - lb_scale * np.abs(pi_value)
            binding_pi_bounds[1][pi_index] = pi_value + ub_scale * np.abs(pi_value)
        if problem.H > 0:
            binding_rho_bounds[0][rho_index] = rho_value - lb_scale * np.abs(rho_value)
            binding_rho_bounds[1][rho_index] = rho_value + ub_scale * np.abs(rho_value)
        if problem.K1 > 0:
            binding_beta_bounds[0][beta_index] = beta_value - lb_scale * np.abs(beta_value)
            binding_beta_bounds[1][beta_index] = beta_value + ub_scale * np.abs(beta_value)
        if problem.K3 > 0:
            binding_gamma_bounds[0][gamma_index] = gamma_value - lb_scale * np.abs(gamma_value)
            binding_gamma_bounds[1][gamma_index] = gamma_value + ub_scale * np.abs(gamma_value)

        # update options with the binding bounds
        binding_solve_options = copy.deepcopy(updated_solve_options)
        binding_solve_options.update({
            'sigma': np.clip(binding_solve_options['sigma'], *binding_sigma_bounds),
            'pi': np.clip(binding_solve_options['pi'], *binding_pi_bounds),
            'rho': np.clip(binding_solve_options['rho'], *binding_rho_bounds),
            'sigma_bounds': binding_sigma_bounds,
            'pi_bounds': binding_pi_bounds,
            'rho_bounds': binding_rho_bounds,
            'beta_bounds': binding_beta_bounds,
            'gamma_bounds': binding_gamma_bounds
        })
        if problem.K1 > 0:
            binding_solve_options['beta'] = binding_solve_options.get('beta', np.full_like(simulation.beta, np.nan))
            binding_solve_options['beta'][beta_index] = beta_value
            with np.errstate(invalid='ignore'):
                binding_solve_options['beta'] = np.clip(binding_solve_options['beta'], *binding_beta_bounds)
        if problem.K3 > 0:
            binding_solve_options['gamma'] = binding_solve_options.get('gamma', np.full_like(simulation.gamma, np.nan))
            binding_solve_options['gamma'][gamma_index] = gamma_value
            with np.errstate(invalid='ignore'):
                binding_solve_options['gamma'] = np.clip(binding_solve_options['gamma'], *binding_gamma_bounds)

        # solve the problem and test that the bounds are respected (ignore a warning about minimal gradient changes)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            binding_results = problem.solve(**binding_solve_options)
        assert_array_less = lambda a, b: np.testing.assert_array_less(a, b + 1e-14, verbose=True)
        if problem.K2 > 0:
            assert_array_less(binding_sigma_bounds[0], binding_results.sigma)
            assert_array_less(binding_results.sigma, binding_sigma_bounds[1])
        if problem.D > 0:
            assert_array_less(binding_pi_bounds[0], binding_results.pi)
            assert_array_less(binding_results.pi, binding_pi_bounds[1])
        if problem.H > 0:
            assert_array_less(binding_rho_bounds[0], binding_results.rho)
            assert_array_less(binding_results.rho, binding_rho_bounds[1])
        if problem.K1 > 0:
            assert_array_less(binding_beta_bounds[0], binding_results.beta)
            assert_array_less(binding_results.beta, binding_beta_bounds[1])
        if problem.K3 > 0:
            assert_array_less(binding_gamma_bounds[0], binding_results.gamma)
            assert_array_less(binding_results.gamma, binding_gamma_bounds[1])


@pytest.mark.usefixtures('simulated_problem')
def test_extra_nodes(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that agents in a simulated problem are identical to agents in a problem created with agent data built
    according to the same integration specification but containing unnecessary columns of nodes.
    """
    simulation, simulation_results, problem, _, _ = simulated_problem

    # skip simulations without agents
    if simulation.K2 == 0:
        return pytest.skip("There are no nodes.")

    # reconstruct the problem with unnecessary columns of nodes
    assert simulation.agent_data is not None
    product_data = simulation_results.product_data
    extra_agent_data = {k: simulation.agent_data[k] for k in simulation.agent_data.dtype.names}
    extra_agent_data['nodes'] = np.c_[extra_agent_data['nodes'], extra_agent_data['nodes']]
    new_problem = Problem(problem.product_formulations, product_data, problem.agent_formulation, extra_agent_data)

    # test that the agents are essentially identical
    for key in problem.agents.dtype.names:
        if problem.agents[key].dtype != np.object_:
            np.testing.assert_allclose(problem.agents[key], new_problem.agents[key], atol=1e-14, rtol=0, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
def test_extra_demographics(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that agents in a simulated problem are identical to agents in a problem created with agent data built
    according to the same integration specification and but containing unnecessary rows of demographics.
    """
    simulation, simulation_results, problem, _, _ = simulated_problem

    # skip simulations without demographics
    if simulation.D == 0:
        return pytest.skip("There are no demographics.")

    # also skip those with custom agents
    assert simulation.agent_data is not None
    agent_data = simulation.agent_data
    if 'weights' in agent_data.dtype.names:
        return pytest.skip("There are custom agents.")

    # reconstruct the problem with unnecessary rows of demographics
    product_data = simulation_results.product_data
    extra_agent_data = {k: np.r_[agent_data[k], agent_data[k]] for k in agent_data.dtype.names}
    new_problem = Problem(
        problem.product_formulations, product_data, problem.agent_formulation, extra_agent_data, simulation.integration
    )

    # test that the agents are essentially identical
    for key in problem.agents.dtype.names:
        if problem.agents[key].dtype != np.object_:
            np.testing.assert_allclose(problem.agents[key], new_problem.agents[key], atol=1e-14, rtol=0, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
def test_logit_errors(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that first and second choice probabilities are correct by comparing frequencies of second choices when
    simulated according to analytic expression that integrate over logit errors when simulating logit errors directly.
    For nested logit errors, call the R package evd by Alec Stephenson.
    """
    simulation, simulation_results, problem, _, _ = simulated_problem

    # skip problems with supply since this test will be the same
    if problem.K3 > 0:
        return pytest.skip("Configurations with supply give the same test.")

    # skip problems with differing product availability
    if problem.agents.availability.size > 0:
        return pytest.skip("This test is not supported for differing product availability.")

    # select only a few products (if there's nesting, select two from each of two nests)
    J = 4
    small_product_data = simulation_results.product_data[:J].copy()
    small_product_data.shares[:] = np.c_[[0.1, 0.2, 0.3, 0.15]]
    small_product_data.market_ids[:] = simulation.unique_market_ids[0]
    if small_product_data.ownership.size > 0:
        small_product_data = update_matrices(small_product_data, {
            'ownership': (small_product_data.ownership[:, :J], small_product_data.ownership.dtype)
        })
    if simulation.H > 0:
        small_product_data.nesting_ids[:2] = simulation.unique_nesting_ids[0]
        small_product_data.nesting_ids[2:] = simulation.unique_nesting_ids[1]

    # select just the first consumer type
    small_agent_data = None
    if simulation.agent_data is not None:
        small_agent_data = simulation.agent_data[[0]].copy()
        small_agent_data.weights[:] = 1

    # create a simulation with only a few products and one consumer type to reduce the number of needed simulation draws
    small_product_data = update_matrices(small_product_data, {'extra': (np.zeros(J), np.float64)})
    assert simulation.product_formulations[0] is not None
    product_formulations = (
        Formulation(f'{simulation.product_formulations[0]._formula} + extra'),
        simulation.product_formulations[1],
    )
    beta = np.r_[simulation.beta.flatten(), 1]
    small_simulation = Simulation(
        product_formulations, small_product_data, beta, simulation.sigma, simulation.pi, rho=simulation.rho[:2],
        agent_formulation=simulation.agent_formulation, agent_data=small_agent_data, xi=simulation.xi[:J],
        rc_types=simulation.rc_types, epsilon_scale=simulation.epsilon_scale
    )
    small_results = small_simulation.replace_exogenous('extra')

    # simulate according to analytic second choice probabilities, integrating over idiosyncratic preferences
    observations = 1_000_000
    dataset = MicroDataset(
        name="Second Choices",
        observations=observations,
        compute_weights=lambda _, p, a: np.ones((a.size, 1 + p.size, 1 + p.size)),
    )
    micro_data = small_results.simulate_micro_data(dataset, seed=0)

    # compute random coefficients
    coefficients = small_simulation.sigma @ small_simulation.agents.nodes.T
    if small_simulation.D > 0:
        coefficients = coefficients + small_simulation.pi @ small_simulation.agents.demographics.T

    for k, rc_type in enumerate(small_simulation.rc_types):
        if rc_type == 'log':
            if len(coefficients.shape) == 2:
                coefficients[k] = np.exp(coefficients[k])
            else:
                coefficients[:, k] = np.exp(coefficients[:, k])
        elif rc_type == 'logit':
            if len(coefficients.shape) == 2:
                coefficients[k] = scipy.special.expit(coefficients[k])
            else:
                coefficients[:, k] = scipy.special.expit(coefficients[:, k])

    # compute mu
    if len(coefficients.shape) == 2:
        mu = small_simulation.products.X2 @ coefficients
    else:
        mu = (small_simulation.products.X2[..., None] * coefficients).sum(axis=1)

    # simulate logit shocks, either in Python for simple logit or by calling R for nested logit
    epsilon: Array
    if simulation.H == 0:
        epsilon = simulation.epsilon_scale * np.random.default_rng(0).gumbel(size=(observations, 1 + J))
    else:
        if shutil.which('Rscript') is None:
            return pytest.skip("Failed to find an R executable in this environment.")

        # build asymmetry and dependence configurations for simulating nested logit shocks with the evd package
        asymmetry = []
        dependence = []
        for size in range(1, 2 + J):
            for combination in itertools.combinations(range(1 + J), size):
                asymmetry.append(len(combination) * [int(combination in {(0,), (1, 2), (3, 4)})])
                if size > 1:
                    if combination == (1, 2):
                        dependence.append(1 - float(simulation.rho if simulation.rho.size == 1 else simulation.rho[0]))
                    elif combination == (3, 4):
                        dependence.append(1 - float(simulation.rho if simulation.rho.size == 1 else simulation.rho[1]))
                    else:
                        dependence.append(1)

        # simulate the shocks with R, save them to a temporary file, and load them into memory
        epsilon = None
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            asy = 'list(' + ', '.join('c(' + ', '.join(str(v) for v in a) + ')' for a in asymmetry) + ')'
            dep = 'c(' + ', '.join(str(v) for v in dependence) + ')'
            path = handle.name.replace('\\', '/')
            try:
                command = ';'.join([
                    "suppressWarnings(library('evd'))",
                    "set.seed(0)",
                    f"epsilon = rmvevd(n={observations}, dep={dep}, asy={asy}, model='alog', d={1 + J})",
                    f"write.table(epsilon, file='{path}', row.names=FALSE, col.names=FALSE)",
                ])
                subprocess.run(['Rscript', '-e', command])
                epsilon = np.loadtxt(handle.name)
            finally:
                try:
                    os.remove(handle.name)
                except OSError:
                    pass

    # compute deterministic choices given simulated idiosyncratic preferences
    utilities = np.r_[0, small_results.delta.flatten() + mu.flatten()] + epsilon
    choice_indices = np.argmax(utilities, axis=1)
    utilities[np.arange(observations), choice_indices] = -np.inf
    second_choice_indices = np.argmax(utilities, axis=1)

    # compute histograms
    histogram1 = np.zeros(1 + J)
    histogram2 = np.zeros(1 + J)
    second_histogram1 = np.zeros((1 + J, 1 + J))
    second_histogram2 = np.zeros((1 + J, 1 + J))
    np.add.at(histogram1, micro_data.choice_indices, 1 / observations)
    np.add.at(histogram2, choice_indices, 1 / observations)
    np.add.at(second_histogram1, (micro_data.choice_indices, micro_data.second_choice_indices), 1 / observations)
    np.add.at(second_histogram2, (choice_indices, second_choice_indices), 1 / observations)

    # compare histograms
    np.testing.assert_allclose(histogram2, histogram1, rtol=0.01, atol=0.001)
    np.testing.assert_allclose(second_histogram2, second_histogram1, rtol=0.01, atol=0.001)


@pytest.mark.usefixtures('simulated_problem')
def test_micro_values(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that true micro values are close to those computed from simulated micro data."""
    simulation, simulation_results, problem, solve_options, problem_results = simulated_problem

    # skip simulations without micro moments
    if not solve_options['micro_moments']:
        return pytest.skip("There are no micro moments.")

    # test each micro value separately
    for moment in solve_options['micro_moments']:
        part_values = []
        for part in moment.parts:
            # simulate micro data
            dataset = MicroDataset(
                name=part.dataset.name,
                observations=1_000_000,
                compute_weights=part.dataset.compute_weights,
                market_ids=part.dataset.market_ids,
                eliminated_product_ids_index=part.dataset.eliminated_product_ids_index,
            )
            micro_data = simulation_results.simulate_micro_data(dataset, seed=0)

            # collect market IDs
            if dataset.market_ids is None:
                market_ids = problem.unique_market_ids
            else:
                market_ids = np.asarray(list(dataset.market_ids))

            # collect values market-by-market
            values_list = []
            for t in market_ids:
                products = problem.products[problem.products.market_ids.flat == t]
                agents = problem.agents[problem.agents.market_ids.flat == t]
                values = part.compute_values(t, products, agents)

                data = micro_data[micro_data.market_ids.flat == t]
                indices = [data.agent_indices.flatten(), data.choice_indices.flatten()]
                if len(values.shape) == 3:
                    indices.append(data.second_choice_indices.flatten())

                values_list.append(values[tuple(indices)])

            part_values.append(np.concatenate(values_list).mean())

        # compute the micro values
        value = moment.compute_value(np.array(part_values))
        np.testing.assert_allclose(moment.value, value, atol=0, rtol=0.015, err_msg=moment)


@pytest.mark.usefixtures('simulated_problem')
def test_micro_scores(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that true micro scores without any unobserved heterogeneity are zero, and that micro scores with unobserved
    heterogeneity based on micro data are of the same order of magnitude as the truth.
    """
    simulation, simulation_results, problem, solve_options, problem_results = simulated_problem

    # skip simulations without micro moments
    if not solve_options['micro_moments']:
        return pytest.skip("There are no micro moments.")

    # skip simulations with supply because testing their demand-only version is sufficient
    if problem.K3 > 0:
        return pytest.skip("There is a supply side.")

    # test each results type and dataset separately
    results_mapping = {
        'simulation': simulation_results,
        'problem': problem_results,
    }
    datasets = {p.dataset for m in solve_options['micro_moments'] for p in m.parts}
    for (name, results), dataset in itertools.product(results_mapping.items(), datasets):
        # test that true micro score expectations without any unobserved heterogeneity are zero
        true_observed_scores = results.compute_agent_scores(dataset)
        true_observed_moments = []
        for p, true_observed_scores_p in enumerate(true_observed_scores):
            true_observed_moments.append(MicroMoment(
                name=f'{dataset.name}, parameter {p}',
                value=0,
                parts=MicroPart(
                    name=f'{dataset.name}, parameter {p}',
                    dataset=dataset,
                    compute_values=lambda t, _, __, v=true_observed_scores_p: np.nan_to_num(v[t]),
                ),
            ))

        true_observed_means = results.compute_micro_values(true_observed_moments)
        np.testing.assert_allclose(true_observed_means, 0, atol=1e-13, rtol=0, err_msg=f'{name}: {dataset}')

        # either use the simulation's integration configuration or choose one
        integration = simulation.integration
        if integration is None:
            integration = Integration('product', 2)

        # test that micro scores with unobserved heterogeneity based micro data are of the same order of magnitude as
        #   the truth
        true_scores = results.compute_agent_scores(dataset, integration=integration)
        true_moments = []
        for p, true_scores_p in enumerate(true_scores):
            true_moments.append(MicroMoment(
                name=f'{dataset.name}, parameter {p}',
                value=0,
                parts=MicroPart(
                    name=f'{dataset.name}, parameter {p}',
                    dataset=dataset,
                    compute_values=lambda t, _, __, v=true_scores_p: np.nan_to_num(v[t]),
                ),
            ))

        true_means = results.compute_micro_values(true_moments)

        dataset = MicroDataset(
            name=dataset.name,
            observations=2_000,
            compute_weights=dataset.compute_weights,
            market_ids=dataset.market_ids,
            eliminated_product_ids_index=dataset.eliminated_product_ids_index,
        )
        data = data_to_dict(results.simulate_micro_data(dataset, seed=0))

        # merge in demographics
        assert simulation.agent_data is not None
        for key in simulation.agent_data.dtype.names:
            if key not in data:
                data[key] = np.zeros((data['micro_ids'].size, *simulation.agent_data[key].shape[1:]))
                for s, agent_indices in simulation._agent_market_indices.items():
                    indices = data['market_ids'] == s
                    data[key][indices] = simulation.agent_data[key][agent_indices][data['agent_indices'][indices]]

        scores = results.compute_micro_scores(dataset, data, integration)
        means = np.array(scores).mean(axis=1)
        np.testing.assert_allclose(means, true_means, atol=1e-13, rtol=10, err_msg=f'{name}: {dataset}')


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('eliminate', [
    pytest.param('all', id="all possible linear parameters eliminated"),
    pytest.param('some', id="some linear parameters eliminated"),
    pytest.param('no', id="no linear parameters eliminated")
])
@pytest.mark.parametrize('fe', [
    pytest.param(False, id="no FE"),
    pytest.param(True, id="FE"),
])
@pytest.mark.parametrize('demand', [
    pytest.param(True, id="demand"),
    pytest.param(False, id="no demand")
])
@pytest.mark.parametrize('supply', [
    pytest.param(True, id="supply"),
    pytest.param(False, id="no supply")
])
@pytest.mark.parametrize('covariance', [
    pytest.param(True, id="covariance"),
    pytest.param(False, id="no covariance")
])
@pytest.mark.parametrize('micro', [
    pytest.param(True, id="micro"),
    pytest.param(False, id="no micro")
])
def test_objective_gradient(
        simulated_problem: SimulatedProblemFixture, eliminate: str, fe: bool, demand: bool, supply: bool,
        covariance: bool, micro: bool) -> None:
    """Implement central finite differences in a custom optimization routine to test that analytic gradient values
    are close to estimated values.
    """
    simulation, simulation_results, problem, solve_options, problem_results = simulated_problem

    # skip some redundant tests
    if supply and problem.K3 == 0:
        return pytest.skip("The problem does not have supply-side moments to test.")
    if covariance and (problem.K3 == 0 or problem.MC == 0):
        return pytest.skip("The problem does not have covariance moments to test.")
    if micro and not solve_options['micro_moments']:
        return pytest.skip("The problem does not have micro moments to test.")
    if not demand and not supply and not covariance and not micro:
        return pytest.skip("There are no moments to test.")

    # configure the options used to solve the problem
    updated_solve_options = copy.deepcopy(solve_options)
    updated_solve_options.update({k: 0.9 * solve_options[k] for k in ['sigma', 'pi', 'rho', 'beta']})

    # optionally include linear parameters in theta
    if eliminate == 'no':
        if problem.K1 > 0:
            updated_solve_options['beta'] = 0.9 * simulation.beta
        if problem.K3 > 0:
            updated_solve_options['gamma'] = 0.9 * simulation.gamma
    elif eliminate == 'some':
        if problem.K1 > 0:
            updated_solve_options['beta'][-1] = 0.9 * simulation.beta[-1]
        if problem.K3 > 0:
            updated_solve_options['gamma'] = np.full_like(simulation.gamma, np.nan)
            updated_solve_options['gamma'][-1] = 0.9 * simulation.gamma[-1]
    else:
        assert eliminate == 'all'

    # optionally add a fixed effect
    if fe:
        # make product data mutable and add instruments
        product_data = {k: simulation_results.product_data[k] for k in simulation_results.product_data.dtype.names}
        product_data.update({
            'demand_instruments': problem.products.ZD[:, :-problem.K1],
            'supply_instruments': problem.products.ZS[:, :-problem.K3],
            'covariance_instruments': problem.products.ZC,
        })

        # remove constants and delete associated elements in the initial beta
        product_formulations = list(problem.product_formulations).copy()
        assert product_formulations[0] is not None
        constant_indices = [i for i, e in enumerate(product_formulations[0]._expressions) if not e.free_symbols]
        updated_solve_options['beta'] = np.delete(updated_solve_options['beta'], constant_indices, axis=0)

        # add fixed effect IDs to the data
        state = np.random.RandomState(seed=0)
        ids = state.choice(['a', 'b', 'c'], problem.N)
        product_data['demand_ids'] = ids

        # update the problem
        product_formulations[0] = Formulation(f'{product_formulations[0]._formula} - 1', absorb='demand_ids')
        problem = Problem(
            product_formulations, product_data, problem.agent_formulation, simulation.agent_data,
            rc_types=simulation.rc_types, epsilon_scale=simulation.epsilon_scale, costs_type=simulation.costs_type
        )

    # zero out weighting matrix blocks to only test individual contributions of the gradient
    updated_solve_options['W'] = np.eye(problem.MD + problem.MS + problem.MC + len(solve_options['micro_moments']))
    if not demand:
        updated_solve_options['W'][:problem.MD, :problem.MD] = 0
    if not supply and problem.K3 > 0:
        MDS = problem.MD + problem.MS
        updated_solve_options['W'][problem.MD:MDS, problem.MD:MDS] = 0
    if not covariance and problem.K3 > 0:
        MDS = problem.MD + problem.MS
        MDSC = MDS + problem.MC
        updated_solve_options['W'][MDS:MDSC, MDS:MDSC] = 0
    if not micro and updated_solve_options['micro_moments']:
        MM = len(updated_solve_options['micro_moments'])
        updated_solve_options['W'][-MM:, -MM:] = 0

    # use a restrictive iteration tolerance
    updated_solve_options['iteration'] = Iteration('squarem', {'atol': 1e-14})

    # compute the analytic gradient
    updated_solve_options['optimization'] = Optimization('return')
    exact = problem.solve(**updated_solve_options).gradient

    def test_finite_differences(theta: Array, _: Any, objective_function: Callable, __: Any) -> Tuple[Array, bool]:
        """Test central finite differences around starting parameter values."""
        approximated = compute_finite_differences(lambda x: objective_function(x)[0], theta, epsilon_scale=10.0)
        np.testing.assert_allclose(approximated.flatten(), exact.flatten(), atol=1e-8, rtol=1e-2)
        return theta, True

    # test the gradient
    updated_solve_options['optimization'] = Optimization(test_finite_differences, compute_gradient=False)
    problem.solve(**updated_solve_options)


@pytest.mark.usefixtures('simulated_problem')
def test_sigma_squared_se(simulated_problem: SimulatedProblemFixture) -> None:
    """Test that standard errors for sigma * sigma' computed with the delta method match a simple expression for when
    sigma is diagonal.
    """
    _, _, problem, _, results = simulated_problem

    # skip some unneeded tests
    if problem.K2 == 0:
        return pytest.skip("There's nothing to test without random coefficients.")
    if (np.tril(results.sigma, k=-1) != 0).any():
        return pytest.skip("There isn't a simple expression for when sigma isn't diagonal.")

    # compute standard errors with the simple expression and compare
    sigma_squared_se = np.nan_to_num(2 * results.sigma.diagonal() * results.sigma_se.diagonal())
    np.testing.assert_allclose(sigma_squared_se, results.sigma_squared_se.diagonal(), atol=1e-14, rtol=0)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('method', [
    pytest.param('1s', id="one-step"),
    pytest.param('2s', id="two-step")
])
@pytest.mark.parametrize('center_moments', [pytest.param(True, id="centered"), pytest.param(False, id="uncentered")])
@pytest.mark.parametrize('W_type', [
    pytest.param('robust', id="robust W"),
    pytest.param('unadjusted', id="unadjusted W"),
    pytest.param('clustered', id="clustered W")
])
@pytest.mark.parametrize('se_type', [
    pytest.param('robust', id="robust SEs"),
    pytest.param('unadjusted', id="unadjusted SEs"),
    pytest.param('clustered', id="clustered SEs")
])
def test_logit(
        simulated_problem: SimulatedProblemFixture, method: str, center_moments: bool, W_type: str, se_type: str) -> (
        None):
    """Test that Logit estimates are the same as those from the linearmodels package."""
    _, simulation_results, problem, _, _ = simulated_problem

    # skip more complicated simulations
    if problem.K2 > 0 or problem.K3 > 0 or problem.H > 0 or problem.epsilon_scale != 1:
        return pytest.skip("This simulation cannot be tested against linearmodels.")

    # solve the problem
    results1 = problem.solve(method=method, center_moments=center_moments, W_type=W_type, se_type=se_type)

    # compute the delta from the logit problem
    delta = np.log(simulation_results.product_data.shares)
    for t in problem.unique_market_ids:
        shares_t = simulation_results.product_data.shares[simulation_results.product_data.market_ids == t]
        delta[simulation_results.product_data.market_ids == t] -= np.log(1 - shares_t.sum())

    # configure covariance options
    W_options = {'clusters': simulation_results.product_data.clustering_ids} if W_type == 'clustered' else {}
    se_options = {'clusters': simulation_results.product_data.clustering_ids} if se_type == 'clustered' else {}

    # solve the problem with linearmodels, monkey-patching a problematic linearmodels method that shouldn't be called
    #   but is anyways
    import linearmodels
    linearmodels.iv.model._IVLSModelBase._estimate_kappa = lambda _: 1
    model = linearmodels.IVGMM(
        delta, exog=None, endog=problem.products.X1, instruments=problem.products.ZD, center=center_moments,
        weight_type=W_type, **W_options
    )
    results2 = model.fit(iter_limit=1 if method == '1s' else 2, cov_type=se_type, **se_options)

    # test that results are essentially identical (unadjusted second stage standard errors will be different because
    #   linearmodels still constructs a S matrix)
    for key1, key2 in [('beta', 'params'), ('xi', 'resids'), ('beta_se', 'std_errors')]:
        if not (se_type == 'unadjusted' and key1 == 'beta_se'):
            values1 = getattr(results1, key1)
            values2 = np.c_[getattr(results2, key2)]
            np.testing.assert_allclose(values1, values2, atol=1e-10, rtol=1e-8, err_msg=key1)

    # test that test statistics in the second stage (when they make sense) are essentially identical
    if method == '2s' and se_type != 'unadjusted':
        nonconstant = (problem.products.X1[0] != problem.products.X1).any(axis=0)
        F1 = results1.run_wald_test(results1.parameters[nonconstant], np.eye(results1.parameters.size)[nonconstant])
        J1 = results1.run_hansen_test()
        F2 = results2.f_statistic.stat
        J2 = results2.j_stat.stat
        np.testing.assert_allclose(F1, F2, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(J1, J2, atol=1e-10, rtol=1e-8)
