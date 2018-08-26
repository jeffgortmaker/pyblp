"""Primary tests."""

import pytest
import numpy as np
import linearmodels
import scipy.optimize

from pyblp import parallel, build_matrix, Problem, Iteration, Optimization, Formulation


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('solve_options_update', [
    pytest.param({'steps': 1}, id="one step"),
    pytest.param({'fp_type': 'nonlinear'}, id="nonlinear fixed point"),
    pytest.param({'delta_behavior': 'first'}, id="conservative starting delta values"),
    pytest.param({'error_behavior': 'punish', 'error_punishment': 1e5}, id="error punishment"),
    pytest.param({'center_moments': False, 'covariance_type': 'unadjusted'}, id="simple covariance matrices"),
    pytest.param({'covariance_type': 'clustered'}, id="clustered covariance matrices")
])
def test_accuracy(simulated_problem, solve_options_update):
    """Test that starting parameters that are half their true values give rise to errors of less than 10%. Use loose
    bounds to avoid overflow.
    """
    simulation, _, problem, solve_options, _ = simulated_problem

    # update the default options and solve the problem
    updated_solve_options = solve_options.copy()
    updated_solve_options.update({
        'sigma': 0.5 * simulation.sigma,
        'pi': 0.5 * simulation.pi,
        'rho': 0.5 * simulation.rho,
        **solve_options_update
    })
    results = problem.solve(**updated_solve_options)

    # test the accuracy of the estimated parameters
    keys = ['beta', 'sigma', 'pi', 'rho']
    if problem.K3 > 0:
        keys.append('gamma')
    for key in keys:
        np.testing.assert_allclose(getattr(simulation, key), getattr(results, key), atol=0, rtol=0.1, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('solve_options_update', [
    pytest.param({'costs_bounds': (-1e10, 1e10)}, id="non-binding costs bounds")
])
def test_trivial_changes(simulated_problem, solve_options_update):
    """Test that solving a problem with arguments that shouldn't give rise to meaningful differences doesn't give rise
    to any differences.
    """
    simulation, _, problem, solve_options, results = simulated_problem
    
    # solve the problem with the updated options
    updated_solve_options = solve_options.copy()
    updated_solve_options.update(solve_options_update)
    updated_results = problem.solve(**updated_solve_options)

    # test that all arrays in the results are essentially identical
    for key, result in results.__dict__.items():
        if isinstance(result, np.ndarray) and result.dtype != np.object:
            np.testing.assert_allclose(result, getattr(updated_results, key), atol=1e-14, rtol=0, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
def test_parallel(simulated_problem):
    """Test that solving simulations, solving problems, and computing results with parallelization gives rise to the
    same results as when using serial processing.
    """
    simulation, product_data, problem, solve_options, results = simulated_problem

    # compute marginal costs as a test of results (everything else has already been computed without parallelization)
    costs = results.compute_costs()

    # solve the simulation, solve the problem, and compute costs in parallel
    with parallel(2):
        parallel_product_data = simulation.solve()
        parallel_problem = Problem(
            problem.product_formulations, parallel_product_data, problem.agent_formulation, simulation.agent_data
        )
        parallel_results = parallel_problem.solve(**solve_options)
        parallel_costs = parallel_results.compute_costs()

    # test that product data are essentially identical
    for key in product_data.dtype.names:
        if product_data[key].dtype != np.object:
            np.testing.assert_allclose(product_data[key], parallel_product_data[key], atol=1e-14, rtol=0, err_msg=key)

    # test that all arrays in the results are essentially identical
    for key, result in results.__dict__.items():
        if isinstance(result, np.ndarray) and result.dtype != np.object:
            np.testing.assert_allclose(result, getattr(parallel_results, key), atol=1e-14, rtol=0, err_msg=key)

    # test that marginal costs are essentially equal
    np.testing.assert_allclose(costs, parallel_costs, atol=1e-14, rtol=0)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize(['ED', 'ES', 'absorb_method'], [
    pytest.param(1, 0, None, id="1 demand-side FE, default method"),
    pytest.param(0, 1, None, id="1 supply-side FE, default method"),
    pytest.param(1, 1, 'simple', id="1 demand- and 1 supply-side FE, simple de-meaning"),
    pytest.param(2, 0, None, id="2 demand-side FEs, default method"),
    pytest.param(2, 0, 'memory', id="2 demand-side FEs, memory"),
    pytest.param(2, 2, 'speed', id="2 demand- and 2 supply-side FEs, speed"),
    pytest.param(3, 0, None, id="3 demand-side FEs"),
    pytest.param(0, 3, None, id="3 supply-side FEs"),
    pytest.param(3, 3, None, id="3 demand- and 3 supply-side FEs, default method"),
    pytest.param(2, 1, None, id="2 demand- and 1 supply-side FEs, default method"),
    pytest.param(1, 2, Iteration('simple', {'tol': 1e-12}), id="1 demand- and 2 supply-side FEs, iteration")
])
def test_fixed_effects(simulated_problem, ED, ES, absorb_method):
    """Test that absorbing different numbers of demand- and supply-side fixed effects gives rise to essentially
    identical first-stage results as does including indicator variables. Also test that results that should be equal
    when there aren't any fixed effects are indeed equal, and that marginal costs are equal as well (this is a check
    for equality of post-estimation results).
    """
    simulation, product_data, problem, solve_options, results = simulated_problem

    # test that results that should be equal when there aren't any fixed effects are indeed equal
    for key in ['delta', 'tilde_costs', 'xi', 'omega', 'xi_jacobian', 'omega_jacobian']:
        result = getattr(results, key)
        true_result = getattr(results, f'true_{key}')
        np.testing.assert_allclose(result, true_result, atol=1e-14, rtol=0, err_msg=key)

    # there cannot be supply-side fixed effects if there isn't a supply side
    if problem.K3 == 0:
        ES = 0
    if ED == ES == 0:
        return

    # add fixed effect IDs to the data
    np.random.seed(0)
    demand_names = []
    supply_names = []
    product_data = {k: product_data[k] for k in product_data.dtype.names}
    for side, count, names in [('demand', ED, demand_names), ('supply', ES, supply_names)]:
        for index in range(count):
            name = f'{side}_ids{index}'
            ids = np.random.choice(['a', 'b', 'c'], product_data['market_ids'].size, [0.7, 0.2, 0.1])
            product_data[name] = ids
            names.append(name)

    # remove constants
    product_formulations = list(problem.product_formulations).copy()
    if ED > 0:
        product_formulations[0] = Formulation(f'{product_formulations[0]._formula} - 1')
        product_data['demand_instruments'] = product_data['demand_instruments'][:, 1:]
    if ES > 0:
        product_formulations[2] = Formulation(f'{product_formulations[2]._formula} - 1')
        product_data['supply_instruments'] = product_data['supply_instruments'][:, 1:]

    # build formulas for the IDs
    demand_formula = ' + '.join(demand_names)
    supply_formula = ' + '.join(supply_names)

    # solve the first stage of a problem in which the fixed effects are absorbed
    product_formulations1 = product_formulations.copy()
    if ED > 0:
        product_formulations1[0] = Formulation(product_formulations[0]._formula, demand_formula, absorb_method)
    if ES > 0:
        product_formulations1[2] = Formulation(product_formulations[2]._formula, supply_formula, absorb_method)
    problem1 = Problem(product_formulations1, product_data, problem.agent_formulation, simulation.agent_data)
    results1 = problem1.solve(**solve_options)

    # solve the first stage of a problem in which fixed effects are included as indicator variables
    product_data2 = product_data.copy()
    product_formulations2 = product_formulations.copy()
    if ED > 0:
        demand_indicators2 = build_matrix(Formulation(demand_formula), product_data)
        product_data2['demand_instruments'] = np.c_[product_data['demand_instruments'], demand_indicators2]
        product_formulations2[0] = Formulation(f'{product_formulations[0]._formula} + {demand_formula}')
    if ES > 0:
        supply_indicators2 = build_matrix(Formulation(supply_formula), product_data)
        product_data2['supply_instruments'] = np.c_[product_data['supply_instruments'], supply_indicators2]
        product_formulations2[2] = Formulation(f'{product_formulations[2]._formula} + {supply_formula}')
    problem2 = Problem(product_formulations2, product_data2, problem.agent_formulation, simulation.agent_data)
    results2 = problem2.solve(**solve_options)

    # solve the first stage of a problem in which some fixed effects are absorbed and some are included as indicators
    results3 = results2
    if ED > 1 or ES > 1:
        product_data3 = product_data.copy()
        product_formulations3 = product_formulations.copy()
        if ED > 0:
            demand_indicators3 = build_matrix(Formulation(demand_names[0]), product_data)[:, int(ED > 1):]
            product_data3['demand_instruments'] = np.c_[product_data['demand_instruments'], demand_indicators3]
            product_formulations3[0] = Formulation(
                f'{product_formulations[0]._formula} + {demand_names[0]}', ' + '.join(demand_names[1:]) or None
            )
        if ES > 0:
            supply_indicators3 = build_matrix(Formulation(supply_names[0]), product_data)[:, int(ES > 1):]
            product_data3['supply_instruments'] = np.c_[product_data['supply_instruments'], supply_indicators3]
            product_formulations3[2] = Formulation(
                f'{product_formulations[2]._formula} + {supply_names[0]}', ' + '.join(supply_names[1:]) or None
            )
        problem3 = Problem(product_formulations3, product_data3, problem.agent_formulation, simulation.agent_data)
        results3 = problem3.solve(**solve_options)

    # test that all arrays expected to be identical are identical
    keys = [
        'theta', 'sigma', 'pi', 'rho', 'beta', 'gamma', 'sigma_se', 'pi_se', 'rho_se', 'beta_se', 'gamma_se',
        'true_delta', 'true_tilde_costs', 'true_xi', 'true_omega', 'true_xi_jacobian', 'true_omega_jacobian',
        'objective', 'gradient', 'sigma_gradient', 'pi_gradient', 'rho_gradient'
    ]
    for key in keys:
        result1 = getattr(results1, key)
        result2 = getattr(results2, key)
        result3 = getattr(results3, key)
        if 'beta' in key or 'gamma' in key:
            result2 = result2[:result1.size]
            result3 = result3[:result1.size]
        np.testing.assert_allclose(result1, result2, atol=1e-8, rtol=1e-5, err_msg=key)
        np.testing.assert_allclose(result1, result3, atol=1e-8, rtol=1e-5, err_msg=key)

    # test that post-estimation results are identical (just check marginal costs, since they encompass a lot of
    #   post-estimation machinery)
    costs1 = results1.compute_costs()
    costs2 = results2.compute_costs()
    costs3 = results3.compute_costs()
    np.testing.assert_allclose(costs1, costs2, atol=1e-8, rtol=1e-5)
    np.testing.assert_allclose(costs1, costs3, atol=1e-8, rtol=1e-5)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('compute_prices_options', [
    pytest.param({}, id="defaults"),
    pytest.param({'iteration': Iteration('simple')}, id="configured iteration")
])
def test_merger(simulated_problem, compute_prices_options):
    """Test that prices and shares simulated under changed firm IDs are reasonably close to prices and shares computed
    from the results of a solved problem. In particular, test that unchanged prices and shares are farther from their
    simulated counterparts than those computed by approximating a merger, which in turn are farther from their simulated
    counterparts than those computed by fully solving a merger. Also test that simple acquisitions increase HHI. These
    inequalities are only guaranteed because of the way in which the simulations are configured.
    """
    simulation, _, _, _, results = simulated_problem

    # get changed prices and shares
    changed_product_data = simulation.solve(firms_index=1)

    # solve for approximate and actual changed prices and shares
    approximated_prices = results.compute_approximate_prices()
    estimated_prices = results.compute_prices(**compute_prices_options)
    approximated_shares = results.compute_shares(approximated_prices)
    estimated_shares = results.compute_shares(estimated_prices)

    # test that estimated prices are closer to changed prices than approximate prices
    approximated_prices_error = np.linalg.norm(changed_product_data.prices - approximated_prices)
    estimated_prices_error = np.linalg.norm(changed_product_data.prices - estimated_prices)
    np.testing.assert_array_less(estimated_prices_error, approximated_prices_error, verbose=True)

    # test that estimated shares are closer to changed shares than approximate shares
    approximated_shares_error = np.linalg.norm(changed_product_data.shares - approximated_shares)
    estimated_shares_error = np.linalg.norm(changed_product_data.shares - estimated_shares)
    np.testing.assert_array_less(estimated_shares_error, approximated_shares_error, verbose=True)

    # test that HHI increases
    hhi = results.compute_hhi()
    changed_hhi = results.compute_hhi(firms_index=1, shares=estimated_shares)
    np.testing.assert_array_less(hhi, changed_hhi, verbose=True)


@pytest.mark.usefixtures('simulated_problem')
def test_shares(simulated_problem):
    """Test that shares computed from estimated parameters are essentially equal to actual shares."""
    _, product_data, _, _, results = simulated_problem
    shares = results.compute_shares()
    np.testing.assert_allclose(product_data.shares, shares, atol=1e-14, rtol=0)


@pytest.mark.usefixtures('simulated_problem')
def test_shares_by_prices_jacobian(simulated_problem):
    """Use central finite differences to test that analytic values in the Jacobian of shares with respect to prices are
    essentially within 0.1% of estimated values.
    """
    simulation, product_data, _, _, results = simulated_problem

    # extract the Jacobian from the analytic expression for elasticities
    exact = np.nan_to_num(results.compute_elasticities())
    for t in simulation.unique_market_ids:
        prices_t = product_data.prices[product_data.market_ids.flat == t]
        shares_t = product_data.shares[product_data.market_ids.flat == t]
        exact[product_data.market_ids.flat == t] /= prices_t.T / shares_t

    # estimate the Jacobian with central finite differences
    estimated = np.zeros_like(exact)
    change = np.sqrt(np.finfo(np.float64).eps)
    for index in range(estimated.shape[1]):
        prices1 = product_data.prices.copy()
        prices2 = product_data.prices.copy()
        for t in simulation.unique_market_ids:
            if index < np.sum(product_data.market_ids.flat == t):
                prices1_t = prices1[product_data.market_ids.flat == t]
                prices2_t = prices2[product_data.market_ids.flat == t]
                prices1_t[index] += change / 2
                prices2_t[index] -= change / 2
                prices1[product_data.market_ids.flat == t] = prices1_t
                prices2[product_data.market_ids.flat == t] = prices2_t
        estimated[:, [index]] = (results.compute_shares(prices1) - results.compute_shares(prices2)) / change

    # compare the two sets of elasticity matrices
    np.testing.assert_allclose(exact, estimated, atol=1e-8, rtol=0)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('factor', [pytest.param(0.01, id="large"), pytest.param(0.0001, id="small")])
def test_elasticity_aggregates_and_means(simulated_problem, factor):
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

    # test the same inequality but for all non-price variables
    for name in {n for f in simulation._X1_formulations + simulation._X2_formulations for n in f.names} - {'prices'}:
        np.testing.assert_array_less(
            np.abs(results.compute_aggregate_elasticities(factor, name)),
            np.abs(results.extract_diagonal_means(results.compute_elasticities(name))),
            err_msg=name,
            verbose=True
        )


@pytest.mark.usefixtures('simulated_problem')
def test_diversion_ratios(simulated_problem):
    """Test simulated diversion ratio rows sum to one."""
    simulation, _, _, _, results = simulated_problem

    # test price-based ratios
    for compute in [results.compute_diversion_ratios, results.compute_long_run_diversion_ratios]:
        ratios = compute()
        np.testing.assert_allclose(ratios.sum(axis=1), 1, atol=1e-14, rtol=0, err_msg=compute.__name__)

    # test ratios based on other variables
    for name in {n for f in simulation._X1_formulations + simulation._X2_formulations for n in f.names} - {'prices'}:
        ratios = results.compute_diversion_ratios(name)
        np.testing.assert_allclose(ratios.sum(axis=1), 1, atol=1e-14, rtol=0, err_msg=name)


@pytest.mark.usefixtures('simulated_problem')
def test_result_positivity(simulated_problem):
    """Test that simulated markups, profits, consumer surpluses are positive, both before and after a merger."""
    _, _, _, _, results = simulated_problem

    # compute post-merger prices and shares
    changed_prices = results.compute_approximate_prices()
    changed_shares = results.compute_shares(changed_prices)

    # compute surpluses and test positivity
    np.testing.assert_array_less(0, results.compute_markups(), verbose=True)
    np.testing.assert_array_less(0, results.compute_profits(), verbose=True)
    np.testing.assert_array_less(0, results.compute_consumer_surpluses(), verbose=True)
    np.testing.assert_array_less(0, results.compute_markups(changed_prices), verbose=True)
    np.testing.assert_array_less(0, results.compute_profits(changed_prices, changed_shares), verbose=True)
    np.testing.assert_array_less(0, results.compute_consumer_surpluses(changed_prices), verbose=True)


@pytest.mark.usefixtures('simulated_problem')
def test_second_step(simulated_problem):
    """Test that results from two-step GMM on simulated data are identical to results from one-step GMM configured with
    results from a first step.
    """
    simulation, _, problem, solve_options, _ = simulated_problem

    # use two steps and remove sigma bounds so that it can't get stuck at zero
    updated_solve_options = solve_options.copy()
    updated_solve_options.update({
        'steps': 2,
        'sigma_bounds': (np.full_like(simulation.sigma, -np.inf), np.full_like(simulation.sigma, +np.inf))
    })

    # get two-step GMM results
    results12 = problem.solve(**updated_solve_options)
    assert results12.step == 2 and results12.last_results.step == 1 and results12.last_results.last_results is None

    # get results from the first step
    updated_solve_options1 = updated_solve_options.copy()
    updated_solve_options1['steps'] = 1
    results1 = problem.solve(**updated_solve_options1)

    # get results from the second step
    updated_solve_options2 = updated_solve_options1.copy()
    updated_solve_options2.update({
        'sigma': results1.sigma,
        'pi': results1.pi,
        'rho': results1.rho,
        'delta': results1.delta,
        'WD': results1.updated_WD,
        'WS': results1.updated_WS
    })
    results2 = problem.solve(**updated_solve_options2)
    assert results1.step == results2.step == 1 and results1.last_results is None and results2.last_results is None

    # test that results are essentially identical
    for key, result12 in results12.__dict__.items():
        if 'cumulative' not in key and isinstance(result12, np.ndarray) and result12.dtype != np.object:
            np.testing.assert_allclose(result12, getattr(results2, key), atol=1e-14, rtol=0, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('scipy_method', [
    pytest.param('l-bfgs-b', id="L-BFGS-B"),
    pytest.param('slsqp', id="SLSQP")
])
def test_gradient_optionality(simulated_problem, scipy_method):
    """Test that the option of not computing the gradient for simulated data does not affect estimates when the gradient
    isn't used.
    """
    simulation, _, problem, solve_options, _ = simulated_problem

    # skip simulations without gradients
    if simulation.K2 == simulation.H == 0:
        return

    # define a custom optimization method that doesn't use gradients
    def custom_method(initial, bounds, objective_function, _):
        wrapper = lambda x: objective_function(x)[0]
        results = scipy.optimize.minimize(wrapper, initial, method=scipy_method, bounds=bounds)
        return results.x, results.success

    # solve the problem when not using gradients and when not computing them
    updated_solve_options1 = solve_options.copy()
    updated_solve_options2 = solve_options.copy()
    updated_solve_options1['optimization'] = Optimization(custom_method)
    updated_solve_options2['optimization'] = Optimization(scipy_method, compute_gradient=False)
    results1 = problem.solve(**updated_solve_options1)
    results2 = problem.solve(**updated_solve_options2)

    # test that all arrays are essentially identical
    for key, result1 in results1.__dict__.items():
        if isinstance(result1, np.ndarray) and result1.dtype != np.object:
            np.testing.assert_allclose(result1, getattr(results2, key), atol=1e-14, rtol=0, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('method', [
    pytest.param('l-bfgs-b', id="L-BFGS-B"),
    pytest.param('tnc', id="TNC"),
    pytest.param('slsqp', id="SLSQP"),
    pytest.param('knitro', id="Knitro")
])
def test_bounds(simulated_problem, method):
    """Test that non-binding bounds on parameters in simulated problems do not affect estimates and that binding bounds
    are respected.
    """
    simulation, _, problem, solve_options, _ = simulated_problem

    # skip simulations without nonlinear parameters to bound
    if simulation.K2 == simulation.H == 0:
        return

    # skip optimization methods that haven't been configured properly
    updated_solve_options = solve_options.copy()
    try:
        updated_solve_options['optimization'] = Optimization(method)
    except OSError as exception:
        return pytest.skip(f"Failed to use the {method} method in this environment: {exception}.")

    # solve the problem when unbounded
    unbounded_solve_options = updated_solve_options.copy()
    unbounded_solve_options.update({
        'sigma_bounds': (np.full_like(simulation.sigma, -np.inf), np.full_like(simulation.sigma, +np.inf)),
        'pi_bounds': (np.full_like(simulation.pi, -np.inf), np.full_like(simulation.pi, +np.inf)),
        'rho_bounds': (np.full_like(simulation.rho, -np.inf), np.full_like(simulation.rho, +np.inf))
    })
    unbounded_results = problem.solve(**unbounded_solve_options)

    # choose an element in sigma and identify its estimated value
    sigma_index = sigma_value = None
    if simulation.K2 > 0:
        sigma_index = (simulation.sigma.nonzero()[0][0], simulation.sigma.nonzero()[1][0])
        sigma_value = unbounded_results.sigma[sigma_index]

    # do the same for pi
    pi_index = pi_value = None
    if simulation.D > 0:
        pi_index = (simulation.pi.nonzero()[0][0], simulation.pi.nonzero()[1][0])
        pi_value = unbounded_results.pi[pi_index]

    # do the same for rho
    rho_index = rho_value = None
    if simulation.H > 0:
        rho_index = (simulation.rho.nonzero()[0][0], simulation.rho.nonzero()[1][0])
        rho_value = unbounded_results.rho[rho_index]

    # use different types of binding bounds
    for lb_scale, ub_scale in [(-0.1, +np.inf), (+1, -0.1), (0, 0)]:
        binding_sigma_bounds = (np.full_like(simulation.sigma, -np.inf), np.full_like(simulation.sigma, +np.inf))
        binding_pi_bounds = (np.full_like(simulation.pi, -np.inf), np.full_like(simulation.pi, +np.inf))
        binding_rho_bounds = (np.full_like(simulation.rho, -np.inf), np.full_like(simulation.rho, +np.inf))
        if simulation.K2 > 0:
            binding_sigma_bounds[0][sigma_index] = sigma_value - lb_scale * np.abs(sigma_value)
            binding_sigma_bounds[1][sigma_index] = sigma_value + ub_scale * np.abs(sigma_value)
        if simulation.D > 0:
            binding_pi_bounds[0][pi_index] = pi_value - lb_scale * np.abs(pi_value)
            binding_pi_bounds[1][pi_index] = pi_value + ub_scale * np.abs(pi_value)
        if simulation.H > 0:
            binding_rho_bounds[0][rho_index] = rho_value - lb_scale * np.abs(rho_value)
            binding_rho_bounds[1][rho_index] = rho_value + ub_scale * np.abs(rho_value)

        # solve the problem with binding bounds and test that they are essentially respected
        binding_solve_options = updated_solve_options.copy()
        binding_solve_options.update({
            'sigma': np.clip(binding_solve_options['sigma'], *binding_sigma_bounds),
            'pi': np.clip(binding_solve_options['pi'], *binding_pi_bounds),
            'rho': np.clip(binding_solve_options['rho'], *binding_rho_bounds),
            'sigma_bounds': binding_sigma_bounds,
            'pi_bounds': binding_pi_bounds,
            'rho_bounds': binding_rho_bounds
        })
        binding_results = problem.solve(**binding_solve_options)
        assert_array_less = lambda a, b: np.testing.assert_array_less(a, b + 1e-14, verbose=True)
        if simulation.K2 > 0:
            assert_array_less(binding_sigma_bounds[0], binding_results.sigma)
            assert_array_less(binding_results.sigma, binding_sigma_bounds[1])
        if simulation.D > 0:
            assert_array_less(binding_pi_bounds[0], binding_results.pi)
            assert_array_less(binding_results.pi, binding_pi_bounds[1])
        if simulation.H > 0:
            assert_array_less(binding_rho_bounds[0], binding_results.rho)
            assert_array_less(binding_results.rho, binding_rho_bounds[1])


@pytest.mark.usefixtures('simulated_problem')
def test_extra_nodes(simulated_problem):
    """Test that agents in a simulated problem are identical to agents in a problem created with agent data built
    according to the same integration specification but containing unnecessary columns of nodes.
    """
    simulation, product_data, problem, _, _ = simulated_problem

    # skip simulations without agents
    if simulation.K2 == 0:
        return

    # reconstruct the problem with unnecessary columns of nodes
    extra_agent_data = {k: simulation.agent_data[k] for k in simulation.agent_data.dtype.names}
    extra_agent_data['nodes'] = np.c_[extra_agent_data['nodes'], extra_agent_data['nodes']]
    extra_problem = Problem(problem.product_formulations, product_data, problem.agent_formulation, extra_agent_data)

    # test that the agents are essentially identical
    for key in problem.agents.dtype.names:
        if problem.agents[key].dtype != np.object:
            np.testing.assert_allclose(problem.agents[key], extra_problem.agents[key], atol=1e-14, rtol=0, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
def test_extra_demographics(simulated_problem):
    """Test that agents in a simulated problem are identical to agents in a problem created with agent data built
    according to the same integration specification and but containing unnecessary rows of demographics.
    """
    simulation, product_data, problem, _, _ = simulated_problem

    # skip simulations without demographics
    if simulation.D == 0:
        return

    # reconstruct the problem with unnecessary rows of demographics
    agent_data = simulation.agent_data
    extra_agent_data = {k: np.r_[agent_data[k], agent_data[k]] for k in agent_data.dtype.names}
    extra_problem = Problem(
        problem.product_formulations, product_data, problem.agent_formulation, extra_agent_data, simulation.integration
    )

    # test that the agents are essentially identical
    for key in problem.agents.dtype.names:
        if problem.agents[key].dtype != np.object:
            np.testing.assert_allclose(problem.agents[key], extra_problem.agents[key], atol=1e-14, rtol=0, err_msg=key)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('solve_options_update', [
    pytest.param({}, id="default"),
    pytest.param({'fp_type': 'nonlinear'}, id="nonlinear fixed point")
])
def test_objective_gradient(simulated_problem, solve_options_update):
    """Implement central finite differences in a custom optimization routine to test that analytic gradient values
    are within 0.1% of estimated values.
    """
    simulation, _, problem, solve_options, _ = simulated_problem

    # skip simulations without gradients
    if simulation.K2 == simulation.H == 0:
        return

    # define a custom optimization routine that tests central finite differences around starting parameter values
    def test_finite_differences(*args):
        theta, _, objective_function, _ = args
        exact = objective_function(theta)[1]
        estimated = np.zeros_like(exact)
        change = np.sqrt(np.finfo(np.float64).eps)
        for index in range(theta.size):
            theta1 = theta.copy()
            theta2 = theta.copy()
            theta1[index] += change / 2
            theta2[index] -= change / 2
            estimated[index] = (objective_function(theta1)[0] - objective_function(theta2)[0]) / change
        np.testing.assert_allclose(exact, estimated, atol=0, rtol=0.001)
        return theta, True

    # test the gradient at parameter values slightly different from the true ones so that the objective is sizable
    updated_solve_options = solve_options.copy()
    updated_solve_options.update({
        'sigma': 0.9 * simulation.sigma,
        'pi': 0.9 * simulation.pi,
        'rho': 0.9 * simulation.rho,
        'steps': 1,
        'optimization': Optimization(test_finite_differences),
        'iteration': Iteration('squarem', {'tol': 1e-15 if solve_options.get('fp_type') == 'nonlinear' else 1e-14}),
        **solve_options_update
    })
    problem.solve(**updated_solve_options)


@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('steps', [pytest.param(1, id="one-step"), pytest.param(2, id="two-step")])
@pytest.mark.parametrize('covariance_type', [
    pytest.param('robust', id="robust SEs"),
    pytest.param('unadjusted', id="unadjusted SEs"),
    pytest.param('clustered', id="clustered SEs")
])
@pytest.mark.parametrize('center_moments', [pytest.param(True, id="centered"), pytest.param(False, id="uncentered")])
def test_logit(simulated_problem, steps, covariance_type, center_moments):
    """Test that Logit estimates are the same as those from the the linearmodels package."""
    _, product_data, problem, _, _ = simulated_problem

    # skip more complicated simulations
    if problem.K2 > 0 or problem.K3 > 0 or problem.H > 0:
        return

    # solve the problem
    results1 = problem.solve(steps=steps, covariance_type=covariance_type, center_moments=center_moments)

    # compute delta
    delta = np.log(product_data['shares'])
    for t in problem.unique_market_ids:
        shares_t = product_data['shares'][product_data['market_ids'] == t]
        delta[product_data['market_ids'] == t] -= np.log(1 - shares_t.sum())

    # configure covariance options
    covariance_options = {'clusters': product_data.clustering_ids} if covariance_type == 'clustered' else {}

    # monkey-patch a problematic linearmodels method that shouldn't be called but is anyways
    linearmodels.IVLIML._estimate_kappa = lambda _: 1

    # solve the problem with linearmodels
    model = linearmodels.IVGMM(
        delta, exog=None, endog=problem.products.X1, instruments=problem.products.ZD, weight_type=covariance_type,
        center=center_moments, **covariance_options
    )
    results2 = model.fit(iter_limit=steps, cov_type=covariance_type, **covariance_options)

    # test that results are essentially identical
    for key1, key2 in [('beta', 'params'), ('beta_se', 'std_errors'), ('xi', 'resids'), ('WD', 'weight_matrix')]:
        values1 = getattr(results1, key1)
        values2 = np.c_[getattr(results2, key2)]
        np.testing.assert_allclose(values1, values2, atol=1e-10, rtol=1e-6, err_msg=key1)
