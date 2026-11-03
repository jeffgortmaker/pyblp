"""Regression test comparing direct tangent JVPs with the legacy tensor product."""
import numpy as np
import pyblp
from pyblp.markets.market import Market
from pyblp.parameters import BetaParameter, GammaParameter, PhiParameter


def build_micro_problem():
    rng = np.random.default_rng(84121)
    T, J, I = 3, 12, 80
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    prices = rng.uniform(0.3, 2.5, N)
    x = rng.normal(size=N)
    X = np.c_[np.ones(N), prices, x]
    delta = X @ np.array([-2.8, -1.3, 0.35]) + 0.2 * rng.normal(size=N)
    sigma = np.diag([0.25, 0.4, 0.35])
    nodes = rng.normal(size=(T * I, 3))
    income = 0.4 * nodes[:, 1] + rng.normal(size=T * I)
    shares = np.empty(N)
    numerator = denominator = 0.0
    for t in range(T):
        products = slice(t * J, (t + 1) * J)
        agents = slice(t * I, (t + 1) * I)
        exponentials = np.exp(delta[products, None] + X[products] @ sigma @ nodes[agents].T)
        probabilities = exponentials / (1 + exponentials.sum(axis=0))
        shares[products] = probabilities.mean(axis=1)
        numerator += np.sum(probabilities * x[products, None] * income[agents]) / I
        denominator += np.sum(probabilities) / I

    products = {
        'market_ids': market_ids, 'prices': prices, 'x': x, 'shares': shares,
        'demand_instruments': np.c_[prices, prices**2, x**2, prices * x],
    }
    agents = {
        'market_ids': np.repeat(np.arange(T), I),
        'nodes': nodes, 'weights': np.full(T * I, 1 / I),
        'income': income,
    }
    problem = pyblp.Problem(
        [pyblp.Formulation('1 + prices + x'), pyblp.Formulation('1 + prices + x')],
        products, agent_data=agents, agent_formulation=pyblp.Formulation('0 + income'),
    )
    dataset = pyblp.MicroDataset(
        'inside purchases', 10000,
        lambda t, p, a: np.ones((a.size, p.size)),
    )
    part = pyblp.MicroPart(
        'income times x', dataset,
        lambda t, p, a: a.demographics[:, [0]] * p.X2[:, 2][None, :],
    )
    moment = pyblp.MicroMoment('income times chosen x', numerator / denominator, part)
    options = dict(
        sigma=sigma, pi=np.zeros((3, 1)), method='1s',
        optimization=pyblp.Optimization('return'),
        check_optimality='gradient', initial_update=False,
        W=np.eye(problem.MD + 1), micro_moments=[moment],
        error_behavior='raise',
    )
    return problem, options


def legacy_tensor_update(
    self, probabilities_tangent_mapping, conditionals_tangent_mapping,
    probabilities, conditionals, xi_jacobian,
):
    probabilities_tensor = conditionals_tensor = None
    for p, parameter in enumerate(self.parameters.unfixed):
        probabilities_tangent = probabilities_tangent_mapping[p]
        conditionals_tangent = conditionals_tangent_mapping[p]
        if isinstance(parameter, BetaParameter):
            probabilities_tangent[:] = 0
            if conditionals_tangent is not None:
                conditionals_tangent[:] = 0
            continue
        if isinstance(parameter, (GammaParameter, PhiParameter)):
            continue
        if probabilities_tensor is None:
            probabilities_tensor, conditionals_tensor = self.compute_probabilities_by_xi_tensor(
                probabilities, conditionals,
                compute_conditionals_tensor=any(
                    tangent is not None for tangent in conditionals_tangent_mapping.values()
                ),
            )
        probabilities_tangent += np.squeeze(
            np.moveaxis(probabilities_tensor, 0, 2) @ xi_jacobian[:, [p]], axis=2,
        )
        if conditionals_tangent is not None:
            assert conditionals_tensor is not None
            conditionals_tangent += np.squeeze(
                np.moveaxis(conditionals_tensor, 0, 2) @ xi_jacobian[:, [p]], axis=2,
            )


def test_micro_gradient_matches_tensor_reference(monkeypatch):
    problem, options = build_micro_problem()
    fast = problem.solve(**options)
    with monkeypatch.context() as patcher:
        patcher.setattr(Market, 'update_probabilities_by_parameter_tangent_mapping', legacy_tensor_update)
        reference = problem.solve(**options)

    for field in (
        'objective', 'gradient', 'delta', 'beta', 'micro_values',
        'micro_covariances', 'parameter_covariances',
    ):
        np.testing.assert_allclose(
            getattr(fast, field), getattr(reference, field),
            rtol=3e-8, atol=3e-9, equal_nan=True,
        )
