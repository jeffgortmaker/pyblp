"""Market underlying the BLP model."""

import functools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import exceptions, options
from ..configurations.iteration import ContractionResults, Iteration
from ..economies.economy import Economy
from ..moments import (
    EconomyMoments, MarketMoments, Moment, DemographicExpectationMoment, DemographicCovarianceMoment,
    DiversionProbabilityMoment, DiversionCovarianceMoment, CustomMoment
)
from ..parameters import BetaParameter, GammaParameter, NonlinearCoefficient, Parameter, Parameters, RhoParameter
from ..primitives import Container
from ..utilities.algebra import approximately_invert, approximately_solve
from ..utilities.basics import Array, Bounds, RecArray, Error, Groups, SolverStats, update_matrices


class Market(Container):
    """A market underlying the BLP model."""

    t: Any
    groups: Groups
    epsilon_scale: float
    costs_type: str
    J: int
    I: int
    K1: int
    K2: int
    K3: int
    D: int
    H: int
    sigma: Array
    pi: Array
    beta: Optional[Array]
    gamma: Optional[Array]
    rho_size: int
    group_rho: Array
    rho: Array
    delta: Optional[Array]
    mu: Array
    parameters: Parameters
    moments: Optional[MarketMoments]

    def __init__(
            self, economy: Economy, t: Any, parameters: Parameters, sigma: Array, pi: Array, rho: Array,
            beta: Optional[Array] = None, gamma: Optional[Array] = None, delta: Optional[Array] = None,
            moments: Optional[EconomyMoments] = None, data_override: Optional[Dict[str, Array]] = None,
            agents_override: Optional[RecArray] = None) -> None:
        """Store or compute information about formulations, data, parameters, and utility."""

        # structure relevant data
        self.t = t
        super().__init__(
            economy.products[economy._product_market_indices[t]],
            economy.agents[economy._agent_market_indices[t]] if agents_override is None else agents_override
        )

        # drop unneeded product data fields to save memory
        products_update_mapping = {}
        for key in ['market_ids', 'demand_ids', 'supply_ids', 'clustering_ids', 'X1', 'X3', 'ZD', 'ZS']:
            products_update_mapping[key] = (None, self.products[key].dtype)
        self.products = update_matrices(self.products, products_update_mapping)

        # drop unneeded agent data fields and fill missing columns of integration nodes (associated with zeros in sigma)
        #   with zeros
        agents_update_mapping = {'market_ids': (None, self.agents.market_ids.dtype)}
        if not parameters.nonzero_sigma_index.all():
            nodes = np.zeros((self.agents.shape[0], economy.K2), self.agents.nodes.dtype)
            nodes[:, parameters.nonzero_sigma_index] = self.agents.nodes[:, :parameters.nonzero_sigma_index.sum()]
            agents_update_mapping['nodes'] = (nodes, nodes.dtype)
        self.agents = update_matrices(self.agents, agents_update_mapping)

        # create nesting groups
        self.groups = Groups(self.products.nesting_ids)

        # store other configuration information
        self.epsilon_scale = economy.epsilon_scale
        self.costs_type = economy.costs_type

        # count dimensions
        self.J = self.products.shape[0]
        self.I = self.agents.shape[0]
        self.K1 = economy.K1
        self.K2 = economy.K2
        self.K3 = economy.K3
        self.D = economy.D
        self.H = self.groups.group_count

        # override any data
        if data_override is not None:
            for name, variable in data_override.items():
                self.products[name][:] = variable[economy._product_market_indices[t]]
            for index, formulation in enumerate(self._X2_formulations):
                if any(n in formulation.names for n in data_override):
                    self.products.X2[:, [index]] = formulation.evaluate(self.products)

        # store parameters (expand rho to all groups and all products)
        self.parameters = parameters
        self.sigma = sigma
        self.pi = pi
        self.beta = beta
        self.gamma = gamma
        self.rho_size = rho.size
        if self.rho_size == 1:
            self.group_rho = np.full((self.H, 1), float(rho))
            self.rho = np.full((self.J, 1), float(rho))
        else:
            self.group_rho = rho[np.searchsorted(economy.unique_nesting_ids, self.groups.unique)]
            self.rho = self.groups.expand(self.group_rho)

        # store delta and compute mu
        self.delta = None if delta is None else delta[economy._product_market_indices[t]]
        self.mu = self.compute_mu()

        # store moments relevant to this market
        self.moments = None if moments is None else MarketMoments(moments, t)

    def get_product(self, product_id: Any) -> int:
        """Get the product index associated with a product ID. This assumes that the market has already been validated
        to make sure that the product ID appears exactly once.
        """
        return int(np.argmax(self.products.product_ids == product_id))

    def get_membership_matrix(self) -> Array:
        """Build a membership matrix from nesting IDs."""
        tiled_ids = np.tile(self.products.nesting_ids, self.J)
        return np.where(tiled_ids == tiled_ids.T, 1, 0)

    def get_ownership_matrix(self, firm_ids: Optional[Array] = None, ownership: Optional[Array] = None) -> Array:
        """Get a pre-computed ownership matrix or build one. By default, use unchanged firm IDs."""
        if ownership is not None:
            return ownership[:, :self.J]
        if firm_ids is not None:
            tiled_ids = np.tile(firm_ids, self.J)
            return np.where(tiled_ids == tiled_ids.T, 1, 0)
        if self.products.ownership.shape[1] > 0:
            return self.products.ownership[:, :self.J]
        if self.products.firm_ids.size == 0:
            raise ValueError("Either firm IDs or an ownership matrix must have been specified.")
        tiled_ids = np.tile(self.products.firm_ids, self.J)
        return np.where(tiled_ids == tiled_ids.T, 1, 0)

    def compute_random_coefficients(self, sigma: Optional[Array] = None, pi: Optional[Array] = None) -> Array:
        """Compute all random coefficients. By default, use unchanged parameter values."""
        if sigma is None:
            sigma = self.sigma
        if pi is None:
            pi = self.pi

        coefficients = sigma @ self.agents.nodes.T
        if self.D > 0:
            coefficients += pi @ self.agents.demographics.T

        for k, distribution in enumerate(self.parameters.distributions):
            if distribution == 'lognormal':
                coefficients[k] = np.exp(coefficients[k])

        return coefficients

    def compute_single_random_coefficient(self, k: int) -> Array:
        """Compute a single random coefficient."""
        coefficient = self.sigma[[k], :] @ self.agents.nodes.T
        if self.D > 0:
            coefficient += self.pi[[k], :] @ self.agents.demographics.T

        if self.parameters.distributions[k] == 'lognormal':
            coefficient = np.exp(coefficient)

        return coefficient

    def compute_mu(
            self, X2: Optional[Array] = None, sigma: Optional[Array] = None, pi: Optional[Array] = None) -> Array:
        """Compute mu. By default, use unchanged X2 and parameters."""
        if X2 is None:
            X2 = self.products.X2
        return X2 @ self.compute_random_coefficients(sigma, pi)

    def update_delta_with_variable(self, name: str, variable: Array) -> Array:
        """Update delta to reflect a changed variable by adding any parameter-weighted characteristic changes to X1."""
        assert self.beta is not None and self.delta is not None

        # if the variable does not contribute to X1, delta remains unchanged
        if not any(name in f.names for f in self._X1_formulations):
            return self.delta

        # if the variable does contribute to X1, delta may change
        delta = self.delta.copy()
        override = {name: variable}
        for index, formulation in enumerate(self._X1_formulations):
            if name in formulation.names:
                change = formulation.evaluate(self.products, override) - formulation.evaluate(self.products)
                delta += self.beta[index] * change

        return delta

    def update_mu_with_variable(self, name: str, variable: Array) -> Array:
        """Update mu to reflect a changed variable by re-computing mu under the changed X2."""

        # if the variable does not contribute to X2, mu remains unchanged
        if not any(name in f.names for f in self._X2_formulations):
            return self.mu

        # if the variable does contribute to X2, mu may change
        X2 = self.products.X2.copy()
        override = {name: variable}
        for index, formulation in enumerate(self._X2_formulations):
            if name in formulation.names:
                X2[:, [index]] = formulation.evaluate(self.products, override)

        return self.compute_mu(X2)

    def update_costs_with_variable(self, costs: Array, name: str, variable: Array) -> Array:
        """Update marginal costs to reflect a changed variable by adding any parameter-weighted characteristic changes
        to X3, taking into account whether costs are linear or log-linear.
        """
        assert self.gamma is not None

        # if the variable does not contribute to X3, costs remain unchanged
        if not any(name in f.names for f in self._X3_formulations):
            return costs

        # if the variable does contribute to X3, costs may change
        costs = costs.copy()
        override = {name: variable}
        for index, formulation in enumerate(self._X3_formulations):
            if name in formulation.names:
                change = formulation.evaluate(self.products, override) - formulation.evaluate(self.products)
                if self.costs_type == 'linear':
                    costs += self.gamma[index] * change
                else:
                    assert self.costs_type == 'log'
                    costs *= np.exp(self.gamma[index] * change)

        return costs

    def compute_X1_derivatives(self, name: str, variable: Optional[Array] = None) -> Array:
        """Compute derivatives of X1 with respect to a variable. By default, use unchanged variable values."""
        override = None if variable is None else {name: variable}
        derivatives = np.zeros((self.J, self.K1), options.dtype)
        for index, formulation in enumerate(self._X1_formulations):
            if name in formulation.names:
                derivatives[:, [index]] = formulation.evaluate_derivative(name, self.products, override)

        return derivatives

    def compute_X2_derivatives(self, name: str, variable: Optional[Array] = None) -> Array:
        """Compute derivatives of X2 with respect to a variable. By default, use unchanged variable values."""
        override = None if variable is None else {name: variable}
        derivatives = np.zeros((self.J, self.K2), options.dtype)
        for index, formulation in enumerate(self._X2_formulations):
            if name in formulation.names:
                derivatives[:, [index]] = formulation.evaluate_derivative(name, self.products, override)

        return derivatives

    def compute_utility_derivatives(self, name: str, variable: Optional[Array] = None) -> Array:
        """Compute derivatives of utility with respect to a variable. By default, use unchanged variable values."""
        assert self.beta is not None
        derivatives = np.tile(self.compute_X1_derivatives(name, variable) @ np.nan_to_num(self.beta), self.I)

        if self.K2 > 0:
            derivatives += self.compute_X2_derivatives(name, variable) @ self.compute_random_coefficients()

        if self.epsilon_scale != 1:
            derivatives /= self.epsilon_scale

        return derivatives

    def compute_probabilities(
            self, delta: Array = None, mu: Optional[Array] = None, linear: bool = True, safe: bool = True,
            utility_reduction: Optional[Array] = None, numerator: Optional[Array] = None,
            eliminate_outside: bool = False, eliminate_product: Optional[int] = None) -> Tuple[Array, Optional[Array]]:
        """Compute choice probabilities. By default, use unchanged delta and mu values. If linear is False, delta and mu
        must be specified and already be exponentiated. If safe is True, scale the logit equation by the exponential of
        negative the maximum utility for each agent, and if utility_reduction is specified, it should be values that
        have already been subtracted from the specified utility for each agent. If the numerator is specified, it will
        be used as the numerator in the non-nested logit expression. If any products are eliminated, eliminate the
        outside option, an inside product, or both from the choice set.
        """
        if delta is None:
            assert self.delta is not None
            delta = self.delta
        if mu is None:
            mu = self.mu
        if self.K2 == 0:
            mu = int(not linear)

        # compute exponentiated utilities, optionally re-scaling the logit expression
        if not linear:
            assert self.epsilon_scale == 1
            exp_utilities = np.array(delta * mu)
            if self.H > 0:
                exp_utilities **= 1 / (1 - self.rho)
        else:
            utilities = delta + mu
            if self.H > 0:
                utilities /= 1 - self.rho

            if self.epsilon_scale != 1:
                assert self.H == 0
                utilities /= self.epsilon_scale

            if safe:
                utility_reduction = np.clip(utilities.max(axis=0, keepdims=True), 0, None)
                utilities -= utility_reduction

            exp_utilities = np.exp(utilities)

        # compute any components used to re-scale the logit expression
        scale = scale_weights = 1
        if utility_reduction is not None:
            if self.H == 0:
                scale = np.exp(-utility_reduction)
            else:
                scale = np.exp(-utility_reduction * (1 - self.group_rho))
                if self.rho_size > 1:
                    scale_weights = np.exp(-utility_reduction[None] * (self.group_rho.T - self.group_rho)[..., None])

        # optionally eliminate the outside option from the choice set
        if eliminate_outside:
            scale = 0

        # optionally eliminate a product from the choice set
        if eliminate_product is not None:
            exp_utilities[eliminate_product] = 0

        # compute standard probabilities
        if self.H == 0:
            if numerator is None:
                numerator = exp_utilities
            probabilities = numerator / (scale + exp_utilities.sum(axis=0, keepdims=True))
            return probabilities, None

        # compute nested probabilities
        exp_inclusives = self.groups.sum(exp_utilities)
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_weighted_inclusives = np.exp(np.log(exp_inclusives) * (1 - self.group_rho))
            conditionals = exp_utilities / self.groups.expand(exp_inclusives)
        exp_weighted_inclusives[~np.isfinite(exp_weighted_inclusives)] = 0
        conditionals[~np.isfinite(conditionals)] = 0
        marginals = exp_weighted_inclusives / (scale + (scale_weights * exp_weighted_inclusives[None]).sum(axis=1))
        probabilities = conditionals * self.groups.expand(marginals)

        return probabilities, conditionals

    def compute_eliminated_probabilities(
            self, probabilities: Array, delta: Optional[Array] = None, eliminate_outside: bool = False,
            eliminate_product: Optional[int] = None) -> Array:
        """Convert choice probabilities into probabilities with the outside option, an inside product, or both removed
        from the choice set. If there are any errors, revert to the full derivation of these eliminated probabilities,
        using the delta with which this market was initialized.
        """

        # compute the denominator of the expression
        if eliminate_outside and eliminate_product is not None:
            denominator = np.delete(probabilities, eliminate_product, axis=0).sum(axis=0, keepdims=True)
        elif eliminate_outside:
            denominator = probabilities.sum(axis=0, keepdims=True)
        elif eliminate_product is not None:
            denominator = 1 - probabilities[[eliminate_product]]
        else:
            return probabilities

        # try to compute the eliminated probabilities, ignoring any divisions by zero or underflow
        with np.errstate(all='ignore'):
            eliminated = probabilities / denominator

        # the probability of choosing an eliminated inside product is zero
        if eliminate_product is not None:
            eliminated[eliminate_product] = 0

        # if there were any errors, compute the eliminated probabilities directly
        if not np.isfinite(eliminated).all():
            eliminated, _ = self.compute_probabilities(
                delta, eliminate_outside=eliminate_outside, eliminate_product=eliminate_product
            )

        return eliminated

    def compute_delta(
            self, initial_delta: Array, iteration: Iteration, fp_type: str, shares_bounds: Bounds) -> (
            Tuple[Array, Array, SolverStats, List[Error]]):
        """Compute the mean utility for this market that equates market shares to observed values by solving a fixed
        point problem.
        """
        errors: List[Error] = []

        # default assumption is that no shares were clipped at the end of fixed point iteration
        clipped_shares = np.zeros((self.J, 1), np.bool)

        # if there is no heterogeneity, use the closed-form solution
        if self.K2 == 0:
            log_shares = np.log(self.products.shares)
            log_outside_share = np.log(1 - self.products.shares.sum())
            delta = log_shares - log_outside_share

            if self.H > 0:
                log_group_shares = np.log(self.groups.expand(self.groups.sum(self.products.shares)))
                delta -= self.rho * (log_shares - log_group_shares)

            if self.epsilon_scale != 1:
                assert self.H == 0
                delta *= self.epsilon_scale

            return delta, clipped_shares, SolverStats(), errors

        # solve for delta with a linear fixed point
        if 'linear' in fp_type:
            log_shares = np.log(self.products.shares)
            compute_probabilities = functools.partial(self.compute_probabilities, safe='safe' in fp_type)

            # define the function used to clip shares outside of potentially pre-specified bounds
            clip_shares = lambda _: None
            if np.isfinite(shares_bounds).all():
                def clip_shares(shares: Array) -> None:
                    """Clip shares from below and above."""
                    nonlocal clipped_shares
                    small_shares = shares < shares_bounds[0]
                    shares[small_shares] = shares_bounds[0]
                    large_shares = shares > shares_bounds[1]
                    shares[large_shares] = shares_bounds[1]
                    clipped_shares = small_shares | large_shares

            elif np.isfinite(shares_bounds[0]):
                def clip_shares(shares: Array) -> None:
                    """Clip shares from below."""
                    nonlocal clipped_shares
                    clipped_shares = shares < shares_bounds[0]
                    shares[clipped_shares] = shares_bounds[0]

            elif np.isfinite(shares_bounds[1]):
                def clip_shares(shares: Array) -> None:
                    """Clip shares from above."""
                    nonlocal clipped_shares
                    clipped_shares = shares > shares_bounds[1]
                    shares[clipped_shares] = shares_bounds[1]

            # define the linear contraction
            if self.H == 0:
                def contraction(x: Array) -> ContractionResults:
                    """Compute the next linear delta and optionally its Jacobian."""
                    probabilities = compute_probabilities(x)[0]
                    shares = probabilities @ self.agents.weights
                    clip_shares(shares)
                    x = x + log_shares - np.log(shares)
                    if not iteration._compute_jacobian:
                        return x, None, None
                    weighted_probabilities = self.agents.weights * probabilities.T
                    jacobian = (probabilities @ weighted_probabilities) / shares
                    return x, None, jacobian
            else:
                dampener = 1 - self.rho
                rho_membership = self.rho * self.get_membership_matrix()

                def contraction(x: Array) -> ContractionResults:
                    """Compute the next linear delta and optionally its Jacobian under nesting."""
                    probabilities, conditionals = compute_probabilities(x)
                    shares = probabilities @ self.agents.weights
                    clip_shares(shares)
                    x = x + (log_shares - np.log(shares)) * dampener
                    if not iteration._compute_jacobian:
                        return x, None, None
                    weighted_probabilities = self.agents.weights * probabilities.T
                    probabilities_part = dampener * (probabilities @ weighted_probabilities)
                    conditionals_part = rho_membership * (conditionals @ weighted_probabilities)
                    jacobian = (probabilities_part + conditionals_part) / shares
                    return x, None, jacobian

            # solve the linear fixed point problem
            delta, stats = iteration._iterate(initial_delta, contraction)
            return delta, clipped_shares, stats, errors

        # solve for delta with a nonlinear fixed point
        assert 'nonlinear' in fp_type and self.epsilon_scale == 1
        if 'safe' in fp_type:
            utility_reduction = np.clip(self.mu.max(axis=0, keepdims=True), 0, None)
            exp_mu = np.exp(self.mu - utility_reduction)
            compute_probabilities = functools.partial(
                self.compute_probabilities, mu=exp_mu, utility_reduction=utility_reduction, linear=False
            )
        else:
            exp_mu = np.exp(self.mu)
            compute_probabilities = functools.partial(self.compute_probabilities, mu=exp_mu, linear=False)

        # define the nonlinear contraction
        if self.H == 0:
            def contraction(x: Array) -> ContractionResults:
                """Compute the next exponentiated delta and optionally its Jacobian."""
                probability_ratios = compute_probabilities(x, numerator=exp_mu)[0]
                share_ratios = probability_ratios @ self.agents.weights
                x0, x = x, self.products.shares / share_ratios
                if not iteration._compute_jacobian:
                    return x, None, None
                shares = x0 * share_ratios
                probabilities = x0 * probability_ratios
                weighted_probabilities = self.agents.weights * probabilities.T
                jacobian = x / x0.T * (probabilities @ weighted_probabilities) / shares
                return x, None, jacobian
        else:
            dampener = 1 - self.rho
            rho_membership = self.rho * self.get_membership_matrix()

            def contraction(x: Array) -> ContractionResults:
                """Compute the next exponentiated delta and optionally its Jacobian under nesting."""
                probabilities, conditionals = compute_probabilities(x)
                shares = probabilities @ self.agents.weights
                x0, x = x, x * (self.products.shares / shares) ** dampener
                if not iteration._compute_jacobian:
                    return x, None, None
                weighted_probabilities = self.agents.weights * probabilities.T
                probabilities_part = dampener * (probabilities @ weighted_probabilities)
                conditionals_part = rho_membership * (conditionals @ weighted_probabilities)
                jacobian = x / x0.T * (probabilities_part + conditionals_part) / shares
                return x, None, jacobian

        # solve the nonlinear fixed point problem
        exp_delta, stats = iteration._iterate(np.exp(initial_delta), contraction)
        delta = np.log(exp_delta)
        return delta, clipped_shares, stats, errors

    def compute_capital_lamda(self, value_derivatives: Array) -> Array:
        """Compute the diagonal capital lambda matrix used to decompose markups."""
        diagonal = value_derivatives @ self.agents.weights

        if self.H > 0:
            diagonal /= 1 - self.rho

        return np.diagflat(diagonal)

    def compute_capital_gamma(
            self, value_derivatives: Array, probabilities: Array, conditionals: Optional[Array]) -> Array:
        """Compute the dense capital gamma matrix used to decompose markups."""
        weighted_value_derivatives = self.agents.weights * value_derivatives.T
        capital_gamma = probabilities @ weighted_value_derivatives

        if self.H > 0:
            membership = self.get_membership_matrix()
            capital_gamma += self.rho / (1 - self.rho) * membership * (conditionals @ weighted_value_derivatives)

        return capital_gamma

    def compute_eta(
            self, ownership_matrix: Optional[Array] = None, delta: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Compute the markup term in the BLP-markup equation. By default, get an unchanged ownership matrix and use
        the delta with which this market was initialized.
        """
        errors: List[Error] = []
        if ownership_matrix is None:
            ownership_matrix = self.get_ownership_matrix()
        if delta is None:
            assert self.delta is not None
            delta = self.delta

        utility_derivatives = self.compute_utility_derivatives('prices')
        probabilities, conditionals = self.compute_probabilities(delta)
        jacobian = self.compute_shares_by_variable_jacobian(utility_derivatives, probabilities, conditionals)
        capital_delta = -ownership_matrix * jacobian
        eta, replacement = approximately_solve(capital_delta, self.products.shares)
        if replacement:
            errors.append(exceptions.IntraFirmJacobianInversionError(capital_delta, replacement))

        return eta, errors

    def compute_zeta(
            self, costs: Array, ownership_matrix: Optional[Array] = None, utility_derivatives: Optional[Array] = None,
            prices: Optional[Array] = None) -> Tuple[Array, Array, Array]:
        """Compute the markup term in the zeta-markup equation. By default, get an unchanged ownership matrix, compute
        derivatives of utilities with respect to prices, and use unchanged prices. Also return and updated marginal
        costs (if they depend on shares), along with the intermediate diagonal of the capital lambda matrix, which is
        used for weighting during fixed point iteration. If prices are specified and X3 depends on shares, update the
        specified costs according to the shares that arise under these prices.
        """
        if ownership_matrix is None:
            ownership_matrix = self.get_ownership_matrix()
        if utility_derivatives is None:
            utility_derivatives = self.compute_utility_derivatives('prices')
        if prices is None:
            probabilities, conditionals = self.compute_probabilities()
            shares = self.products.shares
        else:
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
            probabilities, conditionals = self.compute_probabilities(delta, mu)
            shares = probabilities @ self.agents.weights
            costs = self.update_costs_with_variable(costs, 'shares', shares)

        value_derivatives = probabilities * utility_derivatives
        capital_lamda_diagonal = self.compute_capital_lamda(value_derivatives).diagonal()
        capital_lamda_inverse = np.diag(1 / capital_lamda_diagonal)
        capital_gamma = self.compute_capital_gamma(value_derivatives, probabilities, conditionals)
        zeta = (
            capital_lamda_inverse @ (ownership_matrix * capital_gamma).T @ (prices - costs) -
            capital_lamda_inverse @ shares
        )
        return zeta, costs, capital_lamda_diagonal

    def compute_equilibrium_prices(
            self, costs: Array, iteration: Iteration, prices: Optional[Array] = None,
            ownership_matrix: Optional[Array] = None) -> Tuple[Array, SolverStats]:
        """Compute equilibrium prices by iterating over the zeta-markup equation. By default, use unchanged firm IDs
        and use unchanged prices as initial values.
        """
        if ownership_matrix is None:
            ownership_matrix = self.get_ownership_matrix()
        if prices is None:
            prices = self.products.prices

        # derivatives of utilities with respect to prices change during iteration only if they depend on prices
        formulations = self._X1_formulations + self._X2_formulations
        if any(s.name == 'prices' for f in formulations for s in f.differentiate('prices').free_symbols):
            get_derivatives = lambda p: self.compute_utility_derivatives('prices', p)
        else:
            derivatives = self.compute_utility_derivatives('prices')
            get_derivatives = lambda _: derivatives

        def contraction(x: Array) -> ContractionResults:
            """Compute the next equilibrium prices."""
            zeta, updated_costs, capital_lamda_diagonal = self.compute_zeta(
                costs, ownership_matrix, get_derivatives(x), x
            )
            x = updated_costs + zeta
            return x, capital_lamda_diagonal, None

        # solve the fixed point problem
        prices, stats = iteration._iterate(prices, contraction)
        return prices, stats

    def compute_shares(self, prices: Optional[Array] = None) -> Array:
        """Compute shares evaluated at specific prices. By default, use unchanged prices."""
        if prices is None:
            prices = self.products.prices

        delta = self.update_delta_with_variable('prices', prices)
        mu = self.update_mu_with_variable('prices', prices)
        shares = self.compute_probabilities(delta, mu)[0] @ self.agents.weights
        return shares

    def compute_utility_derivatives_by_parameter_tangent(
            self, parameter: Parameter, X1_derivatives: Array, X2_derivatives: Array) -> Array:
        """Compute the tangent with respect to a parameter of derivatives of utility with respect to a variable."""
        tangent = np.zeros((self.J, self.I), options.dtype)
        if isinstance(parameter, BetaParameter):
            tangent += X1_derivatives[:, [parameter.location[0]]]
        elif isinstance(parameter, NonlinearCoefficient):
            v = parameter.get_agent_characteristic(self)
            if parameter.get_distribution(self) == 'lognormal':
                v *= self.compute_single_random_coefficient(parameter.location[0]).T
            tangent += X2_derivatives[:, [parameter.location[0]]] @ v.T
        else:
            assert isinstance(parameter, (GammaParameter, RhoParameter))

        if self.epsilon_scale != 1:
            tangent /= self.epsilon_scale

        return tangent

    def compute_probabilities_by_parameter_tangent(
            self, parameter: Parameter, probabilities: Array, conditionals: Optional[Array],
            delta: Optional[Array] = None, mu: Optional[Array] = None) -> Tuple[Array, Optional[Array]]:
        """Compute the tangent of probabilities with respect to a parameter. By default, use unchanged delta and mu."""
        if delta is None:
            assert self.delta is not None
            delta = self.delta
        if mu is None:
            mu = self.mu

        # without nesting, compute only the tangent of probabilities with respect to the parameter
        if self.H == 0:
            if isinstance(parameter, BetaParameter):
                x = parameter.get_product_characteristic(self)
                probabilities_tangent = probabilities * (x - x.T @ probabilities)
            elif isinstance(parameter, NonlinearCoefficient):
                x = parameter.get_product_characteristic(self)
                v = parameter.get_agent_characteristic(self)
                if parameter.get_distribution(self) == 'lognormal':
                    v *= self.compute_single_random_coefficient(parameter.location[0]).T
                probabilities_tangent = probabilities * v.T * (x - x.T @ probabilities)
            else:
                assert isinstance(parameter, GammaParameter)
                probabilities_tangent = np.zeros_like(probabilities)

            if self.epsilon_scale != 1:
                probabilities_tangent /= self.epsilon_scale

            return probabilities_tangent, None

        # marginal probabilities are needed to compute tangents with nesting
        marginals = self.groups.sum(probabilities)

        # compute the tangent of conditional and marginal probabilities with respect to the parameter
        if isinstance(parameter, BetaParameter):
            x = parameter.get_product_characteristic(self)

            # compute the tangent of conditional probabilities with respect to the parameter
            A = conditionals * x
            A_sums = self.groups.sum(A)
            conditionals_tangent = conditionals * (x - self.groups.expand(A_sums)) / (1 - self.rho)

            # compute the tangent of marginal probabilities with respect to the parameter
            B = marginals * A_sums
            marginals_tangent = B - marginals * B.sum(axis=0, keepdims=True)

        elif isinstance(parameter, NonlinearCoefficient):
            x = parameter.get_product_characteristic(self)
            v = parameter.get_agent_characteristic(self)
            if parameter.get_distribution(self) == 'lognormal':
                v *= self.compute_single_random_coefficient(parameter.location[0]).T

            # compute the tangent of conditional probabilities with respect to the parameter
            A = conditionals * x
            A_sums = self.groups.sum(A)
            conditionals_tangent = conditionals * v.T * (x - self.groups.expand(A_sums)) / (1 - self.rho)

            # compute the tangent of marginal probabilities with respect to the parameter
            B = marginals * A_sums * v.T
            marginals_tangent = B - marginals * B.sum(axis=0, keepdims=True)

        elif isinstance(parameter, RhoParameter):
            group_associations = parameter.get_group_associations(self.groups)
            associations = self.groups.expand(group_associations)

            # utilities are needed to compute tangents with respect to rho
            utilities = (delta + mu) / (1 - self.rho)

            # compute the tangent of conditional probabilities with respect to the parameter
            A = conditionals * utilities / (1 - self.rho)
            A_sums = self.groups.sum(A)
            conditionals_tangent = associations * (A - conditionals * self.groups.expand(A_sums))

            # compute the tangent of marginal probabilities with respect to the parameter (re-scale for robustness)
            utility_reduction = np.clip(utilities.max(axis=0, keepdims=True), 0, None)
            with np.errstate(divide='ignore', invalid='ignore'):
                B = marginals * (
                    A_sums * (1 - self.group_rho) -
                    (np.log(self.groups.sum(np.exp(utilities - utility_reduction))) + utility_reduction)
                )
                marginals_tangent = group_associations * B - marginals * (group_associations.T @ B)
            marginals_tangent[~np.isfinite(marginals_tangent)] = 0

        else:
            assert isinstance(parameter, GammaParameter)
            conditionals_tangent = np.zeros_like(conditionals)
            marginals_tangent = np.zeros_like(marginals)

        # compute the tangent of probabilities with respect to the parameter
        probabilities_tangent = (
            conditionals_tangent * self.groups.expand(marginals) +
            conditionals * self.groups.expand(marginals_tangent)
        )
        return probabilities_tangent, conditionals_tangent

    def compute_probabilities_by_xi_tensor(
            self, probabilities: Array, conditionals: Optional[Array]) -> Tuple[Array, Optional[Array]]:
        """Use choice probabilities to compute their tensor derivatives (holding beta fixed) with respect to xi
        (equivalently, to delta), indexed with the first axis.
        """
        probabilities_tensor = -probabilities[None] * probabilities[None].swapaxes(0, 1)
        probabilities_tensor[np.diag_indices(self.J)] += probabilities
        conditionals_tensor = None

        if self.epsilon_scale != 1:
            probabilities_tensor /= self.epsilon_scale

        if self.H > 0:
            assert conditionals is not None
            membership = self.get_membership_matrix()
            multiplied_probabilities = self.rho / (1 - self.rho) * probabilities
            multiplied_conditionals = 1 / (1 - self.rho) * conditionals
            probabilities_tensor -= membership[..., None] * (
                conditionals[None] * multiplied_probabilities[None].swapaxes(0, 1)
            )
            conditionals_tensor = -membership[..., None] * (
                conditionals[None] * multiplied_conditionals[None].swapaxes(0, 1)
            )
            probabilities_tensor[np.diag_indices(self.J)] += multiplied_probabilities
            conditionals_tensor[np.diag_indices(self.J)] += multiplied_conditionals

        return probabilities_tensor, conditionals_tensor

    def compute_shares_by_variable_jacobian(
            self, utility_derivatives: Array, probabilities: Optional[Array] = None,
            conditionals: Optional[Array] = None) -> Array:
        """Compute the Jacobian of market shares with respect to a variable. By default, compute unchanged choice
        probabilities.
        """
        if probabilities is None:
            probabilities, conditionals = self.compute_probabilities()
        value_derivatives = probabilities * utility_derivatives
        capital_lamda = self.compute_capital_lamda(value_derivatives)
        capital_gamma = self.compute_capital_gamma(value_derivatives, probabilities, conditionals)
        return capital_lamda - capital_gamma

    def compute_shares_by_xi_jacobian(self, probabilities: Array, conditionals: Optional[Array]) -> Array:
        """Compute the Jacobian (holding beta fixed) of shares with respect to xi (equivalently, to delta)."""
        diagonal_shares = np.diagflat(self.products.shares)
        weighted_probabilities = self.agents.weights * probabilities.T
        jacobian = diagonal_shares - probabilities @ weighted_probabilities

        if self.epsilon_scale != 1:
            jacobian /= self.epsilon_scale

        if self.H > 0:
            membership = self.get_membership_matrix()
            jacobian += self.rho / (1 - self.rho) * (
                diagonal_shares - membership * (conditionals @ weighted_probabilities)
            )

        return jacobian

    def compute_shares_by_theta_jacobian(
            self, delta: Array, probabilities: Array, conditionals: Optional[Array]) -> Array:
        """Compute the Jacobian of shares with respect to theta."""
        jacobian = np.zeros((self.J, self.parameters.P), options.dtype)
        for p, parameter in enumerate(self.parameters.unfixed):
            tangent, _ = self.compute_probabilities_by_parameter_tangent(parameter, probabilities, conditionals, delta)
            jacobian[:, [p]] = tangent @ self.agents.weights

        return jacobian

    def compute_capital_lamda_by_parameter_tangent(
            self, parameter: Parameter, value_derivatives: Array, value_derivatives_tangent: Array) -> Array:
        """Compute the tangent of the diagonal capital lambda matrix with respect to a parameter."""
        diagonal = value_derivatives_tangent @ self.agents.weights

        if self.H > 0:
            diagonal /= 1 - self.rho
            if isinstance(parameter, RhoParameter):
                associations = self.groups.expand(parameter.get_group_associations(self.groups))
                diagonal += associations / (1 - self.rho)**2 * (value_derivatives @ self.agents.weights)

        return np.diagflat(diagonal)

    def compute_capital_lamda_by_xi_tensor(self, value_derivatives_tensor: Array) -> Array:
        """Compute the tensor derivative of the diagonal capital lambda matrix with respect to xi, indexed by the first
        axis.
        """
        diagonal = value_derivatives_tensor @ self.agents.weights

        if self.H > 0:
            diagonal /= 1 - self.rho[None]

        tensor = np.zeros((self.J, self.J, self.J), options.dtype)
        tensor[:, np.arange(self.J), np.arange(self.J)] = np.c_[np.squeeze(diagonal)]
        return tensor

    def compute_capital_gamma_by_parameter_tangent(
            self, parameter: Parameter, value_derivatives: Array, value_derivatives_tangent: Array,
            probabilities: Array, probabilities_tangent: Array, conditionals: Optional[Array],
            conditionals_tangent: Optional[Array]) -> Array:
        """Compute the tangent of the dense capital gamma matrix with respect to a parameter."""
        weighted_value_derivatives = self.agents.weights * value_derivatives.T
        weighted_value_derivatives_tangent = self.agents.weights * value_derivatives_tangent.T
        tangent = (
            probabilities_tangent @ weighted_value_derivatives +
            probabilities @ weighted_value_derivatives_tangent
        )

        if self.H > 0:
            assert conditionals is not None and conditionals_tangent is not None
            membership = self.get_membership_matrix()
            tangent += membership * self.rho / (1 - self.rho) * (
                conditionals_tangent @ weighted_value_derivatives +
                conditionals @ weighted_value_derivatives_tangent
            )
            if isinstance(parameter, RhoParameter):
                associations = self.groups.expand(parameter.get_group_associations(self.groups))
                tangent += associations * membership / (1 - self.rho)**2 * (conditionals @ weighted_value_derivatives)

        return tangent

    def compute_capital_gamma_by_xi_tensor(
            self, value_derivatives: Array, value_derivatives_tensor: Array, probabilities: Array,
            probabilities_tensor: Array, conditionals: Optional[Array], conditionals_tensor: Optional[Array]) -> Array:
        """Compute the tensor derivative of the dense capital gamma matrix with respect to xi, indexed with the first
        axis.
        """
        weighted_value_derivatives = self.agents.weights * value_derivatives.T
        weighted_probabilities = self.agents.weights.T * probabilities
        tensor = (
            probabilities_tensor @ weighted_value_derivatives +
            weighted_probabilities @ value_derivatives_tensor.swapaxes(1, 2)
        )

        if self.H > 0:
            assert conditionals is not None and conditionals_tensor is not None
            membership = self.get_membership_matrix()
            weighted_conditionals = self.agents.weights.T * conditionals
            tensor += membership[None] * self.rho[None] / (1 - self.rho[None]) * (
                conditionals_tensor @ weighted_value_derivatives +
                weighted_conditionals @ value_derivatives_tensor.swapaxes(1, 2)
            )

        return tensor

    def compute_eta_by_theta_jacobian(self, xi_jacobian: Array) -> Tuple[Array, List[Error]]:
        """Compute the Jacobian of the markup term in the BLP-markup equation with respect to theta."""
        errors: List[Error] = []

        # compute derivatives of aggregate inclusive values with respect to prices
        probabilities, conditionals = self.compute_probabilities()
        utility_derivatives = self.compute_utility_derivatives('prices')
        value_derivatives = probabilities * utility_derivatives

        # compute the capital delta matrix, which, when inverted and multiplied by shares, gives eta
        ownership = self.get_ownership_matrix()
        capital_lamda = self.compute_capital_lamda(value_derivatives)
        capital_gamma = self.compute_capital_gamma(value_derivatives, probabilities, conditionals)
        capital_delta = -ownership * (capital_lamda - capital_gamma)

        # compute the inverse of capital delta and use it to compute eta
        capital_delta_inverse, replacement = approximately_invert(capital_delta)
        if replacement:
            errors.append(exceptions.IntraFirmJacobianInversionError(capital_delta, replacement))
        eta = capital_delta_inverse @ self.products.shares

        # compute the tensor derivative (holding beta fixed) with respect to xi (equivalently, to delta), indexed with
        # the first axis, of derivatives of aggregate inclusive values
        probabilities_tensor, conditionals_tensor = self.compute_probabilities_by_xi_tensor(probabilities, conditionals)
        value_derivatives_tensor = probabilities_tensor * utility_derivatives

        # compute the tensor derivatives (holding beta fixed) of capital delta with respect to xi (equivalently, to
        #   delta)
        capital_lamda_tensor = self.compute_capital_lamda_by_xi_tensor(value_derivatives_tensor)
        capital_gamma_tensor = self.compute_capital_gamma_by_xi_tensor(
            value_derivatives, value_derivatives_tensor, probabilities, probabilities_tensor, conditionals,
            conditionals_tensor
        )
        capital_delta_tensor = -ownership[None] * (capital_lamda_tensor - capital_gamma_tensor)

        # compute the product of the tensor and eta
        capital_delta_tensor_times_eta = np.c_[np.squeeze(capital_delta_tensor @ eta)]

        # compute derivatives of X1 and X2 with respect to prices
        X1_derivatives = self.compute_X1_derivatives('prices')
        X2_derivatives = self.compute_X2_derivatives('prices')

        # fill the Jacobian of eta with respect to theta parameter-by-parameter
        eta_jacobian = np.zeros((self.J, self.parameters.P), options.dtype)
        for p, parameter in enumerate(self.parameters.unfixed):
            # compute the tangent with respect to the parameter of derivatives of aggregate inclusive values
            probabilities_tangent, conditionals_tangent = self.compute_probabilities_by_parameter_tangent(
                parameter, probabilities, conditionals
            )
            utility_derivatives_tangent = self.compute_utility_derivatives_by_parameter_tangent(
                parameter, X1_derivatives, X2_derivatives
            )
            value_derivatives_tangent = (
                probabilities_tangent * utility_derivatives +
                probabilities * utility_derivatives_tangent
            )

            # compute the tangent of capital delta with respect to the parameter
            capital_lamda_tangent = self.compute_capital_lamda_by_parameter_tangent(
                parameter, value_derivatives, value_derivatives_tangent
            )
            capital_gamma_tangent = self.compute_capital_gamma_by_parameter_tangent(
                parameter, value_derivatives, value_derivatives_tangent, probabilities, probabilities_tangent,
                conditionals, conditionals_tangent
            )
            capital_delta_tangent = -ownership * (capital_lamda_tangent - capital_gamma_tangent)

            # extract the tangent of xi with respect to the parameter and compute the associated tangent of eta
            eta_jacobian[:, [p]] = -capital_delta_inverse @ (
                capital_delta_tangent @ eta + capital_delta_tensor_times_eta.T @ xi_jacobian[:, [p]]
            )

        return eta_jacobian, errors

    def compute_xi_by_theta_jacobian(self, delta: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Use the Implicit Function Theorem to compute the Jacobian (holding beta fixed) of xi (equivalently, of delta)
        with respect to theta. By default, use unchanged delta values.
        """
        errors: List[Error] = []
        if delta is None:
            assert self.delta is not None
            delta = self.delta

        probabilities, conditionals = self.compute_probabilities(delta)
        shares_by_xi_jacobian = self.compute_shares_by_xi_jacobian(probabilities, conditionals)
        shares_by_theta_jacobian = self.compute_shares_by_theta_jacobian(delta, probabilities, conditionals)
        xi_by_theta_jacobian, replacement = approximately_solve(shares_by_xi_jacobian, -shares_by_theta_jacobian)
        if replacement:
            errors.append(exceptions.SharesByXiJacobianInversionError(shares_by_xi_jacobian, replacement))

        return xi_by_theta_jacobian, errors

    def compute_omega_by_theta_jacobian(self, tilde_costs: Array, xi_jacobian: Array) -> Tuple[Array, List[Error]]:
        """Compute the Jacobian (holding gamma fixed) of omega (equivalently, of transformed marginal costs) with
        respect to theta.
        """
        errors: List[Error] = []

        # compute the Jacobian of the markup term in the BLP-markup equation with respect to theta
        eta_jacobian, eta_jacobian_errors = self.compute_eta_by_theta_jacobian(xi_jacobian)
        errors.extend(eta_jacobian_errors)

        # transform the Jacobian according to the marginal cost specification
        if self.costs_type == 'linear':
            omega_jacobian = -eta_jacobian
        else:
            assert self.costs_type == 'log'
            omega_jacobian = -eta_jacobian / np.exp(tilde_costs)

        # incorporate the contributions from any parameters in gamma that haven't been concentrated out
        for p, parameter in enumerate(self.parameters.unfixed):
            if isinstance(parameter, GammaParameter):
                omega_jacobian[:, [p]] = -parameter.get_product_characteristic(self)
        return omega_jacobian, errors

    def compute_micro_values(
            self, delta: Optional[Array] = None) -> (
            Tuple[Array, Array, Optional[Array], Optional[Array], Dict[int, Array], Optional[Array], Dict[int, Array]]):
        """Compute micro moment values. By default, use the delta with which this market was initialized. Return any
        probabilities that were used so they don't have to be re-computed when computing related outputs.
        """
        assert self.moments is not None
        if delta is None:
            assert self.delta is not None
            delta = self.delta

        # pre-compute probabilities
        probabilities, conditionals = self.compute_probabilities(delta)

        # pre-compute probabilities conditional on purchasing an inside good
        inside_probabilities = None
        requires_inside_probabilities = lambda m: any([
            isinstance(m, (DemographicCovarianceMoment, DiversionCovarianceMoment)),
            isinstance(m, DiversionProbabilityMoment) and m.product_id1 is None,
        ])
        if any(requires_inside_probabilities(m) for m in self.moments.micro_moments):
            inside_probabilities = self.compute_eliminated_probabilities(probabilities, delta, eliminate_outside=True)

        # pre-compute second choice probabilities
        eliminated_probabilities: Dict[int, Array] = {}
        for moment in self.moments.micro_moments:
            if isinstance(moment, DiversionProbabilityMoment):
                j = self.get_product(moment.product_id1)
                if j not in eliminated_probabilities:
                    eliminated_probabilities[j] = self.compute_eliminated_probabilities(
                        probabilities, delta, eliminate_product=j
                    )

        # pre-compute second choice probabilities conditional on purchasing an inside good (also compute the sum of
        #   inside probability products over all first choices)
        inside_eliminated_sum = None
        inside_eliminated_probabilities: Dict[int, Array] = {}
        if any(isinstance(m, DiversionCovarianceMoment) for m in self.moments.micro_moments):
            assert inside_probabilities is not None
            inside_eliminated_sum = np.zeros((self.J, self.I), options.dtype)
            for j in range(self.J):
                inside_eliminated_probabilities[j] = self.compute_eliminated_probabilities(
                    probabilities, delta, eliminate_outside=True, eliminate_product=j
                )
                inside_eliminated_sum += inside_probabilities[[j]] * inside_eliminated_probabilities[j]

        # compute the micro moment values
        micro_values = np.zeros((self.moments.MM, 1), options.dtype)
        for m, moment in enumerate(self.moments.micro_moments):
            micro_values[m] = self.agents.weights.T @ self.compute_agent_micro_values(
                moment, delta, probabilities, conditionals, inside_probabilities, eliminated_probabilities,
                inside_eliminated_sum
            )

        return (
            micro_values, probabilities, conditionals, inside_probabilities, eliminated_probabilities,
            inside_eliminated_sum, inside_eliminated_probabilities
        )

    def compute_agent_micro_values(
            self, moment: Moment, delta: Array, probabilities: Array, conditionals: Optional[Array],
            inside_probabilities: Optional[Array], eliminated_probabilities: Dict[int, Array],
            inside_eliminated_sum: Optional[Array]) -> Array:
        """Compute agent-specific micro moment values, which will be aggregated up into means or covariances."""

        # match a demographic expectation for agents who choose the outside good
        if isinstance(moment, DemographicExpectationMoment) and moment.product_id is None:
            assert probabilities is not None
            d = self.agents.demographics[:, [moment.demographics_index]]
            outside_probabilities = 1 - probabilities.sum(axis=0, keepdims=True).T
            outside_share = 1 - self.products.shares.sum()
            return d * outside_probabilities / outside_share

        # match a demographic expectation for agents who choose a certain inside good
        if isinstance(moment, DemographicExpectationMoment):
            assert probabilities is not None
            j = self.get_product(moment.product_id)
            d = self.agents.demographics[:, [moment.demographics_index]]
            return d * probabilities[[j]].T / self.products.shares[j]

        # match a covariance between a product characteristic and a demographic
        if isinstance(moment, DemographicCovarianceMoment):
            assert inside_probabilities is not None
            x = self.products.X2[:, [moment.X2_index]]
            d = self.agents.demographics[:, [moment.demographics_index]]
            z = inside_probabilities.T @ x
            demeaned_z = z - self.agents.weights.T @ z
            demeaned_d = d - self.agents.weights.T @ d
            return demeaned_z * demeaned_d

        # match the second choice probability of a certain inside good for agents who choose the outside good
        if isinstance(moment, DiversionProbabilityMoment) and moment.product_id1 is None:
            assert inside_probabilities is not None
            k = self.get_product(moment.product_id2)
            outside_share = 1 - self.products.shares.sum()
            numerator = inside_probabilities[[k]].T - self.products.shares[k]
            return numerator / outside_share

        # match the second choice probability of the outside good for agents who choose a certain inside good
        if isinstance(moment, DiversionProbabilityMoment) and moment.product_id2 is None:
            j = self.get_product(moment.product_id1)
            eliminated_outside_probabilities = 1 - eliminated_probabilities[j].sum(axis=0, keepdims=True)
            outside_share = 1 - self.products.shares.sum()
            numerator = eliminated_outside_probabilities.T - outside_share
            return numerator / self.products.shares[j]

        # match the second choice probability of a certain inside good for agents who choose a certain inside good
        if isinstance(moment, DiversionProbabilityMoment):
            j = self.get_product(moment.product_id1)
            k = self.get_product(moment.product_id2)
            numerator = eliminated_probabilities[j][[k]].T - self.products.shares[k]
            return numerator / self.products.shares[j]

        # match a covariance between product characteristics of first and second choices
        if isinstance(moment, DiversionCovarianceMoment):
            assert inside_probabilities is not None and inside_eliminated_sum is not None
            x1 = self.products.X2[:, [moment.X2_index1]]
            x2 = self.products.X2[:, [moment.X2_index2]]
            z1 = inside_probabilities.T @ x1
            z2 = inside_eliminated_sum.T @ x2
            demeaned_z1 = z1 - self.agents.weights.T @ z1
            demeaned_z2 = z2 - self.agents.weights.T @ z2
            return demeaned_z1 * demeaned_z2

        def compute_derivatives() -> Array:
            """Compute derivatives of probabilities with respect to theta for use by some custom micro moments."""

            # compute contributions from direct dependence on theta
            probabilities_by_theta = np.zeros((self.J, self.I, self.parameters.P), options.dtype)
            for p, parameter in enumerate(self.parameters.unfixed):
                probabilities_by_theta[:, :, p], _ = self.compute_probabilities_by_parameter_tangent(
                    parameter, probabilities, conditionals, delta
                )

            # compute contributions from indirect dependence on theta through delta
            probabilities_by_xi, _ = self.compute_probabilities_by_xi_tensor(probabilities, conditionals)
            xi_by_theta, _ = self.compute_xi_by_theta_jacobian(delta)

            return probabilities_by_theta + np.moveaxis(probabilities_by_xi, 0, 2) @ xi_by_theta

        # match a custom moment
        assert isinstance(moment, CustomMoment)
        values = moment.compute_custom(
            self.t, self.sigma, self.pi, self.rho, self.products, self.agents, delta, self.mu, probabilities,
            compute_derivatives
        )
        values = np.asarray(values, options.dtype)
        if values.size != self.I:
            raise ValueError("compute_custom must return a vector with as many elements as there are agents.")
        return np.c_[values.flatten()]
