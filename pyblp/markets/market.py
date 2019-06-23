"""Market underlying the BLP model."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import exceptions, options
from ..configurations.iteration import ContractionResults, Iteration
from ..economies.economy import Economy
from ..moments import EconomyMoments, MarketMoments, ProductsAgentsCovarianceMoment
from ..parameters import (
    LinearCoefficient, NonlinearCoefficient, Parameter, Parameters, PiParameter, RhoParameter, SigmaParameter
)
from ..primitives import Container
from ..utilities.algebra import approximately_invert, approximately_solve
from ..utilities.basics import Array, Error, Groups, SolverStats, update_matrices


class Market(Container):
    """A market underlying the BLP model."""

    groups: Groups
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
    rho_size: int
    group_rho: Array
    rho: Array
    delta: Optional[Array]
    mu: Array
    parameters: Parameters
    moments: Optional[MarketMoments]

    def __init__(
            self, economy: Economy, t: Any, parameters: Parameters, sigma: Array, pi: Array, rho: Array,
            beta: Optional[Array] = None, delta: Optional[Array] = None, moments: Optional[EconomyMoments] = None,
            data_override: Optional[Dict] = None) -> None:
        """Store or compute information about formulations, data, parameters, and utility."""

        # structure relevant data
        super().__init__(
            economy.products[economy._product_market_indices[t]],
            economy.agents[economy._agent_market_indices[t]]
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
                self.products[name][:] = variable[:]
            for index, formulation in enumerate(self._X2_formulations):
                if any(n in formulation.names for n in data_override):
                    self.products.X2[:, [index]] = formulation.evaluate(self.products)

        # store parameters (expand rho to all groups and all products)
        self.parameters = parameters
        self.sigma = sigma
        self.pi = pi
        self.beta = beta
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
        tiled_ids = np.tile(self.products.firm_ids, self.J)
        return np.where(tiled_ids == tiled_ids.T, 1, 0)

    def compute_random_coefficients(self, sigma: Optional[Array] = None, pi: Optional[Array] = None) -> Array:
        """Compute the random coefficients by weighting agent characteristics with nonlinear parameters. By default, use
        unchanged parameters.
        """
        if sigma is None:
            sigma = self.sigma
        if pi is None:
            pi = self.pi
        coefficients = sigma @ self.agents.nodes.T
        if self.D > 0:
            coefficients += pi @ self.agents.demographics.T
        return coefficients

    def compute_mu(
            self, X2: Optional[Array] = None, sigma: Optional[Array] = None, pi: Optional[Array] = None) -> Array:
        """Compute mu. By default, use unchanged X2 and parameters."""
        if X2 is None:
            X2 = self.products.X2
        return X2 @ self.compute_random_coefficients(sigma, pi)

    def compute_default_bounds(self, parameters: List[Parameter]) -> List[Tuple[Array, Array]]:
        """Compute default bounds for nonlinear parameters."""

        # define a function to normalize bounds
        def normalize(x: float) -> float:
            """Reduce an initial parameter bound by 1% and round it to two significant figures."""
            if not np.isfinite(x) or x == 0:
                return x
            reduced = 0.99 * x
            return np.round(reduced, 1 + int(reduced < 1) - int(np.log10(reduced)))

        # compute common components of default bounds
        mu_norm = np.abs(self.mu).max()
        mu_max = np.log(np.finfo(np.float64).max)
        bounds: List[Tuple[Array, Array]] = []

        # compute the bounds parameter-by-parameter
        for parameter in parameters:
            if isinstance(parameter, SigmaParameter):
                sigma = self.sigma.copy()
                sigma[parameter.location] = 0
                additional_mu_norm = np.abs(self.compute_mu(sigma=sigma)).max()
                v_norm = np.abs(parameter.get_agent_characteristic(self)).max()
                x_norm = np.abs(parameter.get_product_characteristic(self)).max()
                bound = normalize(max(0, mu_max - additional_mu_norm) / v_norm / x_norm)
                bounds.append((-bound if parameter.location[0] != parameter.location[1] else 0, bound))
            elif isinstance(parameter, PiParameter):
                pi = self.pi.copy()
                pi[parameter.location] = 0
                additional_mu_norm = np.abs(self.compute_mu(pi=pi)).max()
                v_norm = np.abs(parameter.get_agent_characteristic(self)).max()
                x_norm = np.abs(parameter.get_product_characteristic(self)).max()
                bound = normalize(max(0, mu_max - additional_mu_norm) / v_norm / x_norm)
                bounds.append((-bound, bound))
            else:
                assert isinstance(parameter, RhoParameter)
                bounds.append((0, normalize(1 - min(1, mu_norm / mu_max))))
        return bounds

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
                delta += self.beta[index] * (formulation.evaluate(self.products, override) - self.products[name])
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
        return derivatives

    def compute_probabilities(
            self, delta: Array = None, mu: Optional[Array] = None, linear: bool = True, safe: bool = True,
            utility_reduction: Optional[Array] = None, numerator: Optional[Array] = None,
            eliminate_outside: bool = False, eliminate_product: Optional[int] = None) -> Tuple[Array, Optional[Array]]:
        """Compute choice probabilities. By default, use unchanged delta and mu values. If linear is False, delta and mu
        must be specified and already be exponentiated. If safe is True, scale the logit equation by the exponential of
        negative the maximum utility for each agent, and if utility_reduction is specified, it should be values that
        have already been subtracted from the specified utility for each agent. If the numerator is specified, it will
        be used as the numerator in the non-nested logit expression. If eliminate_outside is True, eliminate the outside
        option from the choice set. If eliminate_product is specified, eliminate the product associated with the
        specified index from the choice set.
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
            exp_utilities = np.array(delta * mu)
            if self.H > 0:
                exp_utilities **= 1 / (1 - self.rho)
        else:
            utilities = delta + mu
            if self.H > 0:
                utilities /= 1 - self.rho
            if safe:
                utility_reduction = np.max(utilities, axis=0, keepdims=True)
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
            self, ownership_matrix: Optional[Array] = None, utility_derivatives: Optional[Array] = None,
            prices: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Compute the markup term in the BLP-markup equation. By default, get an unchanged ownership matrix, compute
        derivatives of utilities with respect to prices, and use unchanged prices.
        """
        errors: List[Error] = []
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
        jacobian = self.compute_shares_by_variable_jacobian(utility_derivatives, probabilities, conditionals)
        intra_firm_jacobian = ownership_matrix * jacobian
        eta, replacement = approximately_solve(intra_firm_jacobian, -shares)
        if replacement:
            errors.append(exceptions.IntraFirmJacobianInversionError(intra_firm_jacobian, replacement))
        return eta, errors

    def compute_zeta(
            self, costs: Array, ownership_matrix: Optional[Array] = None, utility_derivatives: Optional[Array] = None,
            prices: Optional[Array] = None) -> Tuple[Array, Array]:
        """Compute the markup term in the zeta-markup equation. By default, get an unchanged ownership matrix, compute
        derivatives of utilities with respect to prices, and use unchanged prices. Also return the intermediate
        diagonal of the capital lambda matrix, which is used for weighting during fixed point iteration.
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
        value_derivatives = probabilities * utility_derivatives
        capital_lamda_diagonal = self.compute_capital_lamda(value_derivatives).diagonal()
        capital_lamda_inverse = np.diag(1 / capital_lamda_diagonal)
        capital_gamma = self.compute_capital_gamma(value_derivatives, probabilities, conditionals)
        tilde_capital_omega = capital_lamda_inverse @ (ownership_matrix * capital_gamma).T
        zeta = tilde_capital_omega @ (prices - costs) - capital_lamda_inverse @ shares
        return zeta, capital_lamda_diagonal

    def compute_equilibrium_prices(
            self, costs: Array, iteration: Iteration, ownership_matrix: Optional[Array] = None,
            prices: Optional[Array] = None) -> Tuple[Array, SolverStats]:
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

        # define the contraction
        def contraction(x: Array) -> ContractionResults:
            """Compute the next equilibrium prices."""
            zeta, capital_lamda_diagonal = self.compute_zeta(costs, ownership_matrix, get_derivatives(x), x)
            x = costs + zeta
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
        if isinstance(parameter, LinearCoefficient):
            tangent += X1_derivatives[:, [parameter.location[0]]]
        elif isinstance(parameter, NonlinearCoefficient):
            v = parameter.get_agent_characteristic(self)
            tangent += X2_derivatives[:, [parameter.location[0]]] @ v.T
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
            if isinstance(parameter, LinearCoefficient):
                x = parameter.get_product_characteristic(self)
                probabilities_tangent = probabilities * (x - x.T @ probabilities)
            else:
                assert isinstance(parameter, NonlinearCoefficient)
                v = parameter.get_agent_characteristic(self)
                x = parameter.get_product_characteristic(self)
                probabilities_tangent = probabilities * v.T * (x - x.T @ probabilities)
            return probabilities_tangent, None

        # marginal probabilities are needed to compute tangents with nesting
        marginals = self.groups.sum(probabilities)

        # compute the tangent of conditional and marginal probabilities with respect to the parameter
        if isinstance(parameter, LinearCoefficient):
            x = parameter.get_product_characteristic(self)

            # compute the tangent of conditional probabilities with respect to the parameter
            A = conditionals * x
            A_sums = self.groups.sum(A)
            conditionals_tangent = conditionals * (x - self.groups.expand(A_sums)) / (1 - self.rho)

            # compute the tangent of marginal probabilities with respect to the parameter
            B = marginals * A_sums
            marginals_tangent = B - marginals * B.sum(axis=0, keepdims=True)
        elif isinstance(parameter, NonlinearCoefficient):
            v = parameter.get_agent_characteristic(self)
            x = parameter.get_product_characteristic(self)

            # compute the tangent of conditional probabilities with respect to the parameter
            A = conditionals * x
            A_sums = self.groups.sum(A)
            conditionals_tangent = conditionals * v.T * (x - self.groups.expand(A_sums)) / (1 - self.rho)

            # compute the tangent of marginal probabilities with respect to the parameter
            B = marginals * A_sums * v.T
            marginals_tangent = B - marginals * B.sum(axis=0, keepdims=True)
        else:
            assert isinstance(parameter, RhoParameter)
            group_associations = parameter.get_group_associations(self.groups)
            associations = self.groups.expand(group_associations)

            # utilities are needed to compute tangents with respect to rho
            utilities = (delta + mu) / (1 - self.rho)

            # compute the tangent of conditional probabilities with respect to the parameter
            A = conditionals * utilities / (1 - self.rho)
            A_sums = self.groups.sum(A)
            conditionals_tangent = associations * (A - conditionals * self.groups.expand(A_sums))

            # compute the tangent of marginal probabilities with respect to the parameter (re-scale for robustness)
            max_utilities = np.max(utilities, axis=0, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                B = marginals * (
                    A_sums * (1 - self.group_rho) -
                    (np.log(self.groups.sum(np.exp(utilities - max_utilities))) + max_utilities)
                )
                marginals_tangent = group_associations * B - marginals * (group_associations.T @ B)
            marginals_tangent[~np.isfinite(marginals_tangent)] = 0

        # compute the tangent of probabilities with respect to the parameter
        probabilities_tangent = (
            conditionals_tangent * self.groups.expand(marginals) +
            conditionals * self.groups.expand(marginals_tangent)
        )
        return probabilities_tangent, conditionals_tangent

    def compute_probabilities_by_xi_tensor(
            self, probabilities: Array, conditionals: Optional[Array]) -> Tuple[Array, Optional[Array]]:
        """Use choice probabilities to compute their tensor derivatives with respect to xi (equivalently, to delta),
        indexed with the first axis.
        """
        probabilities_tensor = -probabilities[None] * probabilities[None].swapaxes(0, 1)
        probabilities_tensor[np.diag_indices(self.J)] += probabilities
        conditionals_tensor = None
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
        """Compute the Jacobian of shares with respect to xi (equivalently, to delta)."""
        diagonal_shares = np.diagflat(self.products.shares)
        weighted_probabilities = self.agents.weights * probabilities.T
        jacobian = diagonal_shares - probabilities @ weighted_probabilities
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
        tensor[:, np.arange(self.J), np.arange(self.J)] = np.squeeze(diagonal)
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

        # compute the tensor derivative with respect to xi (equivalently, to delta), indexed with the first axis, of
        #   derivatives of aggregate inclusive values
        probabilities_tensor, conditionals_tensor = self.compute_probabilities_by_xi_tensor(probabilities, conditionals)
        value_derivatives_tensor = probabilities_tensor * utility_derivatives

        # compute the tensor derivative of capital delta with respect to xi (equivalently, to delta)
        capital_lamda_tensor = self.compute_capital_lamda_by_xi_tensor(value_derivatives_tensor)
        capital_gamma_tensor = self.compute_capital_gamma_by_xi_tensor(
            value_derivatives, value_derivatives_tensor, probabilities, probabilities_tensor, conditionals,
            conditionals_tensor
        )
        capital_delta_tensor = -ownership[None] * (capital_lamda_tensor - capital_gamma_tensor)

        # compute the product of the tensor and eta
        capital_delta_tensor_times_eta = np.squeeze(capital_delta_tensor @ eta)

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

        # return the filled Jacobian
        return eta_jacobian, errors

    def compute_xi_by_theta_jacobian(self, delta: Optional[Array] = None) -> Tuple[Array, List[Error]]:
        """Use the Implicit Function Theorem to compute the Jacobian of xi (equivalently, of delta) with respect to
        theta. By default, use unchanged delta values.
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

    def compute_omega_by_theta_jacobian(
            self, tilde_costs: Array, xi_jacobian: Array, costs_type: str) -> Tuple[Array, List[Error]]:
        """Compute the Jacobian of omega (equivalently, of transformed marginal costs) with respect to theta."""
        errors: List[Error] = []
        eta_jacobian, eta_jacobian_errors = self.compute_eta_by_theta_jacobian(xi_jacobian)
        errors.extend(eta_jacobian_errors)
        if costs_type == 'linear':
            omega_jacobian = -eta_jacobian
        else:
            assert costs_type == 'log'
            omega_jacobian = -eta_jacobian / np.exp(tilde_costs)
        return omega_jacobian, errors

    def compute_micro(self, delta: Optional[Array] = None) -> Tuple[Array, Array, Array]:
        """Compute micro moments. By default, use the delta with which this market was initialized. Also return the
        probabilities with the outside option eliminated so they can be re-used when computing other things related to
        micro moments.
        """
        assert self.moments is not None

        # compute probabilities with the outside option eliminated
        micro_probabilities, micro_conditionals = self.compute_probabilities(delta, eliminate_outside=True)

        # compute the micro moments
        micro = np.zeros((self.moments.MM, 1), options.dtype)
        for m, moment in enumerate(self.moments.micro_moments):
            assert isinstance(moment, ProductsAgentsCovarianceMoment)
            z = micro_probabilities.T @ self.products.X2[:, [moment.X2_index]]
            d = self.agents.demographics[:, [moment.demographics_index]]
            demeaned_z = z - z.T @ self.agents.weights
            demeaned_d = d - d.T @ self.agents.weights
            micro[m] = demeaned_z.T @ (self.agents.weights * demeaned_d) - moment.value
        return micro, micro_probabilities, micro_conditionals
