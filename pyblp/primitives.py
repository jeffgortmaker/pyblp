"""Primitive structures that constitute the foundation of the BLP model."""

import itertools
import collections

import numpy as np
import scipy.linalg
import numpy.lib.recfunctions

from . import options, exceptions
from .configurations import Formulation, Integration
from .utilities import output, extract_matrix, Matrices


class Products(Matrices):
    r"""Structured product data, which contains the following fields:

        - **market_ids** : (`object`) - IDs that associate products with markets.

        - **firm_ids** : (`object`) - IDs that associate products with firms. Any columns after the first represent
          changes such as mergers.

        - **demand_ids** : (`object`) - Categorical variables used to create demand-side fixed effects.

        - **supply_ids** : (`object`) - Categorical variables used to create supply-side fixed effects.

        - **clustering_ids** (`object`) - IDs used to compute clustered standard errors.

        - **ownership** : (`object`) - Stacked :math:`J_t \times J_t` ownership matrices, :math:`O`, for each market
          :math:`t`. Each stack is associated with a `firm_ids` column.

        - **shares** : (`numeric`) - Market shares, :math:`s`.

        - **ZD** : (`numeric`) - Demand-side instruments, :math:`Z_D`, which may have been demeaned to absorb any
          demand-side fixed effects.

        - **ZS** : (`numeric`) - Supply-side instruments, :math:`Z_S`, which may have been demeaned to absorb any
          supply-side fixed effects.

        - **X1** : (`numeric`) - Linear product characteristics, :math:`X_1`, which may have been demeaned to absorb any
          demand-side fixed effects.

        - **X2** : (`numeric`) - Nonlinear product characteristics, :math:`X_2`.

        - **X3** : (`numeric`) - Cost product characteristics, :math:`X_3`, which may have been demeaned to absorb any
          supply-side fixed effects.

        - **prices** : (`numeric`) - Product prices, :math:`p`.

    Any additional fields are the variables underlying `X1`, `X2`, and `X3`.

    """

    def __new__(cls, product_formulations, product_data):
        """Structure product data while absorbing any fixed effects."""

        # validate the formulations
        if isinstance(product_formulations, Formulation):
            product_formulations = [product_formulations]
        elif not isinstance(product_formulations, (list, tuple)) or len(product_formulations) > 3:
            raise TypeError("product_formulations must be a Formulation instance of a tuple of up to three instances.")
        if not all(isinstance(f, Formulation) or f is None for f in product_formulations):
            raise TypeError("Each formulation in product_formulations must be a Formulation instance or None.")
        product_formulations = list(product_formulations) + [None] * (3 - len(product_formulations))
        if product_formulations[0] is None:
            raise ValueError("The formulation for X1 must be specified.")
        if product_formulations[1] is not None and product_formulations[1]._absorbed_terms:
            raise ValueError("The formulation for X2 does not support fixed effect absorption.")

        # build X1
        X1, X1_formulations, X1_data = product_formulations[0]._build_matrix(product_data)
        if 'shares' in X1_data:
            raise NameError("shares cannot be included in the formulation for X1.")

        # build X2
        X2 = None
        X2_data = {}
        X2_formulations = []
        if product_formulations[1] is not None:
            X2, X2_formulations, X2_data = product_formulations[1]._build_matrix(product_data)
            if 'shares' in X2_data:
                raise NameError("shares cannot be included in the formulation for X2.")

        # check that prices are in X1 or X2
        if 'prices' not in X1_data and 'prices' not in X2_data:
            raise NameError("prices must be included in at least one of formulations for X1 or X2.")

        # build X3
        X3 = None
        X3_data = {}
        X3_formulations = []
        if product_formulations[2] is not None:
            X3, X3_formulations, X3_data = product_formulations[2]._build_matrix(product_data)
            if 'shares' in X3_data:
                raise NameError("shares cannot be included in the formulation for X3.")
            if 'prices' in X3_data:
                raise NameError("prices cannot be included in the formulation for X3.")

        # load demand-side instruments
        ZD = extract_matrix(product_data, 'demand_instruments')
        if ZD is None:
            raise KeyError("product_data must have a demand_instruments field.")

        # load supply-side instruments
        ZS = None
        if X3 is not None:
            ZS = extract_matrix(product_data, 'supply_instruments')
            if ZS is None:
                raise KeyError("Since X3 is formulated, product_data must have a supply_instruments field.")

        # load and absorb any demand-side fixed effects
        demand_ids = None
        if product_formulations[0]._absorbed_terms:
            demand_ids = product_formulations[0]._build_ids(product_data)
            X1, X1_errors = product_formulations[0]._demean(X1, demand_ids)
            ZD, ZD_errors = product_formulations[0]._demean(ZD, demand_ids)
            if X1_errors | ZD_errors:
                raise exceptions.MultipleErrors(X1_errors | ZD_errors)

        # load and absorb any demand-side fixed effects
        supply_ids = None
        if product_formulations[2] is not None and product_formulations[2]._absorbed_terms:
            supply_ids = product_formulations[2]._build_ids(product_data)
            X3, X3_errors = product_formulations[2]._demean(X3, supply_ids)
            ZS, ZS_errors = product_formulations[2]._demean(ZS, supply_ids)
            if X3_errors | ZS_errors:
                raise exceptions.MultipleErrors(X3_errors | ZS_errors)

        # load other IDs
        market_ids = extract_matrix(product_data, 'market_ids')
        firm_ids = extract_matrix(product_data, 'firm_ids')
        clustering_ids = extract_matrix(product_data, 'clustering_ids')
        if market_ids is None:
            raise KeyError("product_data must have a market_ids field.")
        if market_ids.shape[1] > 1:
            raise ValueError("The market_ids field of product_data must be one-dimensional.")
        if firm_ids is None and X3 is not None:
            raise KeyError("product_data must have a firm_ids field when X3 is formulated.")
        if clustering_ids is not None:
            if clustering_ids.shape[1] > 1:
                raise ValueError("The clustering_ids field of product_data must be one-dimensional.")
            if np.unique(clustering_ids).size == 1:
                raise ValueError("The clustering_ids field of product_data must have at least two distinct IDs.")

        # load ownership matrices
        ownership = None
        if firm_ids is not None:
            ownership = extract_matrix(product_data, 'ownership')
            if ownership is not None:
                columns = np.unique(market_ids, return_counts=True)[1].max() * firm_ids.shape[1]
                if ownership.shape[1] != columns:
                    raise ValueError(f"The ownership field of product_data must have {columns} columns.")

        # load shares
        shares = extract_matrix(product_data, 'shares')
        if shares is None:
            raise KeyError("product_data must have a shares field.")
        if shares.shape[1] > 1:
            raise ValueError("The shares field of product_data must be one-dimensional.")

        # structure product fields as a mapping
        product_mapping = {
            'market_ids': (market_ids, np.object),
            'firm_ids': (firm_ids, np.object),
            'demand_ids': (demand_ids, np.object),
            'supply_ids': (supply_ids, np.object),
            'clustering_ids': (clustering_ids, np.object),
            'ownership': (ownership, options.dtype),
            'shares': (shares, options.dtype),
            'ZD': (ZD, options.dtype),
            'ZS': (ZS, options.dtype),
            (tuple(X1_formulations), 'X1'): (X1, options.dtype),
            (tuple(X2_formulations), 'X2'): (X2, options.dtype),
            (tuple(X3_formulations), 'X3'): (X3, options.dtype)
        }

        # supplement the mapping with variables underlying X1, X2, and X3
        underlying_data = {**X1_data, **X2_data, **X3_data}
        invalid_names = set(underlying_data) & {k if isinstance(k, str) else k[1] for k in product_mapping}
        if invalid_names:
            raise NameError(f"These reserved names in product_formulations are invalid: {list(invalid_names)}.")
        product_mapping.update({k: (v, options.dtype) for k, v in underlying_data.items()})

        # structure products
        return super().__new__(cls, product_mapping)


class Agents(Matrices):
    r"""Structured agent data, which contains the following fields:

        - **market_ids** : (`object`) - IDs that associate agents with markets.

        - **weights** : (`numeric`) - Integration weights, :math:`w`.

        - **nodes** : (`numeric`) - Unobserved agent characteristics called integration nodes, :math:`\nu`.

        - **demographics** : (`numeric`) - Observed agent characteristics, :math:`d`.

    """

    def __new__(cls, products, agent_formulation=None, agent_data=None, integration=None):
        """Structure agent data."""

        # if there are only linear characteristics, build a trivial set of agents
        K2 = products.X2.shape[1]
        if K2 == 0:
            if agent_formulation is not None or agent_data is not None or integration is not None:
                raise ValueError(
                    "Since X2 is not formulated, none of agent_formulation, agent_data, and integration should be "
                    "specified."
                )
            market_ids = np.unique(products.market_ids)
            weights = np.ones_like(market_ids, options.dtype)
            nodes = demographics = None
            demographics_formulations = []
        else:
            # validate the formulation and build demographics
            demographics = None
            demographics_formulations = []
            if agent_formulation is not None:
                if not isinstance(agent_formulation, Formulation):
                    raise TypeError("agent_formulation must be a Formulation instance.")
                if agent_data is None:
                    raise ValueError("Since agent_formulation is specified, agent_data must be specified as well.")
                if agent_formulation._absorbed_terms:
                    raise ValueError("agent_formulation does not support fixed effect absorption.")
                demographics, demographics_formulations = agent_formulation._build_matrix(agent_data)[:2]

            # load market IDs
            market_ids = None
            if agent_data is not None:
                market_ids = extract_matrix(agent_data, 'market_ids')
                if market_ids is None:
                    raise KeyError("agent_data must have a market_ids field.")
                if market_ids.shape[1] > 1:
                    raise ValueError("The market_ids field of agent_data must be one-dimensional.")
                if set(np.unique(products.market_ids)) != set(np.unique(market_ids)):
                    raise ValueError("The market_ids field of agent_data must have the same IDs as product data.")

            # build nodes and weights
            nodes = weights = None
            if integration is not None:
                if not isinstance(integration, Integration):
                    raise ValueError("integration must be an Integration instance.")
                loaded_market_ids = market_ids
                market_ids, nodes, weights = integration._build_many(K2, np.unique(products.market_ids))

                # delete rows of demographics if there are too many
                if demographics is not None:
                    demographics_list = []
                    for t in np.unique(market_ids):
                        built_rows = (market_ids == t).sum()
                        loaded_rows = (loaded_market_ids == t).sum()
                        demographics_t = demographics[loaded_market_ids.flat == t]
                        if built_rows > loaded_rows:
                            raise ValueError(f"Market '{t}' in agent_data must have at least {built_rows} rows.")
                        if built_rows < loaded_rows:
                            demographics_t = demographics_t[:built_rows]
                        demographics_list.append(demographics_t)
                    demographics = np.concatenate(demographics_list)

            # load any unbuilt nodes and weights
            if integration is None:
                nodes = extract_matrix(agent_data, 'nodes')
                weights = extract_matrix(agent_data, 'weights')
                if nodes is None or weights is None:
                    raise KeyError("Since integration is None, agent_data must have both weights and nodes fields.")
                if nodes.shape[1] < K2:
                    raise ValueError(f"The number of columns in the nodes field of agent_data must be at least {K2}.")
                if weights.shape[1] != 1:
                    raise ValueError("The weights field of agent_data must be one-dimensional.")

                # delete columns of nodes if there are too many
                if nodes.shape[1] > K2:
                    nodes = nodes[:, :K2]

        # structure agents
        return super().__new__(cls, {
            'market_ids': (market_ids, np.object),
            'weights': (weights, options.dtype),
            'nodes': (nodes, options.dtype),
            (tuple(demographics_formulations), 'demographics'): (demographics, options.dtype)
        })


class Economy(object):
    """An economy, which is initialized with product and agent data."""

    def __init__(self, product_formulations, agent_formulation, products, agents):
        """Store information about formulations and data."""

        # store formulations and data
        self.product_formulations = product_formulations
        self.agent_formulation = agent_formulation
        self.products = products
        self.agents = agents

        # identify unique markets and firms
        self.unique_market_ids = np.unique(self.products.market_ids).flatten()

        # count dimensions
        self.N = self.products.shape[0]
        self.T = self.unique_market_ids.size
        self.K1 = self.products.X1.shape[1]
        self.K2 = self.products.X2.shape[1]
        self.K3 = self.products.X3.shape[1]
        self.D = self.agents.demographics.shape[1]
        self.MD = self.products.ZD.shape[1]
        self.MS = self.products.ZS.shape[1]
        self.ED = self.products.demand_ids.shape[1]
        self.ES = self.products.supply_ids.shape[1]

        # identify column formulations
        self._X1_formulations = self.products.dtype.fields['X1'][2]
        self._X2_formulations = self.products.dtype.fields['X2'][2]
        self._X3_formulations = self.products.dtype.fields['X3'][2]
        self._demographics_formulations = self.agents.dtype.fields['demographics'][2]

    def __str__(self):
        """Format economy information as a string."""

        # associate dimensions and formulations with names
        dimension_mapping = collections.OrderedDict([
            ("Products (N)", self.N),
            ("Markets (T)", self.T),
            ("Linear Characteristics (K1)", self.K1),
            ("Nonlinear Characteristics (K2)", self.K2),
            ("Cost Characteristics (K3)", self.K3),
            ("Demographics (D)", self.D),
            ("Demand Instruments (MD)", self.MD),
            ("Supply Instruments (MS)", self.MS),
            ("Demand FEs (ED)", self.ED),
            ("Supply FEs (ES)", self.ES)
        ])
        formulation_mapping = collections.OrderedDict([
            ("Linear Characteristics (X1)", self._X1_formulations),
            ("Nonlinear Characteristics (X2)", self._X2_formulations),
            ("Cost Characteristics (X3)", self._X3_formulations),
            ("Demographics (d)", self._demographics_formulations)
        ])

        # build a dimensions section
        dimension_widths = [max(len(n), len(str(d))) for n, d in dimension_mapping.items()]
        dimension_formatter = output.table_formatter(dimension_widths)
        dimension_section = [
            "Dimensions:",
            dimension_formatter.line(),
            dimension_formatter(dimension_mapping.keys(), underline=True),
            dimension_formatter(dimension_mapping.values()),
            dimension_formatter.line()
        ]

        # build a formulations section
        formulation_header = ["Matrix Columns:"]
        formulation_widths = [max(len(formulation_header[0]), max(map(len, formulation_mapping.keys())))]
        for index in range(max(map(len, formulation_mapping.values()))):
            formulation_header.append(index)
            column_width = 5
            for formulation in formulation_mapping.values():
                if len(formulation) > index:
                    column_width = max(column_width, len(str(formulation[index])))
            formulation_widths.append(column_width)
        formulation_formatter = output.table_formatter(formulation_widths)
        formulation_section = [
            "Formulations:",
            formulation_formatter.line(),
            formulation_formatter(formulation_header, underline=True)
        ]
        for name, formulations in formulation_mapping.items():
            if formulations:
                formulation_section.append(formulation_formatter([name] + list(map(str, formulations))))
        formulation_section.append(formulation_formatter.line())

        # combine the sections into one string
        return "\n\n".join("\n".join(s) for s in [dimension_section, formulation_section])

    def __repr__(self):
        """Defer to the string representation."""
        return str(self)


class Market(object):
    """A single market in an economy."""

    def __init__(self, economy, t, sigma, pi, beta=None, delta=None):
        """Store or compute information about formulations, data, parameters, and utility."""

        # store data
        self.products = np.lib.recfunctions.rec_drop_fields(
            economy.products[economy.products.market_ids.flat == t],
            self.get_unneeded_product_fields(set(economy.products.dtype.names))
        )
        self.agents = np.lib.recfunctions.rec_drop_fields(
            economy.agents[economy.agents.market_ids.flat == t], 'market_ids'
        )

        # count dimensions
        self.J = self.products.shape[0]
        self.I = self.agents.shape[0]
        self.K1 = economy.K1
        self.K2 = economy.K2
        self.K3 = economy.K3
        self.D = economy.D

        # identify column formulations
        self._X1_formulations = economy._X1_formulations
        self._X2_formulations = economy._X2_formulations
        self._X3_formulations = economy._X3_formulations
        self._demographics_formulations = economy._demographics_formulations

        # store parameters
        self.sigma = sigma
        self.pi = pi
        self.beta = beta

        # store delta and compute mu
        self.delta = None if delta is None else delta[economy.products.market_ids.flat == t]
        self.mu = self.compute_mu()

    def get_unneeded_product_fields(self, fields):
        """Collect fields that will be dropped from product data."""
        return fields & {'market_ids', 'X1', 'X3', 'ZD', 'ZS', 'demand_ids', 'supply_ids'}

    def get_ownership_matrix(self, firms_index=0):
        """Get a pre-computed ownership matrix or build one. By default, use unchanged firm IDs."""

        # get a pre-computed ownership matrix
        if self.products.ownership.shape[1] > 0:
            offset = firms_index * self.products.ownership.shape[1] // self.products.firm_ids.shape[1]
            return self.products.ownership[:, offset:offset + self.J]

        # build a standard ownership matrix
        tiled_ids = np.tile(self.products.firm_ids[:, [firms_index]], self.J)
        return np.where(tiled_ids == tiled_ids.T, 1, 0)

    def compute_random_coefficients(self):
        """Compute the random coefficients by weighting agent characteristics with nonlinear parameters."""
        coefficients = self.sigma @ self.agents.nodes.T
        if self.D > 0:
            coefficients += self.pi @ self.agents.demographics.T
        return coefficients

    def compute_mu(self, X2=None):
        """Compute mu. By default, use the unchanged X2."""
        if X2 is None:
            X2 = self.products.X2
        return X2 @ self.compute_random_coefficients()

    def update_delta_with_variable(self, name, variable):
        """Update delta to reflect a changed variable by adding any parameter-weighted characteristic changes to X1."""

        # if the variable does not contribute to X1, delta remains unchanged
        if not any(name in f.names for f in self._X1_formulations):
            return self.delta

        # if the variable does contribute to X1, delta may change
        delta = self.delta.copy()
        products = self.products.copy()
        products[name] = variable
        for index, formulation in enumerate(self._X1_formulations):
            if name in formulation.names:
                delta += self.beta[index] * (formulation.evaluate(products) - self.products[name])
        return delta

    def update_mu_with_variable(self, name, variable):
        """Update mu to reflect a changed variable by re-computing mu under the changed X2."""

        # if the variable does not contribute to X2, mu remains unchanged
        if not any(name in f.names for f in self._X2_formulations):
            return self.mu

        # if the variable does contribute to X2, mu may change
        X2 = self.products.X2.copy()
        products = self.products.copy()
        products[name] = variable
        for index, formulation in enumerate(self._X2_formulations):
            if name in formulation.names:
                X2[:, [index]] = formulation.evaluate(products)
        return self.compute_mu(X2)

    def compute_probabilities(self, delta=None, mu=None, linear=True, eliminate_product=None):
        """Compute choice probabilities. By default, use unchanged delta and mu values. If linear is False, delta and mu
        must be be specified and already be exponentiated. If eliminate_product is specified, eliminate the product
        associated with the specified index from the choice set.
        """
        if delta is None:
            delta = self.delta
        if mu is None:
            mu = self.mu
        if self.K2 == 0:
            mu = 0 if linear else 1
        exponentiated_utilities = np.exp(delta + mu) if linear else delta * mu
        if eliminate_product is not None:
            exponentiated_utilities[eliminate_product] = 0
        return exponentiated_utilities / (1 + exponentiated_utilities.sum(axis=0))

    def compute_X1_by_variable_derivatives(self, name, variable=None):
        """Compute derivatives of X1 with respect to a variable. By default, use unchanged variable values."""
        if variable is None:
            products = self.products
        else:
            products = self.products.copy()
            products[name] = variable
        derivatives = np.zeros((self.J, self.K1), options.dtype)
        for index, formulation in enumerate(self._X1_formulations):
            if name in formulation.names:
                derivatives[:, [index]] = formulation.evaluate_derivative(name, products)
        return derivatives

    def compute_X2_by_variable_derivatives(self, name, variable=None):
        """Compute derivatives of X2 with respect to a variable. By default, use unchanged variable values."""
        if variable is None:
            products = self.products
        else:
            products = self.products.copy()
            products[name] = variable
        derivatives = np.zeros((self.J, self.K2), options.dtype)
        for index, formulation in enumerate(self._X2_formulations):
            if name in formulation.names:
                derivatives[:, [index]] = formulation.evaluate_derivative(name, products)
        return derivatives

    def compute_utility_by_variable_derivatives(self, name, variable=None):
        """Compute derivatives of utility with respect to a variable. By default, use unchanged variable values."""
        derivatives = np.tile(self.compute_X1_by_variable_derivatives(name, variable) @ self.beta, self.I)
        if self.K2 > 0:
            derivatives += self.compute_X2_by_variable_derivatives(name, variable) @ self.compute_random_coefficients()
        return derivatives

    def compute_shares_by_variable_jacobian(self, derivatives, probabilities=None):
        """Use derivatives of utility with respect to a variable to compute the Jacobian of market shares with respect
        to the same variable. By default, compute unchanged choice probabilities.
        """
        if probabilities is None:
            probabilities = self.compute_probabilities()
        V = probabilities * derivatives
        capital_lambda = np.diagflat(V @ self.agents.weights)
        capital_gamma = V @ np.diagflat(self.agents.weights) @ probabilities.T
        return capital_lambda - capital_gamma

    def compute_eta(self, ownership_matrix=None, derivatives=None, prices=None):
        """Compute the markup term in the BLP-markup equation. By default, get an unchanged ownership matrix, compute
        derivatives of utilities with respect to prices, and use unchanged prices.
        """
        if ownership_matrix is None:
            ownership_matrix = self.get_ownership_matrix()
        if derivatives is None:
            derivatives = self.compute_utility_by_variable_derivatives('prices')
        if prices is None:
            probabilities = self.compute_probabilities()
            shares = self.products.shares
        else:
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
            probabilities = self.compute_probabilities(delta, mu)
            shares = probabilities @ self.agents.weights
        shares_jacobian = self.compute_shares_by_variable_jacobian(derivatives, probabilities)
        try:
            return -scipy.linalg.solve(ownership_matrix * shares_jacobian, shares)
        except ValueError:
            return np.full_like(shares, np.nan)

    def compute_zeta(self, ownership_matrix=None, derivatives=None, prices=None, costs=None):
        """Compute the markup term in the zeta-markup equation. By default, get an unchanged ownership matrix, compute
        derivatives of utilities with respect to prices, use unchanged prices, and compute marginal costs.
        """
        if ownership_matrix is None:
            ownership_matrix = self.get_ownership_matrix()
        if derivatives is None:
            derivatives = self.compute_utility_by_variable_derivatives('prices')
        if prices is None:
            probabilities = self.compute_probabilities()
            shares = self.products.shares
        else:
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
            probabilities = self.compute_probabilities(delta, mu)
            shares = probabilities @ self.agents.weights
        if costs is None:
            costs = prices - self.compute_eta(ownership_matrix, derivatives, prices)
        V = probabilities * derivatives
        capital_lambda_inverse = np.diagflat(1 / (V @ self.agents.weights))
        capital_gamma = V @ np.diagflat(self.agents.weights) @ probabilities.T
        tilde_capital_omega = capital_lambda_inverse @ (ownership_matrix * capital_gamma).T
        return tilde_capital_omega @ (prices - costs) - capital_lambda_inverse @ shares

    def compute_bertrand_nash_prices(self, iteration, firms_index=0, prices=None, costs=None):
        """Compute Bertrand-Nash prices by iterating over the zeta-markup equation. By default, use unchanged firm IDs,
        use unchanged prices as initial values, and compute marginal costs.
        """
        if prices is None:
            prices = self.products.prices
        if costs is None:
            costs = self.products.prices - self.compute_eta()

        # derivatives of utilities with respect to prices change during iteration only if second derivatives are nonzero
        if any(f.differentiate('prices', order=2) != 0 for f in self._X1_formulations + self._X2_formulations):
            get_derivatives = lambda p: self.compute_utility_by_variable_derivatives('prices', p)
        else:
            derivatives = self.compute_utility_by_variable_derivatives('prices')
            get_derivatives = lambda _: derivatives

        # solve the fixed point problem
        ownership_matrix = self.get_ownership_matrix(firms_index)
        contraction = lambda p: costs + self.compute_zeta(ownership_matrix, get_derivatives(p), p, costs)
        return iteration._iterate(prices, contraction)


class NonlinearParameter(object):
    """Information about a single nonlinear parameter."""

    def __init__(self, location, bounds):
        """Store the information and determine whether the parameter is fixed or unfixed."""
        self.location = location
        self.lb = bounds[0][location]
        self.ub = bounds[1][location]
        self.value = self.lb if self.lb == self.ub else None


class SigmaParameter(NonlinearParameter):
    """Information about a single parameter in sigma."""

    def get_characteristics(self, products, agents):
        """Get the product and agents characteristics associated with the parameter."""
        return products.X2[:, [self.location[0]]], agents.nodes[:, [self.location[1]]]


class PiParameter(NonlinearParameter):
    """Information about a single parameter in pi."""

    def get_characteristics(self, products, agents):
        """Get the product and agents characteristics associated with the parameter."""
        return products.X2[:, [self.location[0]]], agents.demographics[:, [self.location[1]]]


class NonlinearParameters(object):
    """Information about sigma and pi."""

    def __init__(self, economy, sigma=None, pi=None, sigma_bounds=None, pi_bounds=None, supports_bounds=True):
        """Store information about fixed (equal bounds) and unfixed (unequal bounds) parameters in sigma and pi."""

        # store labels
        self.X2_labels = list(map(str, economy._X2_formulations))
        self.demographics_labels = list(map(str, economy._demographics_formulations))

        # store the upper triangle of sigma
        self.sigma = np.full((economy.K2, economy.K2), np.nan, options.dtype)
        if sigma is not None:
            self.sigma = np.c_[np.asarray(sigma, options.dtype)]
            if economy.K2 == 0 and sigma.size > 0:
                raise ValueError("X2 was not formulated, so sigma should be None.")
            if self.sigma.shape != (economy.K2, economy.K2):
                raise ValueError(f"sigma must be a {economy.K2} by {economy.K2} matrix.")
            self.sigma[np.tril_indices(economy.K2, -1)] = 0

        # store pi
        self.pi = np.full((economy.K2, 0), np.nan, options.dtype)
        if pi is not None:
            self.pi = np.c_[np.asarray(pi, options.dtype)]
            if economy.D == 0 and self.pi.size > 0:
                raise ValueError("Demographics were not formulated, so pi should be None.")
            if pi is None or self.pi.shape != (economy.K2, economy.D):
                raise ValueError(f"pi must be a {economy.K2} by {economy.D} matrix.")

        # construct or validate sigma bounds
        self.sigma_bounds = (
            np.full_like(self.sigma, -np.inf, options.dtype), np.full_like(self.sigma, +np.inf, options.dtype)
        )
        if supports_bounds:
            np.fill_diagonal(self.sigma_bounds[0], 0)
        if economy.K2 > 0 and sigma_bounds is not None and supports_bounds:
            if len(sigma_bounds) != 2:
                raise ValueError("sigma_bounds must be a tuple of the form (lb, ub).")
            self.sigma_bounds = [np.c_[np.asarray(b, options.dtype).copy()] for b in sigma_bounds]
            for bounds_index, bounds in enumerate(self.sigma_bounds):
                bounds[np.isnan(bounds)] = -np.inf if bounds_index == 0 else +np.inf
                if bounds.shape != self.sigma.shape:
                    raise ValueError(f"sigma_bounds[{bounds_index}] must have the same shape as sigma.")
            if ((self.sigma < self.sigma_bounds[0]) | (self.sigma > self.sigma_bounds[1])).any():
                raise ValueError("sigma must be within its bounds.")

        # construct or validate pi bounds
        self.pi_bounds = (
            np.full_like(self.pi, -np.inf, options.dtype), np.full_like(self.pi, +np.inf, options.dtype)
        )
        if economy.D > 0 and pi_bounds is not None and supports_bounds:
            if len(pi_bounds) != 2:
                raise ValueError("pi_bounds must be a tuple of the form (lb, ub).")
            self.pi_bounds = [np.c_[np.asarray(b, options.dtype).copy()] for b in pi_bounds]
            for bounds_index, bounds in enumerate(self.pi_bounds):
                bounds[np.isnan(bounds)] = -np.inf if bounds_index == 0 else +np.inf
                if bounds.shape != self.pi.shape:
                    raise ValueError(f"pi_bounds[{bounds_index}] must have the same shape as pi.")
            if ((self.pi < self.pi_bounds[0]) | (self.pi > self.pi_bounds[1])).any():
                raise ValueError("pi must be within its bounds.")

        # set upper and lower bounds to zero for parameters that are fixed at zero
        self.sigma_bounds[0][np.where(self.sigma == 0)] = self.sigma_bounds[1][np.where(self.sigma == 0)] = 0
        self.pi_bounds[0][np.where(self.pi == 0)] = self.pi_bounds[1][np.where(self.pi == 0)] = 0

        # store information about individual elements in sigma and pi
        self.fixed = []
        self.unfixed = []

        # store information for the upper triangle of sigma
        for location in zip(*np.triu_indices_from(self.sigma)):
            parameter = SigmaParameter(location, self.sigma_bounds)
            if parameter.value is None:
                self.unfixed.append(parameter)
            else:
                self.fixed.append(parameter)

        # store information for pi
        for location in np.ndindex(self.pi.shape):
            parameter = PiParameter(location, self.pi_bounds)
            if parameter.value is None:
                self.unfixed.append(parameter)
            else:
                self.fixed.append(parameter)

        # count the number of unfixed parameters
        self.P = len(self.unfixed)

    def format(self):
        """Format the initial sigma and pi as a string."""
        return self.format_matrices(self.sigma, self.pi)

    def format_lower_bounds(self):
        """Format lower sigma and pi bounds as a string."""
        return self.format_matrices(self.sigma_bounds[0], self.pi_bounds[0])

    def format_upper_bounds(self):
        """Format upper sigma and pi bounds as a string."""
        return self.format_matrices(self.sigma_bounds[1], self.pi_bounds[1])

    def format_estimates(self, sigma, pi, sigma_se, pi_se):
        """Format sigma and pi estimates along with their standard errors as a string."""
        return self.format_matrices(sigma, pi, sigma_se, pi_se)

    def format_matrices(self, sigma_like, pi_like, sigma_se_like=None, pi_se_like=None):
        """Format matrices (and optional standard errors) of the same size as sigma and pi as a string."""

        # there is nothing to format if all characteristics are linear
        if not self.X2_labels:
            return ""

        # construct the table header and formatter
        line_indices = {}
        header = ["Sigma:"] + self.X2_labels
        widths = [max(map(len, header))] + [max(len(k), options.digits + 8) for k in header[1:]]
        if self.demographics_labels:
            line_indices = {len(widths) - 1}
            header.extend(["Pi:"] + self.demographics_labels)
            widths.extend([widths[0]] + [max(len(k), options.digits + 8) for k in header[len(widths) + 1:]])
        formatter = output.table_formatter(widths, line_indices)

        # build the top of the table
        lines = [formatter(header, underline=True)]

        # construct the rows containing parameter information
        for row_index, row_label in enumerate(self.X2_labels):
            # the row is a label, blanks for sigma's lower triangle, sigma values, the label again, and pi values
            values_row = [row_label] + [""] * row_index
            for column_index in range(row_index, sigma_like.shape[1]):
                values_row.append(output.format_number(sigma_like[row_index, column_index]))
            if pi_like.shape[1] > 0:
                values_row.append(row_label)
                for column_index in range(pi_like.shape[1]):
                    values_row.append(output.format_number(pi_like[row_index, column_index]))
            lines.append(formatter(values_row))

            # construct a row of standard errors for unfixed parameters
            if sigma_se_like is not None and pi_se_like is not None:
                # determine which columns in this row correspond to unfixed parameters
                sigma_indices = set()
                pi_indices = set()
                for parameter in self.unfixed:
                    if parameter.location[0] == row_index:
                        if isinstance(parameter, SigmaParameter):
                            sigma_indices.add(parameter.location[1])
                        else:
                            pi_indices.add(parameter.location[1])

                # construct a row similar to the values row without row labels and optionally with standard error
                se_row = [""] * (1 + row_index)
                for column_index in range(row_index, sigma_se_like.shape[1]):
                    se = sigma_se_like[row_index, column_index]
                    se_row.append(output.format_se(se) if column_index in sigma_indices else "")
                if pi_se_like.shape[1] > 0:
                    se_row.append("")
                    for column_index in range(pi_se_like.shape[1]):
                        se = pi_se_like[row_index, column_index]
                        se_row.append(output.format_se(se) if column_index in pi_indices else "")

                # format the row of values and add an additional blank line if there is another row of values
                lines.append(formatter(se_row))
                if row_index < sigma_like.shape[1] - 1:
                    lines.append(formatter.blank())

        # wrap the table in borders and combine the lines into one string
        return "\n".join([formatter.line()] + lines + [formatter.line()])

    def compress(self):
        """Compress the initial sigma and pi into theta."""
        return np.r_[
            self.sigma[list(zip(*[p.location for p in self.unfixed if isinstance(p, SigmaParameter)]))].ravel(),
            self.pi[list(zip(*[p.location for p in self.unfixed if isinstance(p, PiParameter)]))].ravel()
        ]

    def expand(self, theta_like, nullify=False):
        """Recover matrices of the same size as sigma and pi from a vector of the same size as theta. By default,
        fill elements corresponding to fixed parameters with their fixed values.
        """
        sigma_like = np.full_like(self.sigma, np.nan)
        pi_like = np.full_like(self.pi, np.nan)

        # set values for elements that correspond to unfixed parameters
        for parameter, value in zip(self.unfixed, theta_like):
            if isinstance(parameter, SigmaParameter):
                sigma_like[parameter.location] = value
            else:
                pi_like[parameter.location] = value

        # set values for elements that correspond to fixed parameters
        if not nullify:
            sigma_like[np.tril_indices_from(sigma_like, -1)] = 0
            for parameter in self.fixed:
                if isinstance(parameter, SigmaParameter):
                    sigma_like[parameter.location] = parameter.value
                else:
                    pi_like[parameter.location] = parameter.value

        # return the expanded matrices
        return sigma_like, pi_like


class LinearParameters(object):
    """Information about beta and gamma."""

    def __init__(self, economy, beta, gamma=None):
        """Store information about parameters in beta and gamma."""

        # store labels
        self.X1_labels = list(map(str, economy._X1_formulations))
        self.X3_labels = list(map(str, economy._X3_formulations))

        # store beta
        self.beta = np.c_[np.asarray(beta, options.dtype)]
        if self.beta.shape != (economy.K1, 1):
            raise ValueError(f"beta must be a {economy.K1}-vector.")

        # store gamma
        self.gamma = np.full((economy.K3, 0), np.nan, options.dtype)
        if gamma is not None:
            self.gamma = np.c_[np.asarray(gamma, options.dtype)]
            if economy.K3 == 0 and self.gamma.size > 0:
                raise ValueError("X2 was not formulated, so gamma should be None.")
            if self.gamma.shape != (economy.K3, 1):
                raise ValueError(f"gamma must be a {economy.K3}-vector.")

    def format(self):
        """Format the initial beta and gamma as a string."""
        return self.format_vectors(self.beta, self.gamma)

    def format_estimates(self, beta, gamma, beta_se, gamma_se):
        """Format beta and gamma estimates along with their standard errors as a string."""
        return self.format_vectors(beta, gamma, beta_se, gamma_se)

    def format_vectors(self, beta_like, gamma_like, beta_se_like=None, gamma_se_like=None):
        """Format matrices (and optional standard errors) of the same size as beta and gamma as a string."""
        lines = []

        # build the header for beta
        beta_header = ["Beta:"] + self.X1_labels
        beta_widths = [len(beta_header[0])] + [max(len(k), options.digits + 8) for k in beta_header[1:]]

        # build the header for gamma
        gamma_header = None
        gamma_widths = []
        if self.X3_labels:
            gamma_header = ["Gamma:"] + self.X3_labels
            gamma_widths = [len(gamma_header[0])] + [max(len(k), options.digits + 8) for k in gamma_header[1:]]

        # build the table formatter
        widths = [max(w) for w in itertools.zip_longest(beta_widths, gamma_widths, fillvalue=0)]
        formatter = output.table_formatter(widths)

        # build the table
        lines.extend([
            formatter.line(),
            formatter(beta_header, underline=True),
            formatter([""] + [output.format_number(x) for x in beta_like])
        ])
        if beta_se_like is not None:
            lines.append(formatter([""] + [output.format_se(x) for x in beta_se_like]))
        if gamma_like.size > 0:
            lines.extend([
                formatter.line(),
                formatter(gamma_header, underline=True),
                formatter([""] + [output.format_number(x) for x in gamma_like])
            ])
            if gamma_se_like is not None:
                lines.append(formatter([""] + [output.format_se(x) for x in gamma_se_like]))
        lines.append(formatter.line())

        # combine the lines into one string
        return "\n".join(lines)
