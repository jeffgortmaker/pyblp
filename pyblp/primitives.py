"""Primitive structures that constitute the foundation of the BLP model."""

import time
import itertools
import collections

import numpy as np
import numpy.lib.recfunctions

from . import options, exceptions
from .configurations import Formulation, Integration
from .utilities import solve, output, extract_matrix, Matrices, Groups


class Products(Matrices):
    r"""Structured product data, which contains the following fields:

        - **market_ids** : (`object`) - IDs that associate products with markets.

        - **firm_ids** : (`object`) - IDs that associate products with firms. Any columns after the first represent
          changes such as mergers.

        - **demand_ids** : (`object`) - IDs used to create demand-side fixed effects.

        - **supply_ids** : (`object`) - IDs used to create supply-side fixed effects.

        - **nesting_ids** : (`object`) - IDs that associate products with nesting groups.

        - **clustering_ids** (`object`) - IDs used to compute clustered standard errors.

        - **ownership** : (`object`) - Stacked :math:`J_t \times J_t` ownership matrices, :math:`O`, for each market
          :math:`t`. Each stack is associated with a `firm_ids` column.

        - **shares** : (`numeric`) - Market shares, :math:`s`.

        - **ZD** : (`numeric`) - Demand-side instruments, :math:`Z_D`.

        - **ZS** : (`numeric`) - Supply-side instruments, :math:`Z_S`.

        - **X1** : (`numeric`) - Linear product characteristics, :math:`X_1`.

        - **X2** : (`numeric`) - Nonlinear product characteristics, :math:`X_2`.

        - **X3** : (`numeric`) - Cost product characteristics, :math:`X_3`.

        - **prices** : (`numeric`) - Product prices, :math:`p`.

    Any additional fields are the variables underlying `X1`, `X2`, and `X3`.

    """

    def __new__(cls, product_formulations, product_data):
        """Structure product data."""

        # validate the formulations
        if not all(isinstance(f, Formulation) or f is None for f in product_formulations):
            raise TypeError("Each formulation in product_formulations must be a Formulation instance or None.")
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

        # load fixed effect IDs
        demand_ids = supply_ids = None
        if product_formulations[0]._absorbed_terms:
            demand_ids = product_formulations[0]._build_ids(product_data)
        if product_formulations[2] is not None and product_formulations[2]._absorbed_terms:
            supply_ids = product_formulations[2]._build_ids(product_data)

        # load other IDs
        market_ids = extract_matrix(product_data, 'market_ids')
        firm_ids = extract_matrix(product_data, 'firm_ids')
        nesting_ids = extract_matrix(product_data, 'nesting_ids')
        clustering_ids = extract_matrix(product_data, 'clustering_ids')
        if market_ids is None:
            raise KeyError("product_data must have a market_ids field.")
        if market_ids.shape[1] > 1:
            raise ValueError("The market_ids field of product_data must be one-dimensional.")
        if firm_ids is None and X3 is not None:
            raise KeyError("product_data must have a firm_ids field when X3 is formulated.")
        if nesting_ids is not None and nesting_ids.shape[1] > 1:
            raise ValueError("The nesting_ids field of product_data must be one-dimensional.")
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
            'nesting_ids': (nesting_ids, np.object),
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
        """Store information about formulations and data before absorbing any fixed effects."""

        # store formulations and data
        self.product_formulations = product_formulations
        self.agent_formulation = agent_formulation
        self.products = products
        self.agents = agents

        # identify unique markets and nests
        self.unique_market_ids = np.unique(self.products.market_ids).flatten()
        self.unique_nesting_ids = np.unique(self.products.nesting_ids).flatten()

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
        self.H = self.unique_nesting_ids.size

        # identify column formulations
        self._X1_formulations = self.products.dtype.fields['X1'][2]
        self._X2_formulations = self.products.dtype.fields['X2'][2]
        self._X3_formulations = self.products.dtype.fields['X3'][2]
        self._demographics_formulations = self.agents.dtype.fields['demographics'][2]

        # absorb any demand-side fixed effects
        self._absorb_demand_ids = None
        if self.ED > 0:
            start_time = time.time()
            output("")
            output("Absorbing demand-side fixed effects ...")
            self._absorb_demand_ids = product_formulations[0]._build_absorb(self.products.demand_ids)
            self.products.X1, X1_errors = self._absorb_demand_ids(self.products.X1)
            self.products.ZD, ZD_errors = self._absorb_demand_ids(self.products.ZD)
            if X1_errors | ZD_errors:
                raise exceptions.MultipleErrors(X1_errors | ZD_errors)
            end_time = time.time()
            output(f"Absorbed demand-side fixed effects after {output.format_seconds(end_time - start_time)}.")

        # absorb any supply-side fixed effects
        self._absorb_supply_ids = None
        if self.ES > 0:
            start_time = time.time()
            output("")
            output("Absorbing supply-side fixed effects ...")
            self._absorb_supply_ids = product_formulations[2]._build_absorb(self.products.supply_ids)
            self.products.X3, X3_errors = self._absorb_supply_ids(self.products.X3)
            self.products.ZS, ZS_errors = self._absorb_supply_ids(self.products.ZS)
            if X3_errors | ZS_errors:
                raise exceptions.MultipleErrors(X3_errors | ZS_errors)
            end_time = time.time()
            output(f"Absorbed supply-side fixed effects after {output.format_seconds(end_time - start_time)}.")

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
            ("Supply FEs (ES)", self.ES),
            ("Nesting Groups (H)", self.H)
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

    def __init__(self, economy, t, sigma, pi, rho, beta=None, delta=None):
        """Store or compute information about formulations, data, parameters, and utility."""

        # store data
        self.products = np.lib.recfunctions.rec_drop_fields(
            economy.products[economy.products.market_ids.flat == t],
            self.get_unneeded_product_fields(set(economy.products.dtype.names))
        )
        self.agents = np.lib.recfunctions.rec_drop_fields(
            economy.agents[economy.agents.market_ids.flat == t], 'market_ids'
        )

        # create nesting groups
        self.groups = Groups(self.products.nesting_ids)

        # count dimensions
        self.J = self.products.shape[0]
        self.I = self.agents.shape[0]
        self.K1 = economy.K1
        self.K2 = economy.K2
        self.K3 = economy.K3
        self.D = economy.D
        self.H = self.groups.unique.size

        # identify column formulations
        self._X1_formulations = economy._X1_formulations
        self._X2_formulations = economy._X2_formulations
        self._X3_formulations = economy._X3_formulations
        self._demographics_formulations = economy._demographics_formulations

        # store parameters (expand rho to all groups and all products)
        self.sigma = sigma
        self.pi = pi
        self.beta = beta
        if rho.size == 1:
            self.group_rho = np.full((self.H, 1), rho)
            self.rho = np.full((self.J, 1), rho)
        else:
            self.group_rho = rho[np.searchsorted(economy.unique_nesting_ids, self.groups.unique)]
            self.rho = self.groups.expand(self.group_rho)

        # store delta and compute mu
        self.delta = None if delta is None else delta[economy.products.market_ids.flat == t]
        self.mu = self.compute_mu()

    def get_unneeded_product_fields(self, fields):
        """Collect fields that will be dropped from product data."""
        return fields & {'market_ids', 'X1', 'X3', 'ZD', 'ZS', 'demand_ids', 'supply_ids'}

    def get_membership_matrix(self):
        """Build a membership matrix from nesting IDs."""
        tiled_ids = np.tile(self.products.nesting_ids, self.J)
        return np.where(tiled_ids == tiled_ids.T, 1, 0)

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
        assert self.beta is not None and self.delta is not None

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

    def compute_X1_derivatives(self, name, variable=None):
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

    def compute_X2_derivatives(self, name, variable=None):
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

    def compute_utility_derivatives(self, name, variable=None):
        """Compute derivatives of utility with respect to a variable. By default, use unchanged variable values."""
        assert self.beta is not None
        derivatives = np.tile(self.compute_X1_derivatives(name, variable) @ self.beta, self.I)
        if self.K2 > 0:
            derivatives += self.compute_X2_derivatives(name, variable) @ self.compute_random_coefficients()
        return derivatives

    def compute_probabilities(self, delta=None, mu=None, linear=True, eliminate_product=None, keep_conditionals=False):
        """Compute choice probabilities. By default, use unchanged delta and mu values. If linear is False, delta and mu
        must be specified and already be exponentiated. If eliminate_product is specified, eliminate the product
        associated with the specified index from the choice set. If keep_conditionals is True, return a tuple in which
        if there is nesting, the second element are conditional probabilities given that an alternative in a nest is
        chosen.
        """
        if delta is None:
            assert self.delta is not None
            delta = self.delta
        if mu is None:
            mu = self.mu
        if self.K2 == 0:
            mu = int(not linear)

        # compute exponentiated utilities, optionally eliminating a product from the choice set
        exp_utilities = np.exp(delta + mu) if linear else np.array(delta * mu)
        if eliminate_product is not None:
            exp_utilities[eliminate_product] = 0

        # compute standard or nested probabilities
        if self.H == 0:
            conditionals = None
            probabilities = exp_utilities / (1 + exp_utilities.sum(axis=0))
        else:
            exp_weighted_utilities = exp_utilities**(1 / (1 - self.rho))
            exp_inclusives = self.groups.sum(exp_weighted_utilities)
            exp_weighted_inclusives = exp_inclusives**(1 - self.group_rho)
            conditionals = exp_weighted_utilities / self.groups.expand(exp_inclusives)
            marginals = exp_weighted_inclusives / (1 + exp_weighted_inclusives.sum(axis=0))
            probabilities = conditionals * self.groups.expand(marginals)

        # return either probabilities and their conditional counterparts or just probabilities
        return (probabilities, conditionals) if keep_conditionals else probabilities

    def compute_capital_lamda(self, value_derivatives):
        """Use derivatives of aggregate inclusive values with respect to a variable to compute the diagonal capital
        lambda matrix used to decompose markups.
        """
        capital_lamda = np.diagflat(value_derivatives @ self.agents.weights)
        if self.H > 0:
            capital_lamda /= 1 - self.rho
        return capital_lamda

    def compute_capital_gamma(self, value_derivatives, probabilities, conditionals):
        """Use derivatives of aggregate inclusive values with respect to a variable and choice probabilities to compute
        the dense capital gamma matrix used to decompose markups.
        """
        diagonal_weights = np.diagflat(self.agents.weights)
        capital_gamma = probabilities @ diagonal_weights @ value_derivatives.T
        if self.H > 0:
            membership = self.get_membership_matrix()
            capital_gamma += self.rho / (1 - self.rho) * membership * (
                conditionals @ diagonal_weights @ value_derivatives.T
            )
        return capital_gamma

    def compute_utility_derivatives_by_parameter_tangent(self, parameter, X1_derivatives, X2_derivatives, beta_tangent):
        """Use derivatives of X1 and X2 with respect to a variable and the tangent of beta with respect to a nonlinear
        parameter to compute the tangent with respect to the parameter of derivatives of utility with respect to the
        variable.
        """
        tangent = np.tile(X1_derivatives @ beta_tangent, self.I)
        if isinstance(parameter, RandomCoefficientParameter):
            v = parameter.get_agent_characteristic(self.agents)
            tangent += X2_derivatives[:, [parameter.location[0]]] @ v.T
        return tangent

    def compute_probabilities_by_parameter_tangent(self, parameter, probabilities, conditionals, delta=None, mu=None):
        """Use probabilities to compute their tangent with respect to a nonlinear parameter. By default, use unchanged
        delta and mu.
        """
        if delta is None:
            assert self.delta is not None
            delta = self.delta
        if mu is None:
            mu = self.mu

        # without nesting, compute only the tangent of probabilities with respect to the parameter
        if self.H == 0:
            assert isinstance(parameter, RandomCoefficientParameter)
            v = parameter.get_agent_characteristic(self.agents)
            x = parameter.get_product_characteristic(self.products)
            probabilities_tangent = probabilities * v.T * (x - x.T @ probabilities)
            return probabilities_tangent, None

        # marginal probabilities are needed to compute tangents with nesting
        marginals = self.groups.sum(probabilities)

        # compute the tangent of conditional and marginal probabilities with respect to the parameter
        if isinstance(parameter, RandomCoefficientParameter):
            v = parameter.get_agent_characteristic(self.agents)
            x = parameter.get_product_characteristic(self.products)

            # compute the tangent of conditional probabilities with respect to the parameter
            A = conditionals * x
            A_sums = self.groups.sum(A)
            conditionals_tangent = conditionals * v.T * (x - self.groups.expand(A_sums)) / (1 - self.rho)

            # compute the tangent of marginal probabilities with respect to the parameter
            B = marginals * A_sums * v.T
            marginals_tangent = B - marginals * B.sum(axis=0)
        else:
            assert isinstance(parameter, RhoParameter)
            group_associations = parameter.get_group_associations(self.groups)
            associations = self.groups.expand(group_associations)

            # utilities are needed to compute tangents with respect to rho
            weighted_utilities = (delta + mu) / (1 - self.rho)

            # compute the tangent of conditional probabilities with respect to the parameter
            A = conditionals * weighted_utilities / (1 - self.rho)
            A_sums = self.groups.sum(A)
            conditionals_tangent = associations * (A - conditionals * self.groups.expand(A_sums))

            # compute the tangent of marginal probabilities with respect to the parameter
            B = marginals * (A_sums * (1 - self.group_rho) - np.log(self.groups.sum(np.exp(weighted_utilities))))
            marginals_tangent = group_associations * B - marginals * (group_associations.T @ B)

        # compute the tangent of probabilities with respect to the parameter
        probabilities_tangent = (
            conditionals_tangent * self.groups.expand(marginals) +
            conditionals * self.groups.expand(marginals_tangent)
        )
        return probabilities_tangent, conditionals_tangent

    def compute_shares_by_variable_jacobian(self, utility_derivatives, probabilities=None, conditionals=None):
        """Use derivatives of utility with respect to a variable to compute the Jacobian of market shares with respect
        to the same variable. By default, compute unchanged choice probabilities.
        """
        if probabilities is None or conditionals is None:
            probabilities, conditionals = self.compute_probabilities(keep_conditionals=True)
        value_derivatives = probabilities * utility_derivatives
        capital_lamda = self.compute_capital_lamda(value_derivatives)
        capital_gamma = self.compute_capital_gamma(value_derivatives, probabilities, conditionals)
        return capital_lamda - capital_gamma

    def compute_eta(self, ownership_matrix=None, utility_derivatives=None, prices=None):
        """Compute the markup term in the BLP-markup equation. By default, get an unchanged ownership matrix, compute
        derivatives of utilities with respect to prices, and use unchanged prices.
        """
        if ownership_matrix is None:
            ownership_matrix = self.get_ownership_matrix()
        if utility_derivatives is None:
            utility_derivatives = self.compute_utility_derivatives('prices')
        if prices is None:
            probabilities, conditionals = self.compute_probabilities(keep_conditionals=True)
            shares = self.products.shares
        else:
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
            probabilities, conditionals = self.compute_probabilities(delta, mu)
            shares = probabilities @ self.agents.weights
        jacobian = self.compute_shares_by_variable_jacobian(utility_derivatives, probabilities, conditionals)
        return -solve(ownership_matrix * jacobian, shares)

    def compute_zeta(self, ownership_matrix=None, utility_derivatives=None, prices=None, costs=None):
        """Compute the markup term in the zeta-markup equation. By default, get an unchanged ownership matrix, compute
        derivatives of utilities with respect to prices, use unchanged prices, and compute marginal costs.
        """
        if ownership_matrix is None:
            ownership_matrix = self.get_ownership_matrix()
        if utility_derivatives is None:
            utility_derivatives = self.compute_utility_derivatives('prices')
        if prices is None:
            probabilities, conditionals = self.compute_probabilities(keep_conditionals=True)
            shares = self.products.shares
        else:
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
            probabilities, conditionals = self.compute_probabilities(delta, mu, keep_conditionals=True)
            shares = probabilities @ self.agents.weights
        if costs is None:
            costs = prices - self.compute_eta(ownership_matrix, utility_derivatives, prices)
        value_derivatives = probabilities * utility_derivatives
        capital_lamda_inverse = np.diag(1 / self.compute_capital_lamda(value_derivatives).diagonal())
        capital_gamma = self.compute_capital_gamma(value_derivatives, probabilities, conditionals)
        tilde_capital_omega = capital_lamda_inverse @ (ownership_matrix * capital_gamma).T
        return tilde_capital_omega @ (prices - costs) - capital_lamda_inverse @ shares

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
            get_derivatives = lambda p: self.compute_utility_derivatives('prices', p)
        else:
            derivatives = self.compute_utility_derivatives('prices')
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


class RandomCoefficientParameter(NonlinearParameter):
    """Information about a single nonlinear parameter in sigma or pi."""

    def get_product_characteristic(self, products):
        """Get the product characteristic associated with the parameter."""
        return products.X2[:, [self.location[0]]]

    def get_agent_characteristic(self, agents):
        """Get the agent characteristic associated with the parameter."""
        raise NotImplementedError


class SigmaParameter(RandomCoefficientParameter):
    """Information about a single parameter in sigma."""

    def get_agent_characteristic(self, agents):
        """Get the agent characteristic associated with the parameter."""
        return agents.nodes[:, [self.location[1]]]


class PiParameter(RandomCoefficientParameter):
    """Information about a single parameter in pi."""

    def get_agent_characteristic(self, agents):
        """Get the agent characteristic associated with the parameter."""
        return agents.demographics[:, [self.location[1]]]


class RhoParameter(NonlinearParameter):
    """Information about a single parameter in rho."""

    def __init__(self, location, bounds, single):
        """Store the information along with whether there is only a single parameter for all groups."""
        super().__init__(location, bounds)
        self.single = single

    def get_group_associations(self, groups):
        """Get an indicator for which groups are associated with the parameter."""
        associations = np.ones((groups.unique.size, 1), options.dtype)
        if not self.single:
            associations[:] = 0
            associations[self.location] = 1
        return associations


class NonlinearParameters(object):
    """Information about sigma and pi."""

    def __init__(self, economy, sigma=None, pi=None, rho=None, sigma_bounds=None, pi_bounds=None, rho_bounds=None,
                 bounded=False):
        """Store information about fixed (equal bounds) and unfixed (unequal bounds) elements of sigma, pi, and rho.
        Also verify that parameters have been chosen such that choice probability computation is unlikely to overflow.
        If unspecified, determine reasonable bounds as well.
        """
        self.errors = set()

        # store labels
        self.X2_labels = list(map(str, economy._X2_formulations))
        self.demographics_labels = list(map(str, economy._demographics_formulations))
        self.group_labels = list(map(str, economy.unique_nesting_ids))

        # store the upper triangle of sigma
        self.sigma = np.full((economy.K2, economy.K2), np.nan, options.dtype)
        if sigma is None and economy.K2 > 0:
            raise ValueError("X2 was formulated, so sigma should not be None.")
        elif sigma is not None:
            self.sigma = np.c_[np.asarray(sigma, options.dtype)]
            if economy.K2 == 0 and sigma.size > 0:
                raise ValueError("X2 was not formulated, so sigma should be None.")
            if self.sigma.shape != (economy.K2, economy.K2):
                raise ValueError(f"sigma must be a {economy.K2} by {economy.K2} matrix.")
            self.sigma[np.tril_indices(economy.K2, -1)] = 0

        # store pi
        self.pi = np.full((economy.K2, 0), np.nan, options.dtype)
        if pi is None and economy.D > 0:
            raise ValueError("Demographics were formulated, so pi should not be None.")
        elif pi is not None:
            self.pi = np.c_[np.asarray(pi, options.dtype)]
            if economy.D == 0 and self.pi.size > 0:
                raise ValueError("Demographics were not formulated, so pi should be None.")
            if self.pi.shape != (economy.K2, economy.D):
                raise ValueError(f"pi must be a {economy.K2} by {economy.D} matrix.")

        # store rho
        self.rho = np.full((0, 1), np.nan, options.dtype)
        if rho is None and economy.H > 0:
            raise ValueError("Nesting IDs were specified, so rho should not be None.")
        elif rho is not None:
            self.rho = np.c_[np.asarray(rho, options.dtype)]
            if economy.H == 0 and self.rho.size > 0:
                raise ValueError("Nesting IDs were not specified, so rho should be None.")
            if self.rho.shape not in {(1, 1), (economy.H, 1)}:
                raise ValueError(f"rho must be a scalar or a {economy.H}-vector.")

        # verify that parameters have been chosen such that choice probability computation is unlikely to overflow
        mu_max = np.log(np.finfo(options.dtype).max)
        mu_norm = self.compute_mu_norm(economy, self.sigma, self.pi)
        if mu_max < mu_norm:
            self.errors.add(exceptions.LargeInitialParametersError)

        # construct or validate sigma bounds
        self.sigma_bounds = (
            np.full_like(self.sigma, -np.inf, options.dtype), np.full_like(self.sigma, +np.inf, options.dtype)
        )
        if bounded:
            if economy.K2 == 0 or sigma_bounds is None:
                for location in zip(*self.sigma.nonzero()):
                    value = self.sigma[location]
                    other_sigma = self.sigma.copy()
                    other_sigma[location] = 0
                    other_mu_norm = self.compute_mu_norm(economy, other_sigma, self.pi)
                    products_norm = np.abs(economy.products.X2[:, location[0]]).max()
                    agents_norm = np.abs(economy.agents.nodes[:, location[1]]).max()
                    with np.errstate(divide='ignore'):
                        bound = self.normalize_bound(max(0, mu_max - other_mu_norm) / products_norm / agents_norm)
                    self.sigma_bounds[0][location] = min(value, -bound) if location[0] != location[1] else 0
                    self.sigma_bounds[1][location] = max(value, +bound)
            else:
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
        if bounded:
            if economy.D == 0 or pi_bounds is None:
                for location in zip(*self.pi.nonzero()):
                    value = self.pi[location]
                    other_pi = self.pi.copy()
                    other_pi[location] = 0
                    other_mu_norm = self.compute_mu_norm(economy, self.sigma, other_pi)
                    products_norm = np.abs(economy.products.X2[:, location[0]]).max()
                    agents_norm = np.abs(economy.agents.demographics[:, location[1]]).max()
                    with np.errstate(divide='ignore'):
                        bound = self.normalize_bound(max(0, mu_max - other_mu_norm) / products_norm / agents_norm)
                    self.pi_bounds[0][location] = min(value, -bound)
                    self.pi_bounds[1][location] = max(value, +bound)
            else:
                if len(pi_bounds) != 2:
                    raise ValueError("pi_bounds must be a tuple of the form (lb, ub).")
                self.pi_bounds = [np.c_[np.asarray(b, options.dtype).copy()] for b in pi_bounds]
                for bounds_index, bounds in enumerate(self.pi_bounds):
                    bounds[np.isnan(bounds)] = -np.inf if bounds_index == 0 else +np.inf
                    if bounds.shape != self.pi.shape:
                        raise ValueError(f"pi_bounds[{bounds_index}] must have the same shape as pi.")
                if ((self.pi < self.pi_bounds[0]) | (self.pi > self.pi_bounds[1])).any():
                    raise ValueError("pi must be within its bounds.")

        # construct or validate rho bounds
        self.rho_bounds = (
            np.full_like(self.rho, -np.inf, options.dtype), np.full_like(self.rho, +np.inf, options.dtype)
        )
        if bounded:
            if economy.H == 0 or rho_bounds is None:
                self.rho_bounds[0][:] = 0
                self.rho_bounds[1][:] = self.normalize_bound(1 - min(1, mu_norm / mu_max))
            else:
                if len(rho_bounds) != 2:
                    raise ValueError("rho_bounds must be a tuple of the form (lb, ub).")
                self.rho_bounds = [np.c_[np.asarray(b, options.dtype).copy()] for b in rho_bounds]
                for bounds_index, bounds in enumerate(self.rho_bounds):
                    bounds[np.isnan(bounds)] = -np.inf if bounds_index == 0 else +np.inf
                    if bounds.shape != self.rho.shape:
                        raise ValueError(f"rho_bounds[{bounds_index}] must have the same shape as rho.")
                if ((self.rho < self.rho_bounds[0]) | (self.rho > self.rho_bounds[1])).any():
                    raise ValueError("rho must be within its bounds.")

        # set upper and lower bounds to zero for parameters that are fixed at zero
        self.sigma_bounds[0][np.where(self.sigma == 0)] = self.sigma_bounds[1][np.where(self.sigma == 0)] = 0
        self.pi_bounds[0][np.where(self.pi == 0)] = self.pi_bounds[1][np.where(self.pi == 0)] = 0
        self.rho_bounds[0][np.where(self.rho == 0)] = self.rho_bounds[1][np.where(self.rho == 0)] = 0

        # store information about individual elements in sigma, pi, and rho
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

        # store information for rho
        for location in np.ndindex(self.rho.shape):
            parameter = RhoParameter(location, self.rho_bounds, single=self.rho.size == 1)
            if parameter.value is None:
                self.unfixed.append(parameter)
            else:
                self.fixed.append(parameter)

        # count the number of unfixed parameters
        self.P = len(self.unfixed)

    def compute_mu_norm(self, economy, sigma, pi):
        """Compute the infinity norm of mu computed according to specified parameters."""
        norm = 0
        if economy.K2 > 0:
            for t in economy.unique_market_ids:
                coefficients_t = sigma @ economy.agents.nodes[economy.agents.market_ids.flat == t].T
                if economy.D > 0:
                    coefficients_t += pi @ economy.agents.demographics[economy.agents.market_ids.flat == t].T
                mu_t = economy.products.X2[economy.products.market_ids.flat == t] @ coefficients_t
                norm = max(norm, np.abs(mu_t).max())
        return norm

    def normalize_bound(self, bound):
        """Reduce by 10% and round an initial parameter bound to two significant figures."""
        reduced = 0.9 * bound
        return bound if not np.isfinite(bound) else np.round(reduced, 1 + int(reduced < 1) - int(np.log10(reduced)))

    def format(self):
        """Format the initial sigma, pi, and rho as a string."""
        return self.format_matrices(self.sigma, self.pi, self.rho)

    def format_lower_bounds(self):
        """Format lower sigma, pi, and rho bounds as a string."""
        return self.format_matrices(self.sigma_bounds[0], self.pi_bounds[0], self.rho_bounds[0])

    def format_upper_bounds(self):
        """Format upper sigma, pi, and rho bounds as a string."""
        return self.format_matrices(self.sigma_bounds[1], self.pi_bounds[1], self.rho_bounds[1])

    def format_estimates(self, sigma, pi, rho, sigma_se, pi_se, rho_se):
        """Format sigma, pi, and rho estimates along with their standard errors as a string."""
        return self.format_matrices(sigma, pi, rho, sigma_se, pi_se, rho_se)

    def format_matrices(self, sigma_like, pi_like, rho_like, sigma_se_like=None, pi_se_like=None, rho_se_like=None):
        """Format matrices (and optional standard errors) of the same size as sigma, pi, and rho as a string."""
        lines = []

        # construct the primary table for sigma and pi
        if sigma_like.shape[1] > 0:
            line_indices = {}
            header = ["Sigma:"] + self.X2_labels
            widths = [max(map(len, header))] + [max(len(k), options.digits + 8) for k in header[1:]]
            if self.demographics_labels:
                line_indices = {len(widths) - 1}
                header.extend(["Pi:"] + self.demographics_labels)
                widths.extend([widths[0]] + [max(len(k), options.digits + 8) for k in header[len(widths) + 1:]])
            formatter = output.table_formatter(widths, line_indices)

            # build the top of the table
            lines.extend([formatter.line(), formatter(header, underline=True)])

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
                    unfixed_sigma_indices = set()
                    unfixed_pi_indices = set()
                    for parameter in self.unfixed:
                        if isinstance(parameter, RandomCoefficientParameter) and parameter.location[0] == row_index:
                            if isinstance(parameter, SigmaParameter):
                                unfixed_sigma_indices.add(parameter.location[1])
                            else:
                                assert isinstance(parameter, PiParameter)
                                unfixed_pi_indices.add(parameter.location[1])

                    # construct a row similar to the values row without row labels and optionally with standard errors
                    se_row = [""] * (1 + row_index)
                    for column_index in range(row_index, sigma_se_like.shape[1]):
                        se = sigma_se_like[row_index, column_index]
                        se_row.append(output.format_se(se) if column_index in unfixed_sigma_indices else "")
                    if pi_se_like.shape[1] > 0:
                        se_row.append("")
                        for column_index in range(pi_se_like.shape[1]):
                            se = pi_se_like[row_index, column_index]
                            se_row.append(output.format_se(se) if column_index in unfixed_pi_indices else "")

                    # format the row of values and add an additional blank line if there is another row of values
                    lines.append(formatter(se_row))
                    if row_index < sigma_like.shape[1] - 1:
                        lines.append(formatter.blank())

            # build the bottom of the table
            lines.append(formatter.line())

        # construct a table for rho
        if rho_like.size > 0:
            rho_header = ["Rho:"] + (self.group_labels if rho_like.size > 1 else ["All Groups"])
            rho_widths = [len(rho_header[0])] + [max(len(k), options.digits + 8) for k in rho_header[1:]]
            rho_formatter = output.table_formatter(rho_widths)

            # build the top of the table, replacing the top and bottom of the table for sigma and pi if necessary
            if not lines:
                lines.append(rho_formatter.line())
            elif len(lines[-1]) < len(rho_formatter.line()):
                lines[0] = lines[-1] = rho_formatter.line()
            lines.append(rho_formatter(rho_header, underline=True))

            # determine which elements correspond to unfixed parameters
            unfixed_rho_indices = set()
            for parameter in self.unfixed:
                if isinstance(parameter, RhoParameter):
                    unfixed_rho_indices.add(parameter.location[0])

            # build the content of the table
            lines.append(rho_formatter([""] + [output.format_number(x) for x in rho_like]))
            if rho_se_like is not None:
                se_row = [""]
                for row_index in range(rho_se_like.shape[0]):
                    se = rho_se_like[row_index]
                    se_row.append(output.format_se(se) if row_index in unfixed_rho_indices else "")
                lines.append(rho_formatter(se_row))

            # build the bottom of the table
            lines.append(lines[0])

        # combine the lines into one string
        return "\n".join(lines)

    def compress(self):
        """Compress the initial sigma, pi, and rho into theta."""
        theta = []
        for values, parameter_type in [(self.sigma, SigmaParameter), (self.pi, PiParameter), (self.rho, RhoParameter)]:
            theta.extend(values[p.location] for p in self.unfixed if isinstance(p, parameter_type))
        return np.r_[theta]

    def expand(self, theta_like, nullify=False):
        """Recover matrices of the same size as sigma, pi, and rho from a vector of the same size as theta. By default,
        fill elements corresponding to fixed parameters with their fixed values.
        """
        sigma_like = np.full_like(self.sigma, np.nan)
        pi_like = np.full_like(self.pi, np.nan)
        rho_like = np.full_like(self.rho, np.nan)

        # set values for elements that correspond to unfixed parameters
        for parameter, value in zip(self.unfixed, theta_like):
            if isinstance(parameter, SigmaParameter):
                sigma_like[parameter.location] = value
            elif isinstance(parameter, PiParameter):
                pi_like[parameter.location] = value
            else:
                assert isinstance(parameter, RhoParameter)
                rho_like[parameter.location] = value

        # set values for elements that correspond to fixed parameters
        if not nullify:
            sigma_like[np.tril_indices_from(sigma_like, -1)] = 0
            for parameter in self.fixed:
                if isinstance(parameter, SigmaParameter):
                    sigma_like[parameter.location] = parameter.value
                elif isinstance(parameter, PiParameter):
                    pi_like[parameter.location] = parameter.value
                else:
                    assert isinstance(parameter, RhoParameter)
                    rho_like[parameter.location] = parameter.value

        # return the expanded matrices
        return sigma_like, pi_like, rho_like


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
