"""Primitive structures that constitute the foundation of the BLP model."""

import itertools

import numpy as np
import scipy.linalg
import numpy.lib.recfunctions

from . import options, exceptions
from .configurations import Formulation, Integration
from .utilities import output, extract_matrix, iteratively_demean, Matrices


class Products(Matrices):
    r"""Structured product data. The following fields are always present:

        - **market_ids** : (`object`) - IDs that associate products with markets.

        - **shares** : (`numeric`) - Market shares, :math:`s`.

        - **prices** : (`numeric`) - Product prices, :math:`p`.

        - **X1** : (`numeric`) - Linear product characteristics, :math:`X_1`, which may have been iteratively demeaned
          to absorb any demand-side fixed effects.

        - **X2** : (`numeric`) - Nonlinear product characteristics, :math:`X_2`.

        - **ZD** : (`numeric`) - Demand-side instruments, :math:`Z_D`, which may have been iteratively demeaned to
          absorb any demand-side fixed effects.

    Depending on the configuration, the following fields may also be present:

        - **firm_ids** : (`object`) - IDs that associate products with firms. Any columns after the first represent
          changes such as mergers.

        - **ownership** : (`object`) - Stacked :math:`J_t \times J_t` ownership matrices, :math:`O`, for each market
          :math:`t`. Each stack is associated with a `firm_ids` column.

        - **X3** : (`numeric`) - Cost product characteristics, :math:`X_3`, which may have been iteratively demeaned to
          absorb any supply-side fixed effects.

        - **ZS** : (`numeric`) - Supply-side instruments, :math:`Z_S`, which may have been iteratively demeaned to
          absorb any supply-side fixed effects.

        - **demand_ids** : (`object`) - Categorical variables used to create demand-side fixed effects.

        - **supply_ids** : (`object`) - Categorical variables used to create supply-side fixed effects.

    Any additional fields are the variables underlying `X1`, `X2`, and `X3`.

    """

    def __new__(cls, product_formulations, product_data, demeaning_iteration=None):
        """Validate and structure product data while absorbing any fixed effects."""

        # validate the formulations
        if not all(isinstance(f, Formulation) for f in product_formulations) or len(product_formulations) not in {2, 3}:
            raise TypeError("product_formulations must be a tuple of two or three Formulation instances.")

        # build X1 and X2
        X1, X1_formulations, X1_data = product_formulations[0]._build(product_data)
        X2, X2_formulations, X2_data = product_formulations[1]._build(product_data)
        if 'shares' in X1_data:
            raise NameError("shares cannot be included in the formulation for X1.")
        if 'shares' in X2_data:
            raise NameError("shares cannot be included in the formulation for X2.")
        if 'prices' not in X1_data and 'prices' not in X2_data:
            raise NameError("prices must be included in the formulation for X1 or X2 (or both).")

        # build X3
        X3 = None
        X3_data = {}
        X3_formulations = []
        if len(product_formulations) == 3:
            X3, X3_formulations, X3_data = product_formulations[2]._build(product_data)
            if 'shares' in X3_data:
                raise NameError("shares cannot be included in the formulation for X3.")
            if 'prices' in X3_data:
                raise NameError("prices cannot be included in the formulation for X3.")

        # load instruments
        ZD = extract_matrix(product_data, 'demand_instruments')
        ZS = extract_matrix(product_data, 'supply_instruments')
        if ZD is None:
            raise KeyError("product_data must have a demand_instruments field.")
        if (ZS is None) != (X3 is None):
            raise KeyError("product_data must have a supply_instruments field only when X3 is formulated.")

        # load IDs
        market_ids = extract_matrix(product_data, 'market_ids')
        firm_ids = extract_matrix(product_data, 'firm_ids')
        demand_ids = extract_matrix(product_data, 'demand_ids')
        supply_ids = extract_matrix(product_data, 'supply_ids')
        if market_ids is None:
            raise KeyError("product_data must have a market_ids field.")
        if market_ids.shape[1] > 1:
            raise ValueError("The market_ids field of product_data must be one-dimensional.")
        if firm_ids is None and X3 is not None:
            raise KeyError("product_data must have firm_ids field when X3 is formulated.")
        if supply_ids is not None and X3 is None:
            raise KeyError("product_data must not have a supply_ids field when X3 not is formulated.")

        # load ownership matrices
        ownership = None
        if firm_ids is not None:
            ownership = extract_matrix(product_data, 'ownership')
            if ownership is not None:
                J = np.unique(market_ids, return_counts=True)[1].max()
                if ownership.shape[1] != J * firm_ids.shape[1]:
                    raise ValueError(f"The ownership field of product_data must have {J * firm_ids.shape[1]} columns.")

        # load shares
        shares = extract_matrix(product_data, 'shares')
        if shares is None:
            raise KeyError("product_data must have a shares field.")
        if shares.shape[1] > 1:
            raise ValueError("The shares field of product_data must be one-dimensional.")

        # absorb any demand-side fixed effects with iterative demeaning
        if demand_ids is not None:
            X1, X1_errors = iteratively_demean(X1, demand_ids, demeaning_iteration)
            ZD, ZD_errors = iteratively_demean(ZD, demand_ids, demeaning_iteration)
            if X1_errors | ZD_errors:
                raise exceptions.MultipleErrors(X1_errors | ZD_errors)

        # absorb any supply-side fixed effects with iterative demeaning
        if supply_ids is not None:
            X3, X3_errors = iteratively_demean(X3, supply_ids, demeaning_iteration)
            ZS, ZS_errors = iteratively_demean(ZS, supply_ids, demeaning_iteration)
            if X3_errors | ZS_errors:
                raise exceptions.MultipleErrors(X3_errors | ZS_errors)

        # structure product fields as a mapping
        product_mapping = {
            'market_ids': (market_ids, np.object),
            'firm_ids': (firm_ids, np.object),
            'demand_ids': (demand_ids, np.object),
            'supply_ids': (supply_ids, np.object),
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
            raise NameError(f"These names in product_formulations are invalid: {list(invalid_names)}.")
        product_mapping.update({k: (v, options.dtype) for k, v in underlying_data.items()})

        # structure products
        return super().__new__(cls, product_mapping)


class Agents(Matrices):
    r"""Structured agent data. The following fields are always present:

        - **market_ids** : (`object`) - IDs that associate agents with markets.

        - **weights** : (`numeric`) - Integration weights, :math:`w`.

        - **nodes** : (`numeric`) - Unobserved agent characteristics called integration nodes, :math:`\nu`.

    Depending on the configuration, the following field may also be present:

        - **demographics** : (`numeric`) - Observed agent characteristics, :math:`d`.

    """

    def __new__(cls, products, agent_formulation=None, agent_data=None, integration=None):
        """Validate and structure agent data."""

        # count the number of nonlinear characteristics
        K2 = products.X2.shape[1]

        # validate the formulation and build demographics
        demographics = None
        demographics_formulations = []
        if agent_formulation is not None:
            if not isinstance(agent_formulation, Formulation):
                raise TypeError("agent_formulation must be a Formulation instance.")
            if agent_data is None:
                raise ValueError("agent_formulation is specified, so agent_data must be specified as well.")
            demographics, demographics_formulations = agent_formulation._build(agent_data)[:2]

        # verify that agent data and integration weren't mixed up
        if isinstance(agent_data, Integration):
            raise ValueError("Integration instances must be passed to the integration argument, not agent_data.")

        # load market IDs
        market_ids = None
        if agent_data is not None:
            market_ids = extract_matrix(agent_data, 'market_ids')
            if market_ids is None:
                raise KeyError("agent_data must have a market_ids field.")
            if market_ids.shape[1] > 1:
                raise ValueError("The market_ids field of agent_data must be one-dimensional.")
            if set(np.unique(products.market_ids)) != set(np.unique(market_ids)):
                raise ValueError("The market_ids field of agent_data must have the same set of IDs as product data.")

        # build nodes and weights
        nodes = weights = None
        if integration is not None:
            if not isinstance(integration, Integration):
                raise ValueError("integration must be an Integration instance.")
            loaded_market_ids = market_ids
            market_ids, nodes, weights = integration._build_many(K2, np.unique(products.market_ids))

            # delete demographic rows if there are too many
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

        # if not already built, load nodes and weights
        if integration is None:
            nodes = extract_matrix(agent_data, 'nodes')
            weights = extract_matrix(agent_data, 'weights')
            if nodes is None or weights is None:
                raise KeyError("Since integration is None, agent_data must have both weights and nodes fields.")
            if nodes.shape[1] < K2:
                raise ValueError(f"The number of columns in the nodes field of agent_data must be at least {K2}.")
            if weights.shape[1] != 1:
                raise ValueError("The weights field of agent_data must be one-dimensional.")

            # delete node columns if there are too many
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
        """Store formulations and data data, count dimensions, and identify column formulations."""

        # store formulations and data
        self.product_formulations = product_formulations
        self.agent_formulation = agent_formulation
        self.products = products
        self.agents = agents

        # identify unique markets
        self.unique_market_ids = np.unique(self.products.market_ids).flatten()

        # count dimensions
        self.N = self.products.shape[0]
        self.T = self.unique_market_ids.size
        self.K1 = self.products.X1.shape[1]
        self.K2 = self.products.X2.shape[1]
        self.K3 = self.products.X3.shape[1] if 'X3' in self.products.dtype.fields else 0
        self.D = self.agents.demographics.shape[1] if 'demographics' in self.agents.dtype.fields else 0
        self.MD = self.products.ZD.shape[1]
        self.MS = self.products.ZS.shape[1] if 'ZS' in self.products.dtype.fields else 0
        self.ED = self.products.demand_ids.shape[1] if 'demand_ids' in self.products.dtype.fields else 0
        self.ES = self.products.supply_ids.shape[1] if 'supply_ids' in self.products.dtype.fields else 0

        # identify column formulations
        self._X1_formulations = self.products.dtype.fields['X1'][2]
        self._X2_formulations = self.products.dtype.fields['X2'][2]
        self._X3_formulations = self.products.dtype.fields['X3'][2] if self.K3 > 0 else tuple()
        self._demographics_formulations = self.agents.dtype.fields['demographics'][2] if self.D > 0 else tuple()

    def __str__(self):
        """Format economy information as a string."""
        sections = []

        # collect all dimensions and matrix labels
        dimension_items = [(k, str(getattr(self, k))) for k in 'N T K1 K3 K3 D MD MS ED ES'.split()]
        matrix_items = [
            ('X1', [str(f) for f in self._X1_formulations]),
            ('X2', [str(f) for f in self._X2_formulations]),
            ('X3', [str(f) for f in self._X3_formulations]),
            ('d', [str(f) for f in self._demographics_formulations])
        ]

        # build the section for dimensions
        dimension_header = [k for k, _ in dimension_items]
        dimension_widths = [max(5, len(k), len(v)) for k, v in dimension_items]
        dimension_formatter = output.table_formatter(dimension_widths)
        sections.append([
            "Dimensions:",
            dimension_formatter.border(),
            dimension_formatter(dimension_header, underline=True),
            dimension_formatter([v for _, v in dimension_items]),
            dimension_formatter.border()
        ])

        # build the section for matrix labels
        matrix_header = ["Indices:"]
        matrix_widths = [len(matrix_header[0])]
        for matrix_index in range(max(self.K1, self.K2, self.K3, self.D)):
            matrix_header.append(f'#{matrix_index}')
            matrix_widths.append(max([5] + [len(l[matrix_index]) for _, l in matrix_items if len(l) > matrix_index]))
        matrix_formatter = output.table_formatter(matrix_widths)
        matrix_section = [
            "Matrices:",
            matrix_formatter.border(),
            matrix_formatter(matrix_header, underline=True)
        ]
        for name, labels in matrix_items:
            if labels:
                matrix_section.append(matrix_formatter([name] + labels))
        matrix_section.append(matrix_formatter.border())
        sections.append(matrix_section)

        # combine the sections into one string
        return "\n\n".join("\n".join(s) for s in sections)

    def __repr__(self):
        """Defer to the string representation."""
        return str(self)


class Market(object):
    """A single market in an economy, which is initialized with market-level product and agent data, excluding any
    unneeded product data fields. Given additional information such as components of utility or parameters, class
    methods can compute a variety of market-specific outputs.
    """

    field_whitelist = field_blacklist = None

    def __init__(self, economy, t, delta=None, beta=None, sigma=None, pi=None):
        """Restrict full data matrices and vectors to just this market, count dimensions, identify column formulations,
        store parameter matrices, and either store or pre-compute utility components. If arguments are left unspecified,
        dependent methods will raise exceptions.
        """

        # restrict relevant product data to just this market
        unneeded_product_fields = {'market_ids'}
        if self.field_whitelist is not None:
            unneeded_product_fields |= set(economy.products.dtype.names) - self.field_whitelist
        if self.field_blacklist is not None:
            unneeded_product_fields |= set(economy.products.dtype.names) & self.field_blacklist
        self.products = economy.products[economy.products.market_ids.flat == t]
        self.products = np.lib.recfunctions.rec_drop_fields(self.products, unneeded_product_fields)

        # restrict relevant agent data to just this market
        self.agents = economy.agents[economy.agents.market_ids.flat == t]
        self.agents = np.lib.recfunctions.rec_drop_fields(self.agents, 'market_ids')

        # count data dimensions
        self.J = self.products.shape[0]
        self.I = self.agents.shape[0]
        self.K1 = economy.K1
        self.K2 = economy.K2
        self.K3 = economy.K3
        self.D = economy.D
        self.MD = economy.MD
        self.MS = economy.MS
        self.ED = economy.ED
        self.ES = economy.ES

        # identify column formulations
        self._X1_formulations = economy._X1_formulations
        self._X2_formulations = economy._X2_formulations
        self._X3_formulations = economy._X3_formulations
        self._demographics_formulations = economy._demographics_formulations

        # store parameter matrices
        self.beta = beta
        self.sigma = sigma
        self.pi = pi

        # compute or store utility components
        self.mu = self.delta = None
        if sigma is not None:
            self.mu = self.compute_mu()
        if delta is not None:
            self.delta = delta[economy.products.market_ids.flat == t]

    def get_ownership_matrix(self, firms_index=0):
        """Get a pre-computed ownership matrix or builds one. By default, use unchanged firm IDs."""
        try:
            offset = firms_index * self.products.ownership.shape[1] // self.products.firm_ids.shape[1]
            return self.products.ownership[:, offset:offset + self.J]
        except AttributeError:
            tiled_ids = np.tile(self.products.firm_ids[:, [firms_index]], self.J)
            return np.where(tiled_ids == tiled_ids.T, 1, 0)

    def compute_mu(self, X2=None):
        """Compute mu. By default, use the X2 with which this market was initialized."""
        if X2 is None:
            X2 = self.products.X2
        return X2 @ (self.sigma @ self.agents.nodes.T + (self.pi @ self.agents.demographics.T if self.D > 0 else 0))

    def update_delta_with_variable(self, name, variable):
        """Update delta to reflect a changed variable if it contributes to X1. To do so, add the parameter-weighted
        change in the characteristic (delta cannot simply be recomputed because doing so would not account for any
        fixed effects).
        """

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
        """Update mu to reflect a changed variable if it contributes to X2. To do so, recompute mu with the new X2."""

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
        """Compute choice probabilities. By default, use the delta with which this market was initialized and the mu
        that was computed during initialization. If linear is False, delta and mu must be specified and must have
        already been exponentiated. If eliminate_product is specified, eliminate the product associated with the
        specified index from the choice set.
        """
        if delta is None:
            delta = self.delta
        if mu is None:
            mu = self.mu
        exp_utilities = np.exp(np.tile(delta, self.I) + mu) if linear else np.tile(delta, self.I) * mu
        if eliminate_product is not None:
            exp_utilities[eliminate_product] = 0
        return exp_utilities / (1 + exp_utilities.sum(axis=0))

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
        """Compute derivatives of utility with respect to a variable in X1 or X2 (or both). By default, use unchanged
        variable values.
        """
        nonlinear = self.sigma @ self.agents.nodes.T + (self.pi @ self.agents.demographics.T if self.D > 0 else 0)
        return (
            self.compute_X1_by_variable_derivatives(name, variable) @ self.beta +
            self.compute_X2_by_variable_derivatives(name, variable) @ nonlinear
        )

    def compute_shares_by_variable_jacobian(self, derivatives, probabilities=None):
        """Use derivatives of utility with respect to a variable in X1 or X2 (or both) to compute the Jacobian of market
        shares with respect to the same variable. By default, compute unchanged choice probabilities.
        """
        if probabilities is None:
            probabilities = self.compute_probabilities()
        V = probabilities * derivatives
        capital_lambda = np.diagflat(V @ self.agents.weights)
        capital_gamma = V @ np.diagflat(self.agents.weights) @ probabilities.T
        return capital_lambda - capital_gamma

    def compute_eta(self, ownership_matrix=None, derivatives=None, prices=None):
        """Compute the markup term in the BLP-markup equation. By default, construct an ownership matrix, compute
        derivatives of utilities with respect to prices, and use the prices with which this market was initialized.
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
        """Compute the markup term in the zeta-markup equation. By default, construct an ownership matrix, compute
        derivatives of utilities with respect to prices, use the prices with which this market was initialized, and
        compute marginal costs.
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

    def get_characteristics(self, products, agents):
        """Get the product and agents characteristics associated with the parameter."""
        raise NotImplementedError


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
    """Combined information about sigma and pi."""

    def __init__(self, economy, sigma, pi, sigma_bounds=None, pi_bounds=None, supports_bounds=True):
        """Validate parameter matrices and their bounds. Then, construct lists of information about fixed (equal bounds)
        and unfixed (unequal bounds) elements of sigma and pi.
        """

        # store labels
        self.X2_labels = [str(f) for f in economy._X2_formulations]
        self.demographics_labels = [str(f) for f in economy._demographics_formulations]

        # validate and clean up sigma
        self.sigma = np.c_[np.asarray(sigma, options.dtype)]
        if self.sigma.shape != (economy.K2, economy.K2):
            raise ValueError(f"sigma must be {economy.K2} by {economy.K2} matrix.")
        self.sigma[np.tril_indices(economy.K2, -1)] = 0

        # validate and clean up pi
        if (pi is None) != (economy.D == 0):
            raise ValueError("pi should be None only when there are no demographics.")
        self.pi = None
        if pi is not None:
            self.pi = np.c_[np.asarray(pi, options.dtype)]
            if self.pi.shape != (economy.K2, economy.D):
                raise ValueError(f"pi must be a {economy.K2} by {economy.D} matrix.")

        # construct default sigma bounds or validate specified bounds
        if sigma_bounds is None or not supports_bounds:
            self.sigma_bounds = (
                np.full_like(self.sigma, -np.inf, options.dtype), np.full_like(self.sigma, +np.inf, options.dtype)
            )
            if supports_bounds:
                np.fill_diagonal(self.sigma_bounds[0], 0)
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

        # construct default pi bounds or validate specified bounds
        if self.pi is None:
            self.pi_bounds = (None, None)
        elif pi_bounds is None or not supports_bounds:
            self.pi_bounds = (
                np.full_like(self.pi, -np.inf, options.dtype), np.full_like(self.pi, +np.inf, options.dtype)
            )
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

        # set upper and lower bounds to zero for parameters that are fixed at zero
        self.sigma_bounds[0][np.where(self.sigma == 0)] = self.sigma_bounds[1][np.where(self.sigma == 0)] = 0
        if self.pi is not None:
            self.pi_bounds[0][np.where(self.pi == 0)] = self.pi_bounds[1][np.where(self.pi == 0)] = 0

        # store information about individual sigma and pi elements in lists
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
        if self.pi is not None:
            for location in np.ndindex(self.pi.shape):
                parameter = PiParameter(location, self.pi_bounds)
                if parameter.value is None:
                    self.unfixed.append(parameter)
                else:
                    self.fixed.append(parameter)

        # count the number of unfixed parameters
        self.P = len(self.unfixed)

    def format(self, sigma_like, pi_like, sigma_se=None, pi_se=None):
        """Format matrices that are of the same size as the matrices of nonlinear parameters as a string. If matrices
        of standard errors are given, they will be formatted as numbers surrounded by parentheses underneath the
        elements of sigma and pi.
        """
        lines = []

        # construct the table header and formatter
        line_indices = {}
        header = ["Sigma:"] + self.X2_labels
        widths = [max(map(len, header))] + [max(len(k), options.digits + 8) for k in header[1:]]
        if self.pi is not None:
            line_indices = {len(widths) - 1}
            header.extend(["Pi:"] + self.demographics_labels)
            widths.extend([widths[0]] + [max(len(k), options.digits + 8) for k in header[len(widths) + 1:]])
        formatter = output.table_formatter(widths, line_indices)

        # build the top of the table
        lines.extend([formatter(header, underline=True)])

        # construct the rows containing parameter information
        for row_index, row_label in enumerate(self.X2_labels):
            # the row of values consists of the label, blanks for the lower triangle of sigma, sigma values, the label
            #   again, and finally pi values
            values_row = [row_label] + [""] * row_index
            for column_index in range(row_index, sigma_like.shape[1]):
                values_row.append(output.format_number(sigma_like[row_index, column_index]))
            if self.pi is not None:
                values_row.append(row_label)
                for column_index in range(pi_like.shape[1]):
                    values_row.append(output.format_number(pi_like[row_index, column_index]))
            lines.append(formatter(values_row))

            # construct a row of standard errors for unfixed parameters
            if sigma_se is not None:
                # determine which columns in this row correspond to unfixed parameters
                sigma_indices = set()
                pi_indices = set()
                for parameter in self.unfixed:
                    if parameter.location[0] == row_index:
                        if isinstance(parameter, SigmaParameter):
                            sigma_indices.add(parameter.location[1])
                        else:
                            pi_indices.add(parameter.location[1])

                # construct a row similar to the values row without row labels and with standard error formatting
                se_row = [""] * (1 + row_index)
                for column_index in range(row_index, sigma_se.shape[1]):
                    se = sigma_se[row_index, column_index]
                    se_row.append(output.format_se(se) if column_index in sigma_indices else "")
                if self.pi is not None:
                    se_row.append("")
                    for column_index in range(pi_se.shape[1]):
                        se = pi_se[row_index, column_index]
                        se_row.append(output.format_se(se) if column_index in pi_indices else "")

                # format the row of values and add an additional blank line if there is another row of values
                lines.append(formatter(se_row))
                if row_index < sigma_like.shape[1] - 1:
                    lines.append(formatter.blank())

        # wrap the table in borders and combine the lines into one string
        return "\n".join([formatter.border()] + lines + [formatter.border()])

    def compress(self, sigma, pi):
        """Compress nonlinear parameter matrices into theta."""
        sigma_locations = list(zip(*[p.location for p in self.unfixed if isinstance(p, SigmaParameter)]))
        theta = sigma[sigma_locations].ravel()
        if pi is not None:
            pi_locations = list(zip(*[p.location for p in self.unfixed if isinstance(p, PiParameter)]))
            theta = np.r_[theta, pi[pi_locations].ravel()]
        return theta

    def expand(self, theta_like, fill_fixed=False):
        """Recover nonlinear parameter-sized matrices from a vector of the same size as theta. If fill_fixed is True,
        elements corresponding to fixed parameters will be set to their fixed values instead of to None.
        """
        sigma_like = np.full_like(self.sigma, np.nan)
        pi_like = None if self.pi is None else np.full_like(self.pi, np.nan)

        # set values for elements that correspond to unfixed parameters
        for parameter, value in zip(self.unfixed, theta_like):
            if isinstance(parameter, SigmaParameter):
                sigma_like[parameter.location] = value
            else:
                pi_like[parameter.location] = value

        # set values for elements that correspond to fixed parameters
        if fill_fixed:
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

    def __init__(self, economy, beta, gamma):
        """Validate parameter vectors."""

        # store labels
        self.X1_labels = [str(f) for f in economy._X1_formulations]
        self.X3_labels = [str(f) for f in economy._X3_formulations]

        # validate beta
        self.beta = np.c_[np.asarray(beta, options.dtype)]
        if self.beta.shape != (economy.K1, 1):
            raise ValueError(f"beta must be a {economy.K1}-vector.")

        # validate gamma
        self.gamma = None
        if gamma is not None:
            self.gamma = np.c_[np.asarray(gamma, options.dtype)]
            if self.gamma.shape != (economy.K3, 1):
                raise ValueError(f"gamma must be a {economy.K3}-vector.")

    def format(self, beta_like, gamma_like, beta_se=None, gamma_se=None):
        """Format matrices that are of the same size as the vectors of linear parameters as a string. If vectors of
        standard errors are given, they will be formatted as numbers surrounded by parentheses underneath the elements
        of beta and gamma.
        """
        lines = []

        # build the header for beta
        beta_header = ["Beta:"] + self.X1_labels
        beta_widths = [len(beta_header[0])] + [max(len(k), options.digits + 8) for k in beta_header[1:]]

        # build the header for gamma
        gamma_header = None
        gamma_widths = []
        if self.gamma is not None:
            gamma_header = ["Gamma:"] + self.X3_labels
            gamma_widths = [len(gamma_header[0])] + [max(len(k), options.digits + 8) for k in gamma_header[1:]]

        # build the table formatter
        widths = [max(w) for w in itertools.zip_longest(beta_widths, gamma_widths, fillvalue=0)]
        formatter = output.table_formatter(widths)

        # build the table
        lines.extend([
            formatter.border(),
            formatter(beta_header, underline=True),
            formatter([""] + [output.format_number(x) for x in beta_like])
        ])
        if beta_se is not None:
            lines.append(formatter([""] + [output.format_se(x) for x in beta_se]))
        if self.gamma is not None:
            lines.extend([
                formatter.line(),
                formatter(gamma_header, underline=True),
                formatter([""] + [output.format_number(x) for x in gamma_like])
            ])
            if gamma_se is not None:
                lines.append(formatter([""] + [output.format_se(x) for x in gamma_se]))
        lines.append(formatter.border())

        # combine the lines into one string
        return "\n".join(lines)
