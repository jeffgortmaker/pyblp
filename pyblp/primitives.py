"""Primitive structures that constitute the foundation of the BLP model."""

import numpy as np
import scipy.linalg

from . import options
from .construction import build_ownership
from .utilities import output, extract_matrix, Matrices, Integration


class Products(Matrices):
    r"""Structured product data.

    Attributes
    ----------
    market_ids : `ndarray`
        IDs that associate products with markets.
    firm_ids : `ndarray`
        IDs that associate products with firms. Any columns after the first represent changes such as mergers.
    ownership : `ndarray`
        Stacked :math:`J_t \times J_t` ownership matrices, :math:`O`, for each market :math:`t`. Each stack is
        associated with a :attr:`Products.firm_ids` column. If a market has fewer products than others, extra columns
        will contain ``numpy.nan``.
    shares : `ndarray`
        Shares, :math:`s`.
    prices : `ndarray`
        Prices, :math:`p`.
    X1 : `ndarray`
        Linear product characteristics, :math:`X_1`. The first column contains prices if they are configured to be a
        linear characteristic.
    X2 : `ndarray`
        Nonlinear product characteristics, :math:`X_2`. The first column contains prices if they are configured to be a
        nonlinear characteristic.
    X3 : `ndarray`
        Cost product characteristics, :math:`X_3`.
    ZD : `ndarray`
        Demand-side instruments, :math:`Z_D`.
    ZS : `ndarray`
        Supply-side instruments, :math:`Z_S`.

    """

    def __new__(cls, product_data, linear_prices, nonlinear_prices):
        """Validate and structure the data."""

        # load market, product, and firm IDs
        market_ids = extract_matrix(product_data, 'market_ids')
        firm_ids = extract_matrix(product_data, 'firm_ids')
        if market_ids is None:
            raise KeyError("product_data must have a market_ids field.")
        if market_ids.shape[1] > 1:
            raise ValueError("The market_ids field of product_data must be one-dimensional.")

        # determine the maximum number of products in a market
        J = np.unique(market_ids, return_counts=True)[1].max()

        # load or build ownership matrices
        ownership = extract_matrix(product_data, 'ownership')
        if firm_ids is None:
            ownership = None
        elif ownership is None:
            ownership = build_ownership({'market_ids': market_ids, 'firm_ids': firm_ids})
        elif ownership.shape[1] % J > 0 or ownership.shape[1] > J * firm_ids.shape[1]:
            raise ValueError(
                f"The ownership field of product_data must have a number of columns that is a multiple of {J} and that "
                f"does not exceed {J * firm_ids.shape[1]}."
            )
        elif ownership.shape[1] < J * firm_ids.shape[1]:
            unmatched_firm_ids = firm_ids[:, ownership.shape[1] / J:]
            ownership = np.c_[ownership, build_ownership({'market_ids': market_ids, 'firm_ids': unmatched_firm_ids})]

        # load shares
        shares = extract_matrix(product_data, 'shares')
        if shares is None:
            raise KeyError("product_data must have a shares field.")

        # load prices
        prices = extract_matrix(product_data, 'prices')
        if prices is None:
            raise KeyError("product_data must have a prices field.")
        if prices.shape[1] > 1:
            raise ValueError("The prices field of product_data must be one-dimensional.")

        # load characteristics
        linear_characteristics = extract_matrix(product_data, 'linear_characteristics')
        nonlinear_characteristics = extract_matrix(product_data, 'nonlinear_characteristics')
        cost_characteristics = extract_matrix(product_data, 'cost_characteristics')
        if not linear_prices and not nonlinear_prices:
            raise ValueError("linear_prices and nonlinear_prices cannot both be False.")
        if not linear_prices and linear_characteristics is None:
            raise ValueError("linear_prices is False, so product_data must have a linear_characteristics field")
        if not nonlinear_prices and nonlinear_characteristics is None:
            raise ValueError("nonlinear_prices is False, so product_data must have a nonlinear_characteristics field.")
        if firm_ids is None and cost_characteristics is not None:
            raise ValueError("Since product_data has a cost_characteristics field, it must also have firm_ids.")

        # determine the components of linear and nonlinear characteristics
        X1_list = [prices] if linear_prices else []
        X2_list = [prices] if nonlinear_prices else []
        if linear_characteristics is not None:
            X1_list.append(linear_characteristics)
        if nonlinear_characteristics is not None:
            X2_list.append(nonlinear_characteristics)

        # load instruments
        demand_instruments = extract_matrix(product_data, 'demand_instruments')
        supply_instruments = extract_matrix(product_data, 'supply_instruments')
        if (cost_characteristics is None) != (supply_instruments is None):
            raise KeyError("product_data must have a cost_characteristics field only if it has supply_instruments.")

        # structure the various components of product data
        return super().__new__(cls, {
            'market_ids': (market_ids, np.object),
            'firm_ids': (firm_ids, np.object),
            'ownership': (ownership, options.dtype),
            'shares': (shares, options.dtype),
            'prices': (prices, options.dtype),
            'X1': (np.hstack(X1_list), options.dtype),
            'X2': (np.hstack(X2_list), options.dtype),
            'X3': (cost_characteristics, options.dtype),
            'ZD': (demand_instruments, options.dtype),
            'ZS': (supply_instruments, options.dtype)
        })


class Agents(Matrices):
    r"""Structured agent data.

    Attributes
    ----------
    market_ids : `ndarray`
        IDs that associate agents with markets.
    weights : `ndarray`
        Integration weights, :math:`w`.
    nodes : `ndarray`
        Unobserved agent characteristics called integration nodes, :math:`\nu`.
    demographics : `ndarray`
        Observed agent characteristics, :math:`d`.

    """

    def __new__(cls, products, agent_data=None, integration=None):
        """Validate and structure the data."""

        # count the number of nonlinear characteristics
        K2 = products.X2.shape[1]

        # make sure that agent data and integration weren't mixed up
        if isinstance(agent_data, Integration):
            raise ValueError("Integration instances must be passed as an integration argument, not agent_data.")

        # load market IDs and any demographics
        market_ids = demographics = None
        if agent_data is not None:
            market_ids = extract_matrix(agent_data, 'market_ids')
            demographics = extract_matrix(agent_data, 'demographics')

            # validate the market IDs
            if market_ids is None:
                raise KeyError("agent_data must have a market_ids field.")
            if market_ids.shape[1] > 1:
                raise ValueError("The market_ids field of agent_data must be one-dimensional.")
            if set(np.unique(products.market_ids)) != set(np.unique(market_ids)):
                raise ValueError("The market_ids field of agent_data must have the same set of IDs as product data.")

        # if configured, build nodes and weights
        deleted = False
        nodes = weights = None
        if integration is not None:
            if not isinstance(integration, Integration):
                raise ValueError("integration must be an Integration instance.")

            # output a message about the integration configuration
            output(integration)

            # build nodes and weights for each market, deleting demographic rows if necessary
            output(f"Building {K2}-dimensional nodes and weights for each market ...")
            loaded_market_ids = market_ids
            market_ids, nodes, weights = integration._build_many(K2, np.unique(products.market_ids))
            if demographics is not None:
                demographics_list = []
                for t in np.unique(products.market_ids):
                    built = (market_ids == t).sum()
                    loaded = (loaded_market_ids == t).sum()
                    if built > loaded:
                        raise ValueError(f"Market '{t}' in agent_data must have at least {built} rows.")
                    demographics_t = demographics[loaded_market_ids.flat == t]
                    if built < loaded:
                        demographics_t = demographics_t[:built]
                        deleted = True
                    demographics_list.append(demographics_t)
                demographics = np.concatenate(demographics_list)

            # output a message about deleted demographic rows
            if deleted:
                output(
                    "Ignored at least one row in the demographics field of agent_data so that there are as many rows "
                    "in each market as there are rows of built node and weight rows in each market."
                )

        # if not already build, load nodes and weights
        if integration is None:
            nodes = extract_matrix(agent_data, 'nodes')
            weights = extract_matrix(agent_data, 'weights')

            # validate the data
            if nodes is None or weights is None:
                raise KeyError("Since integration is None, agent_data must have both weights and nodes fields.")
            if nodes.shape[1] < K2:
                raise ValueError(f"The number of columns in the nodes field of agent_data must be at least K2, {K2}.")
            if weights.shape[1] != 1:
                raise ValueError("The weights field of agent_data must be one-dimensional.")

            # delete columns of nodes if there are too many
            if nodes is not None and nodes.shape[1] > K2:
                nodes = nodes[:, :K2]
                output(f"Ignored at least one column in the nodes field of agent_data. Remaining columns: {K2}.")

        # structure the various components of agent data
        return super().__new__(cls, {
            'market_ids': (market_ids, np.object),
            'weights': (weights, options.dtype),
            'nodes': (nodes, options.dtype),
            'demographics': (demographics, options.dtype)
        })


class Market(object):
    """A single market, which is always initialized with product and agent data. Given additional information such as
    utility components or parameters, class methods can compute a variety of market-specific outputs.
    """

    def __init__(self, t, linear_prices, nonlinear_prices, products, agents, delta=None, xi=None, beta=None, sigma=None,
                 pi=None):
        """Restrict full data matrices and vectors to just this market. If arguments are left unspecified, dependent
        methods will raise exceptions.
        """
        self.t = t
        self.linear_prices = linear_prices
        self.nonlinear_prices = nonlinear_prices

        # restrict product and agent data to just this market
        self.products = products[products.market_ids.flat == t]
        self.agents = agents[agents.market_ids.flat == t]

        # count data dimensions
        self.J = self.products.shape[0]
        self.I = self.agents.shape[0]
        self.K1 = self.products.X1.shape[1]
        self.K2 = self.products.X2.shape[1]
        try:
            self.K3 = self.products.X3.shape[1]
        except AttributeError:
            self.K3 = 0
        try:
            self.D = self.agents.demographics.shape[1]
        except AttributeError:
            self.D = 0

        # store parameter matrices
        self.beta = beta
        self.sigma = sigma
        self.pi = pi

        # store utilities or compute them if possible
        self.mu = self.xi = self.delta = None
        if sigma is not None:
            self.mu = self.compute_mu()
        if xi is not None:
            self.xi = xi[products.market_ids.flat == t]
        if delta is not None:
            self.delta = delta[products.market_ids.flat == t]
        elif xi is not None and beta is not None:
            self.delta = self.compute_delta()

    def get_price_indices(self):
        """Get the indices of prices in X1 and X2."""
        X1_index = 0 if self.linear_prices else None
        X2_index = 0 if self.nonlinear_prices else None
        return X1_index, X2_index

    def get_characteristic(self, X1_index=None, X2_index=None):
        """Get the values for a product characteristic in X1 or X2 (or both)."""
        return self.products.X1[:, [X1_index]] if X2_index is None else self.products.X2[:, [X2_index]]

    def get_ownership_matrix(self, firms_index=0):
        """Get an ownership matrix. By default, unchanged firm IDs are used."""
        offset = firms_index * self.products.ownership.shape[1] // self.products.firm_ids.shape[1]
        return self.products.ownership[:, offset:offset + self.J]

    def compute_delta(self, X1=None):
        """Compute delta. By default, the X1 with which this market was initialized is used."""
        if X1 is None:
            X1 = self.products.X1
        return X1 @ self.beta + self.xi

    def compute_mu(self, X2=None):
        """Compute mu. By default, the X2 with which this market was initialized is used."""
        if X2 is None:
            X2 = self.products.X2
        mu = X2 @ self.sigma @ self.agents.nodes.T
        if self.D > 0:
            mu += X2 @ self.pi @ self.agents.demographics.T
        return mu

    def update_delta_with_characteristic(self, characteristic, X1_index=None):
        """Update delta to reflect a changed product characteristic if it is a column in X1."""
        if X1_index is None:
            return self.delta
        X1 = self.products.X1.copy()
        X1[:, [X1_index]] = characteristic
        return self.compute_delta(X1)

    def update_delta_with_prices(self, prices):
        """Update delta to reflect changed prices."""
        X1_index = self.get_price_indices()[0]
        return self.update_delta_with_characteristic(prices, X1_index)

    def update_mu_with_characteristic(self, characteristic, X2_index=None):
        """Update mu to reflect a changed product characteristic if it is a column in X2."""
        if X2_index is None:
            return self.mu
        X2 = self.products.X2.copy()
        X2[:, [X2_index]] = characteristic
        return self.compute_mu(X2)

    def update_mu_with_prices(self, prices):
        """Update mu to reflect changed prices if they are a column in X2."""
        X2_index = self.get_price_indices()[1]
        return self.update_mu_with_characteristic(prices, X2_index)

    def compute_probabilities(self, delta=None, mu=None, linear=True, eliminate_product=None):
        """Compute choice probabilities. By default, the delta with which this market was initialized and the mu that
        was computed during initialization are used. If linear is False, delta and mu must be specified and must have
        already been exponentiated. If eliminate_product is specified, the product associated with the specified index
        is eliminated from the choice set.
        """
        if delta is None:
            delta = self.delta
        if mu is None:
            mu = self.mu
        exp_utilities = np.exp(np.tile(delta, self.I) + mu) if linear else np.tile(delta, self.I) * mu
        if eliminate_product is not None:
            exp_utilities[eliminate_product] = 0
        return exp_utilities / (1 + exp_utilities.sum(axis=0))

    def compute_utilities_by_characteristic_jacobian(self, X1_index=None, X2_index=None):
        """Compute the Jacobian of utilities with respect to a product characteristic in X1 or X2 (or both)."""
        jacobian = np.zeros((self.J, self.I))
        if X1_index is not None:
            jacobian += self.beta[X1_index]
        if X2_index is not None:
            jacobian += self.sigma[X2_index] @ self.agents.nodes.T
            if self.D > 0:
                jacobian += self.pi[X2_index] @ self.agents.demographics.T
        return jacobian

    def compute_utilities_by_prices_jacobian(self):
        """Compute the Jacobian of utilities with respect to prices."""
        X1_index, X2_index = self.get_price_indices()
        return self.compute_utilities_by_characteristic_jacobian(X1_index, X2_index)

    def compute_shares_by_characteristic_jacobian(self, utilities_jacobian, probabilities=None):
        """Use the Jacobian of utilities with respect to a product characteristic in X1 or X2 (or both) to compute the
        Jacobian of market shares with respect to the same characteristic. By default, unchanged choice probabilities
        are computed.
        """
        if probabilities is None:
            probabilities = self.compute_probabilities()
        V = probabilities * utilities_jacobian
        capital_lambda = np.diagflat(V @ self.agents.weights)
        capital_gamma = V @ np.diagflat(self.agents.weights) @ probabilities.T
        return capital_lambda - capital_gamma

    def compute_zeta(self, ownership_matrix, utilities_jacobian, costs, prices=None):
        """Use an ownership matrix, the Jacobian of utilities with respect to prices, and marginal costs to compute the
        markup term in the zeta-markup equation. By default, unchanged choice probabilities are computed and the prices
        and shares with which this market was initialized are used.
        """
        if prices is None:
            probabilities = self.compute_probabilities()
            shares = self.products.shares
        else:
            delta = self.update_delta_with_prices(prices)
            mu = self.update_mu_with_prices(prices)
            probabilities = self.compute_probabilities(delta, mu)
            shares = probabilities @ self.agents.weights
        V = probabilities * utilities_jacobian
        capital_lambda_inverse = np.diagflat(1 / (V @ self.agents.weights))
        capital_gamma = V @ np.diagflat(self.agents.weights) @ probabilities.T
        tilde_capital_omega = capital_lambda_inverse @ (ownership_matrix * capital_gamma).T
        return tilde_capital_omega @ (prices - costs) - capital_lambda_inverse @ shares

    def compute_eta(self, ownership_matrix, utilities_jacobian, prices=None):
        """Use an ownership matrix and the Jacobian of utilities with respect to prices to compute the markup term in
        the BLP-markup equation. By default, unchanged choice probabilities are computed and the prices and shares with
        which this market was initialized are used.
        """
        if prices is None:
            probabilities = self.compute_probabilities()
            shares = self.products.shares
        else:
            delta = self.update_delta_with_prices(prices)
            mu = self.update_mu_with_prices(prices)
            probabilities = self.compute_probabilities(delta, mu)
            shares = probabilities @ self.agents.weights
        shares_jacobian = self.compute_shares_by_characteristic_jacobian(utilities_jacobian, probabilities)
        try:
            return -scipy.linalg.solve(ownership_matrix * shares_jacobian, shares)
        except ValueError:
            return np.full_like(shares, np.nan)

    def compute_costs(self):
        """Compute marginal costs with the BLP-markup equation."""
        ownership_matrix = self.get_ownership_matrix()
        jacobian = self.compute_utilities_by_prices_jacobian()
        return self.products.prices - self.compute_eta(ownership_matrix, jacobian)
