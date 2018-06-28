"""Simulation of synthetic BLP problem data."""

import time

import numpy as np

from . import options, exceptions
from .construction import build_blp_instruments, build_matrix
from .configurations import Formulation, Iteration, Integration
from .utilities import output, extract_matrix, Matrices, ParallelItems
from .primitives import Products, Agents, Economy, Market, NonlinearParameters, LinearParameters


class Simulation(Economy):
    r"""Simulation of synthetic BLP data.

    All data are either loaded or simulated during initialization, except for Bertrand-Nash prices and shares, which are
    computed by :meth:`Simulation.solve`.

    Unspecified exogenous variables that are used to formulate product characteristics, :math:`X_1`, :math:`X_2`, and
    :math:`X_3`, as well as agent demographics, :math:`d`, are all drawn from independent standard uniform
    distributions.

    Unobserved demand- and supply-side product characteristics, :math:`\xi` and :math:`\omega`, are drawn from a
    mean-zero bivariate normal distribution.

    After variables are loaded or simulated, any unspecified nodes and weights will be constructed according to an
    integration configuration. Next, some simple instruments are computed:

    .. math:: Z_D = [1, X, \mathrm{Rival}(X_D), \mathrm{Other}(X_D)]

    and

    .. math:: Z_S = [1, X, \mathrm{Rival}(X_S), \mathrm{Other}(X_S)],

    in which :math:`X` are all non-constant exogenous numerical product variables, :math:`X_D` are all variables in
    :math:`X` used to formulate :math:`X_1` and :math:`X_2`, :math:`X_S` are all variables in :math:`X` used to
    formulate :math:`X_3`, and both :math:`\mathrm{Rival}` and :math:`\mathrm{Other}` are defined in
    :func:`build_blp_instruments`, which is used to construct the traditional BLP instruments.

    .. note::

       These instruments are constructed only for convenience. Especially for more complicated formulations, instrument
       fields in simulated product data should be replaced with better instruments.

    Parameters
    ----------
    product_formulations : `tuple`
        Tuple of three :class:`Formulation` configurations for the matrix of linear product characteristics,
        :math:`X_1`, for the matrix of nonlinear product characteristics, :math:`X_2`, and for the matrix of cost
        characteristics, :math:`X_3`, respectively. The ``shares`` variable should be included in none of the
        formulations and ``prices`` should be included in the formulation for :math:`X_1` or :math:`X_2` (or both). Any
        additional variables that cannot be loaded from `product_data` will be drawn from independent standard uniform
        distributions.
    beta : `array-like`
        Vector of demand-side linear parameters, :math:`\beta`. Elements correspond to columns in :math:`X_1` configured
        by the first formulation in `product_formulations`.
    sigma : `array-like`
        Cholesky decomposition of the covariance matrix that measures agents' random taste distribution, :math:`\Sigma`,
        which is a square matrix with a lower triangle of all zeros. Rows and columns correspond to columns in
        :math:`X_2` configured by the second formulation in `product_formulations`.
    gamma : `array-like`
        Vector of supply-side linear parameters, :math:`\gamma`. Elements correspond to columns in :math:`X_3`
        configured by the third formulation in `product_formulations`.
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products.

        Fields with multiple columns can be either matrices or can be broken up into multiple one-dimensional fields
        with column index suffixes that start at zero. For example, if there are two columns of firm IDs, the `firm_ids`
        field, which in this case should be a matrix with two columns, can be replaced by two one-dimensional fields:
        `firm_ids0` and `firm_ids1`.

        The convenience function :func:`build_id_data` can be used to construct the following required ID data from
        market, product, and firm counts:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object`) - IDs that associate products with firms. Any columns after the first can be
              used in :meth:`Simulation.solve` to compute Bertrand-Nash prices and shares after firm changes, such as
              mergers.

        Custom ownership matrices can be specified as well:

            - **ownership** : (`numeric, optional') - Custom stacked :math:`J_t \times J_t` ownership matrices,
              :math:`O`, for each market :math:`t`, which can be built with :func:`build_ownership`. By default,
              standard ownership matrices are built only when they are needed. If specified, each stack is associated
              with a `firm_ids` column and must have as many columns as there are products in the market with the most
              products.

        Along with `market_ids` and `firm_ids`, the names of any additional fields can be used as variables
        in `product_formulations`.

    agent_formulation : `Formulation, optional`
        :class:`Formulation` configuration for the matrix of observed agent characteristics, :math:`d`, called
        demographics, which will only be included in the model if this formulation is specified. Any variables that
        cannot be loaded from `agent_data` will be drawn from independent standard uniform distributions.
    pi : `array-like, optional`
        Parameters that measure how agent tastes vary with demographics, :math:`\Pi`. Rows correspond to the same
        product characteristics as in `sigma`. Columns correspond to to columns in :math:`d` configured by
        `agent_formulation`.
    agent_data : `structured array-like, optional`
        Each row corresponds to an agent. Markets can have differing numbers of agents. The following field is required:

            - **market_ids** : (`object, optional`) - IDs that associate agents with markets. The set of distinct IDs
              should be the same as the set in `product_data`. If `integration` is specified, there must be at least as
              many rows in each market as the number of nodes and weights that are built for each market.

        If `integration` is not specified, the following fields are required:

            - **weights** : (`numeric, optional`) - Integration weights, :math:`w`.

            - **nodes** : (`numeric, optional`) - Unobserved agent characteristics called integration nodes,
              :math:`\nu`. If there are more than :math:`K_2` columns, only the first :math:`K_2` will be used.

        Along with `market_ids`, the names of any additional fields can be used as variables in `agent_formulation`.

    integration : `Integration, optional`
        :class:`Integration` configuration for how to build nodes and weights for integration over agent utilities,
        which will replace any `nodes` and `weights` fields in `agent_data`. This is required if `nodes` and `weights`
        in `agent_data` are not specified.
    xi_variance : `float, optional`
        Variance of the unobserved demand-side product characteristics, :math:`\xi`. The default value is ``1.0``.
    omega_variance : `float, optional`
        Variance of the unobserved supply-side product characteristics, :math:`\omega`. The default value is ``1.0``.
    correlation : `float, optional`
        Correlation between :math:`\xi` and :math:`\omega`. The default value is ``0.9``.
    linear_costs : `bool, optional`
        Whether to compute marginal costs, :math:`c`, according to a linear or a log-linear marginal cost specification.
        By default, a linear specification is used. That is, :math:`\tilde{c} = c` instead of
        :math:`\tilde{c} = \log c`.
    seed : `int, optional`
        Passed to :class:`numpy.random.RandomState` to seed the random number generator before data are simulated.

    Attributes
    ----------
    product_formulations : `tuple`
        Tuple of three :class:`Formulation` configurations for :math:`X_1`, :math:`X_2`, and :math:`X_3`.
    agent_formulation : `tuple`
        :class:`Formulation` configuration for :math:`d`.
    beta : `ndarray`
        Demand-side linear parameters, :math:`\beta`.
    sigma : `ndarray`
        Cholesky decomposition of the covariance matrix that measures agents' random taste distribution, :math:`\Sigma`.
    gamma : `ndarray`
        Supply-side linear parameters, :math:`\gamma`.
    pi : `ndarray`
        Parameters that measures how agent tastes vary with demographics, :math:`\Pi`.
    product_data : `recarray`
        Synthetic product data that were loaded or simulated during initialization, except for Bertrand-Nash prices and
        shares, which are computed by :meth:`Simulation.solve`.
    agent_data : `recarray`
        Synthetic agent data that were loaded or simulated during initialization.
    products : `Products`
        Structured :attr:`Simulation.product_data`, which is an instance of :class:`primitives.Products`. Matrices of
        product characteristics were built according to :attr:`Simulation.product_formulations`.
    agents : `Agents`
        Structured :attr:`Simulation.agent_data`, which is an instance of :class:`primitives.Agents`. A matrix of
        demographics was built according to :attr:`Simulation.agent_formulation` if it was specified. Nodes and weights
        were build according to :attr:`Simulation.integration` if it was specified.
    unique_market_ids : `ndarray`
        Unique market IDs in product and agent data.
    xi : `ndarray`
        Unobserved demand-side product characteristics, :math:`\xi`, that were simulated during initialization.
    omega : `ndarray`
        Unobserved supply-side product characteristics, :math:`\omega`, that were simulated during initialization.
    costs : `ndarray`
        Marginal costs, :math:`c`, that were simulated during initialization.
    integration : `Integration`
        :class:`Integration` configuration for how nodes and weights for integration over agent utilities were built
        during :class:`Simulation` initialization.
    linear_costs : `bool`
        Whether :attr:`Simulation.costs` were simulated according to a linear or a log-linear marginal cost
        specification during :class:`Simulation` initialization.
    N : `int`
        Number of products across all markets, :math:`N`.
    T : `int`
        Number of markets, :math:`T`.
    K1 : `int`
        Number of linear product characteristics, :math:`K_1`.
    K2 : `int`
        Number of nonlinear product characteristics, :math:`K_2`.
    K3 : `int`
        Number of cost product characteristics, :math:`K_3`.
    D : `int`
        Number of demographic variables, :math:`D`.
    MD : `int`
        Number of demand-side instruments, :math:`M_D`.
    MS : `int`
        Number of supply-side instruments, :math:`M_S`.
    ED : `int`
        Number of absorbed demand-side fixed effects, :math:`E_D`, which is always zero because simulations do not
        support fixed effect absorption.
    ES : `int`
        Number of absorbed supply-side fixed effects, :math:`E_S`, which is always zero because simulations do not
        support fixed effect absorption.

    Example
    -------
    The following code simulates a small amount of data for two markets. Exogenous product data, ``size`` and
    ``weight``, along with a demographic, ``income``, are simulated; unobserved agent data are constructed according to
    a low-level Gauss-Hermite product rule.

    .. ipython:: python

       simulation = pyblp.Simulation(
           product_formulations=(
               pyblp.Formulation('0 + prices + size'),
               pyblp.Formulation('prices'),
               pyblp.Formulation('0 + size + weight')
           ),
           beta=[-10, 1],
           sigma=[
               [2, 0],
               [0, 1]
           ],
           gamma=[1, 2],
           product_data=pyblp.build_id_data(T=2, J=20, F=5),
           agent_formulation=pyblp.Formulation('0 + income'),
           integration=pyblp.Integration('product', 4),
           pi=[
               [0],
               [1]
           ],
           seed=0
       )
       simulation
       simulation.agent_data
       simulation.product_data

    Bertrand-Nash prices and shares, which are initialized as zero above, can be computed by solving the simulation:

    .. ipython:: python

       product_data = simulation.solve()
       product_data

    """

    def __init__(self, product_formulations, beta, sigma, gamma, product_data, agent_formulation=None, pi=None,
                 agent_data=None, integration=None, xi_variance=1, omega_variance=1, correlation=0.9, linear_costs=True,
                 seed=None):
        """Load or simulate all data except for Bertrand-Nash prices and shares."""

        # validate the formulations
        if not all(isinstance(f, Formulation) for f in product_formulations) or len(product_formulations) != 3:
            raise TypeError("product_formulations must be a tuple of three Formulation instances.")
        if any(f._absorbed_terms for f in product_formulations):
            raise ValueError("product_formulations do not support fixed effect absorption in simulations.")
        if agent_formulation is not None and not isinstance(agent_formulation, Formulation):
            raise TypeError("agent_formulation must be a Formulation instance.")
        if agent_formulation is not None and agent_formulation._absorbed_terms:
            raise ValueError("agent_formulation does not support fixed effect absorption.")

        # load IDs
        market_ids = extract_matrix(product_data, 'market_ids')
        firm_ids = extract_matrix(product_data, 'firm_ids')
        if market_ids is None:
            raise KeyError("product_data must have a market_ids field.")
        if firm_ids is None:
            raise KeyError("product_data must have a firm_ids field.")
        if market_ids.shape[1] > 1:
            raise ValueError("The market_ids field of product_data must be one-dimensional.")

        # load ownership matrices
        ownership = extract_matrix(product_data, 'ownership')

        # set the seed before simulating data
        state = np.random.RandomState(seed)

        # load or simulate exogenous product variables (in sorted order so that seeds give rise to the same draws)
        numerical_mapping = {}
        categorical_mapping = {}
        for formulation in product_formulations:
            for name in sorted(formulation._names - set(numerical_mapping) - set(categorical_mapping) - {'prices'}):
                variable = extract_matrix(product_data, name)
                if variable is None:
                    variable = state.uniform(size=market_ids.size).astype(options.dtype)
                elif variable.shape[1] > 1:
                    raise ValueError(f"The {name} variable has a field in product_data with more than one column.")
                if np.issubdtype(variable.dtype, getattr(np, 'number')):
                    numerical_mapping[name] = variable
                else:
                    categorical_mapping[name] = variable

        # construct instruments
        instrument_data = {'market_ids': market_ids, 'firm_ids': firm_ids, **numerical_mapping}
        demand_names = set(numerical_mapping) & (product_formulations[0]._names | product_formulations[1]._names)
        supply_names = set(numerical_mapping) & product_formulations[2]._names
        only_demand_names = demand_names - supply_names
        only_supply_names = supply_names - demand_names
        demand_instruments = np.c_[
            build_matrix(Formulation(' + '.join(sorted(only_supply_names))), numerical_mapping),
            build_blp_instruments(Formulation(' + '.join(['0'] + sorted(demand_names))), instrument_data)
        ]
        supply_instruments = np.c_[
            build_matrix(Formulation(' + '.join(sorted(only_demand_names))), numerical_mapping),
            build_blp_instruments(Formulation(' + '.join(['0'] + sorted(supply_names))), instrument_data)
        ]

        # structure product data fields as a mapping
        product_data_mapping = {
            'market_ids': (market_ids, np.object),
            'firm_ids': (firm_ids, np.object),
            'shares': (np.zeros(market_ids.size), options.dtype),
            'prices': (np.zeros(market_ids.size), options.dtype),
            'demand_instruments': (demand_instruments, options.dtype),
            'supply_instruments': (supply_instruments, options.dtype)
        }
        if ownership is not None:
            product_data_mapping['ownership'] = (ownership, options.dtype)

        # supplement the mapping with exogenous product variables
        variable_mapping = {**numerical_mapping, **categorical_mapping}
        invalid_names = set(variable_mapping) & set(product_data_mapping)
        if invalid_names:
            raise NameError(f"These names in product_formulations are invalid: {list(invalid_names)}.")
        product_data_mapping.update({k: (v, options.dtype) for k, v in variable_mapping.items()})

        # structure product data
        self.product_data = Matrices(product_data_mapping)
        products = Products(product_formulations, self.product_data)

        # determine the number of agents by loading market IDs or by building them along with nodes and weights
        if integration is not None:
            if not isinstance(integration, Integration):
                raise ValueError("integration must be an Integration instance.")
            agent_market_ids, nodes, weights = integration._build_many(products.X2.shape[1], np.unique(market_ids))
        elif agent_data is not None:
            agent_market_ids = extract_matrix(agent_data, 'market_ids')
            nodes = weights = None
        else:
            raise ValueError("Either agent_data or integration (or both) must be specified.")

        # load or simulate agent variables (in sorted order so that seeds give rise to the same draws)
        agent_variable_mapping = {}
        if agent_formulation is not None:
            for name in sorted(agent_formulation._names - set(agent_variable_mapping)):
                variable = extract_matrix(agent_data, name) if agent_data is not None else None
                if variable is None:
                    variable = state.uniform(size=agent_market_ids.size).astype(options.dtype)
                agent_variable_mapping[name] = variable

        # structure agent data fields as a mapping
        agent_data_mapping = {'market_ids': (agent_market_ids, np.object)}
        if nodes is not None:
            agent_data_mapping['nodes'] = (nodes, options.dtype)
        if weights is not None:
            agent_data_mapping['weights'] = (weights, options.dtype)

        # supplement the mapping with agent variables
        invalid_names = set(agent_variable_mapping) & set(agent_data_mapping)
        if invalid_names:
            raise NameError(f"These names in agent_formulation are invalid: {list(invalid_names)}.")
        agent_data_mapping.update({k: (v, options.dtype) for k, v in agent_variable_mapping.items()})

        # structure agent data
        self.integration = integration
        self.agent_data = Matrices(agent_data_mapping)
        agents = Agents(products, agent_formulation, self.agent_data, integration)

        # initialize the underlying economy
        super().__init__(product_formulations, agent_formulation, products, agents)

        # validate parameters
        self._linear_parameters = LinearParameters(self, beta, gamma)
        self._nonlinear_parameters = NonlinearParameters(self, sigma, pi)
        self.beta = self._linear_parameters.beta
        self.gamma = self._linear_parameters.gamma
        self.sigma = self._nonlinear_parameters.sigma
        self.pi = self._nonlinear_parameters.pi if self.D > 0 else None

        # simulate xi and omega
        covariance = correlation * np.sqrt(xi_variance * omega_variance)
        variances = [[xi_variance, covariance], [covariance, omega_variance]]
        try:
            shocks = state.multivariate_normal([0, 0], variances, self.N, check_valid='raise').astype(options.dtype)
        except ValueError:
            raise ValueError("xi_variance, omega_variance, and covariance must furnish a positive-semidefinite matrix.")
        self.xi = shocks[:, [0]]
        self.omega = shocks[:, [1]]

        # compute marginal costs
        self.linear_costs = linear_costs
        self.costs = self.products.X3 @ self.gamma + self.omega
        if not linear_costs:
            self.costs = np.exp(self.costs)

    def __str__(self):
        """Supplement general formatted information with other information about parameters."""
        sections = [
            [super().__str__()],
            ["Linear Parameters:", self._linear_parameters.format()],
            ["Nonlinear Parameters:", self._nonlinear_parameters.format()]
        ]
        return "\n\n".join("\n".join(s) for s in sections)

    def solve(self, firms_index=0, prices=None, iteration=None, error_behavior='raise', processes=1):
        r"""Compute Bertrand-Nash prices and shares.

        Prices and shares are computed by iterating in each market over the :math:`\zeta`-markup equation from
        :ref:`Morrow and Skerlos (2011) <ms11>`,

        .. math:: p \leftarrow c + \zeta(p).

        Parameters
        ----------
        firms_index : `int, optional`
            Index of the column in the `firm_ids` field of `basic_product_data` in :class:`Simulation` initialization
            that defines which firms produce which products. If an `ownership` field was specified, the corresponding
            stack of ownership matrices will be used. By default, unchanged firm IDs are used.
        prices : `array-like, optional`
            Prices at which the fixed point iteration routine will start. By default, marginal costs, :math:`c`, are
            used as starting values.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem. By default,
            ``Iteration('simple', {'tol': 1e-12})`` is used.
        error_behavior : `str, optional`
            How to handle errors when computing prices and shares. For example, the fixed point routine may not converge
            if the effects of nonlinear parameters on price overwhelm the linear parameter on price, which should be
            sufficiently negative. The following behaviors are supported:

                - ``'raise'`` (default) - Raises an exception.

                - ``'warn'`` - Uses the last computed prices and shares. If the fixed point routine fails to converge,
                  these are the last prices and shares computed by the routine. If there are other issues, these are the
                  starting prices and their associated shares.

        processes : `int, optional`
            Number of Python processes that will be used during computation. By default, multiprocessing will not be
            used. For values greater than one, a pool of that many Python processes will be created. Market-by-market
            computation of prices and shares will be distributed among these processes. Using multiprocessing will only
            improve computation speed if gains from parallelization outweigh overhead from creating the process pool.

        Returns
        -------
        `recarray`
            Simulated :attr:`Simulation.product_data` that are updated with Bertrand-Nash prices and shares, which can
            be passed to the `product_data` argument of :class:`Problem` initialization.

        """

        # choose or validate initial prices
        if prices is None:
            prices = self.costs
        else:
            prices = np.c_[np.asarray(prices, options.dtype)]
            if prices.shape != (self.N, 1):
                raise ValueError(f"prices must be a vector with {self.N} elements.")

        # configure or validate integration
        if iteration is None:
            iteration = Iteration('simple', {'tol': 1e-12})
        elif not isinstance(iteration, Iteration):
            raise ValueError("iteration must be an Iteration instance.")

        # validate error behavior
        if error_behavior not in {'raise', 'warn'}:
            raise ValueError("error_behavior must be 'raise' or 'warn'.")

        # time how long it takes to solve for prices and shares
        output("Solving for prices and shares ...")
        start_time = time.time()

        # compute a baseline delta that will be updated when shares and prices are changed
        delta = self.products.X1 @ self.beta + self.xi

        # define a function builds a market along with market-specific arguments used to compute prices and shares
        def market_factory(s):
            market_s = SimulationMarket(self, s, self.sigma, self.pi, self.beta, delta)
            prices_s = prices[self.products.market_ids.flat == s]
            costs_s = self.costs[self.products.market_ids.flat == s]
            return market_s, iteration, firms_index, prices_s, costs_s

        # update prices and shares market-by-market
        errors = set()
        iterations = evaluations = 0
        updated_product_data = self.product_data.copy()
        with ParallelItems(self.unique_market_ids, market_factory, SimulationMarket.solve, processes) as items:
            for t, (prices_t, shares_t, errors_t, iterations_t, evaluations_t) in items:
                updated_product_data.prices[self.products.market_ids.flat == t] = prices_t
                updated_product_data.shares[self.products.market_ids.flat == t] = shares_t
                errors |= errors_t
                iterations += iterations_t
                evaluations += evaluations_t

        # handle any errors
        if errors:
            exception = exceptions.MultipleErrors(errors)
            if error_behavior == 'raise':
                raise exception
            if error_behavior == 'warn':
                output("")
                output(exception)
                output("Using the last computed prices and shares.")
                output("")

        # output a message about computation
        end_time = time.time()
        run_time = end_time - start_time
        output(
            f"Finished computing prices and shares after {output.format_seconds(run_time)}, a total of {iterations} "
            f"major iterations, and a total of {evaluations} contraction evaluations."
        )
        return updated_product_data


class SimulationMarket(Market):
    """A single market in a simulation, which can be used to solve for prices and shares."""

    def solve(self, iteration, firms_index=0, prices=None, costs=None):
        """Solve for prices and shares. By default, use unchanged firm IDs, use unchanged prices as starting values,
        and compute marginal costs.
        """

        # configure NumPy to identify floating point errors
        errors = set()
        with np.errstate(divide='call', over='call', under='ignore', invalid='call'):
            np.seterrcall(lambda *_: errors.add(exceptions.SyntheticPricesFloatingPointError))

            # solve the fixed point problem
            prices, converged, iterations, evaluations = self.compute_bertrand_nash_prices(
                iteration, firms_index, prices, costs
            )

            # compute the associated shares
            delta = self.update_delta_with_variable('prices', prices)
            mu = self.update_mu_with_variable('prices', prices)
            shares = self.compute_probabilities(delta, mu) @ self.agents.weights

        # determine whether the fixed point converged
        if not converged:
            errors.add(exceptions.SyntheticPricesConvergenceError)
        return prices, shares, errors, iterations, evaluations
