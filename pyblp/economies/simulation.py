"""Economy-level simulation of synthetic BLP data."""

import collections
import time
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .abstract_economy import AbstractEconomy
from .results.simulation_results import SimulationResults
from .. import exceptions, options
from ..configurations.formulation import Formulation
from ..configurations.integration import Integration
from ..configurations.iteration import Iteration
from ..construction import build_blp_instruments, build_matrix
from ..markets.simulation_market import SimulationMarket
from ..parameters import LinearParameters, NonlinearParameters
from ..primitives import Agents, Products
from ..utilities.basics import (
    Array, Data, Error, RecArray, extract_matrix, format_seconds, generate_items, output, output_progress,
    structure_matrices
)


class Simulation(AbstractEconomy):
    r"""Simulation of synthetic BLP data.

    All data are either loaded or simulated during initialization, except for synthetic prices and shares, which are
    computed by :meth:`Simulation.solve`.

    Unspecified exogenous variables that are used to formulate product characteristics in :math:`X_1`, :math:`X_2`, and
    :math:`X_3`, as well as agent demographics, :math:`d`, are all drawn from independent standard uniform
    distributions.

    Unobserved demand- and supply-side product characteristics, :math:`\xi` and :math:`\omega`, are drawn from a
    mean-zero bivariate normal distribution.

    After variables are loaded or simulated, any unspecified nodes and weights are constructed according to an
    integration configuration.

    Next, simple excluded demand-side instruments are constructed according to

    .. math:: [X_{S \setminus D}, \mathrm{BLP}(X)],

    in which :math:`X_{S \setminus D}` is all variables used to formulate :math:`X_3` that were not used to formulate
    :math:`X_1` and :math:`X_2`, :math:`X_D` is all variables used to formulate :math:`X_1` and :math:`X_2`, and
    :math:`\mathrm{BLP}(X)` is defined in :func:`build_blp_instruments`, which is used to construct traditional excluded
    BLP instruments.

    Similarly, simple excluded supply-side instruments are constructed according to

    .. math:: [X_{D \setminus S}, \mathrm{BLP}(X)],

    in which :math:`X_{D \setminus S}` is all variables used to formulate :math:`X_1` and :math:`X_2` that were not used
    to formulate :math:`X_3`, and :math:`X_S` is all variables used to formulate :math:`X_3`.

    .. note::

       These excluded instruments are constructed only for convenience. Especially for more complicated formulations,
       instruments in simulated product data should be replaced with better instruments.

    In both `product_data` and `agent_data`, fields with multiple columns can be either matrices or can be broken up
    into multiple one-dimensional fields with column index suffixes that start at zero. For example, if there are two
    columns of firm IDs, the `firm_ids` field, which in this case should be a matrix with two columns, can be replaced
    by two one-dimensional fields: `firm_ids0` and `firm_ids1`.

    Parameters
    ----------
    product_formulations : `tuple`
        Tuple of three :class:`Formulation` configurations for the matrix of linear product characteristics,
        :math:`X_1`, for the matrix of nonlinear product characteristics, :math:`X_2`, and for the matrix of cost
        characteristics, :math:`X_3`, respectively. If the formulation for :math:`X_2` is ``None``, the Logit model will
        be simulated.

        The ``shares`` variable should not be included in any of the formulations and ``prices`` should be included in
        the formulation for :math:`X_1` or :math:`X_2` (or both). Any additional variables that cannot be loaded from
        `product_data` will be drawn from independent standard uniform distributions. Unlike in :class:`Problem`, fixed
        effect absorption is not supported during simulation. All exogenous characteristics in :math:`X_2` should also
        be included in :math:`X_1`.

    beta : `array-like`
        Vector of demand-side linear parameters, :math:`\beta`. Elements correspond to columns in :math:`X_1`, which
        is formulated according to `product_formulations`.
    sigma : `array-like`
        Cholesky decomposition of the covariance matrix that measures agents' random taste distribution, :math:`\Sigma`,
        which is a square matrix with a lower triangle of all zeros. Rows and columns correspond to columns in
        :math:`X_2`, which is formulated according to `product_formulations`. If the formulation for :math:`X_2` is
        ``None``, this should be ``None`` as well.
    gamma : `array-like`
        Vector of supply-side linear parameters, :math:`\gamma`. Elements correspond to columns in :math:`X_3`, which
        is formulated according to `product_formulations`.
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The convenience function
        :func:`build_id_data` can be used to construct the following required ID data from market, product, and firm
        counts:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object`) - IDs that associate products with firms. Any columns after the first can be
              used in :meth:`Simulation.solve` to compute synthetic prices and shares after firm changes, such as
              mergers.

        Custom ownership matrices can be specified as well:

            - **ownership** : (`numeric, optional') - Custom stacked :math:`J_t \times J_t` ownership matrices,
              :math:`O`, for each market :math:`t`, which can be built with :func:`build_ownership`. By default,
              standard ownership matrices are built only when they are needed. If specified, each stack is associated
              with a `firm_ids` column and must have as many columns as there are products in the market with the most
              products.

        To simulate a nested Logit or random coefficients nested Logit (RCNL) model, nesting groups must be specified:

            - **nesting_ids** (`object, optional`) - IDs that associate products with nesting groups. When these IDs are
              specified, `rho` must be specified as well.

        Along with `market_ids`, `firm_ids`, and `nesting_ids`, the names of any additional fields can be used as
        variables in `product_formulations`.

    agent_formulation : `Formulation, optional`
        :class:`Formulation` configuration for the matrix of observed agent characteristics called demographics,
        :math:`d`, which will only be included in the model if this formulation is specified. Any variables that cannot
        be loaded from `agent_data` will be drawn from independent standard uniform distributions.
    pi : `array-like, optional`
        Parameters that measure how agent tastes vary with demographics, :math:`\Pi`. Rows correspond to the same
        product characteristics as in `sigma`. Columns correspond to to columns in :math:`d`, which is formulated
        according to `agent_formulation`.
    agent_data : `structured array-like, optional`
        Each row corresponds to an agent. Markets can have differing numbers of agents. Since simulated agents are only
        used if there are nonlinear product characteristics, agent data should only be specified if :math:`X_2` is
        formulated in `product_formulations`. If agent data are specified, market IDs are required:

            - **market_ids** : (`object, optional`) - IDs that associate agents with markets. The set of distinct IDs
              should be the same as the set of IDs in `product_data`. If `integration` is specified, there must be at
              least as many rows in each market as the number of nodes and weights that are built for each market.

        If `integration` is not specified, the following fields are required:

            - **weights** : (`numeric, optional`) - Integration weights, :math:`w`.

            - **nodes** : (`numeric, optional`) - Unobserved agent characteristics called integration nodes,
              :math:`\nu`. If there are more than :math:`K_2` columns, only the first :math:`K_2` will be used.

        Along with `market_ids`, the names of any additional fields can be used as variables in `agent_formulation`.

    integration : `Integration, optional`
        :class:`Integration` configuration for how to build nodes and weights for integration over agent utilities,
        which will replace any `nodes` and `weights` fields in `agent_data`. This configuration is required if `nodes`
        and `weights` in `agent_data` are not specified. It should not be specified if :math:`X_2` is not formulated
        in `product_formulations`.
    rho : `array-like, optional`
        Parameters that measure within nesting group correlation, :math:`\rho`. If there is only one element, it
        corresponds to all groups defined by the `nesting_ids` field of `product_data`. If there is more than one
        element, there must be as many elements as :math:`H`, the number of distinct nesting groups, and elements
        correspond to group IDs in the sorted order given by :attr:`Simulation.unique_nesting_ids`.
    xi_variance : `float, optional`
        Variance of the unobserved demand-side product characteristics, :math:`\xi`. The default value is ``1.0``.
    omega_variance : `float, optional`
        Variance of the unobserved supply-side product characteristics, :math:`\omega`. The default value is ``1.0``.
    correlation : `float, optional`
        Correlation between :math:`\xi` and :math:`\omega`. The default value is ``0.9``.
    costs_type : `str, optional`
        Marginal cost specification. The following specifications are supported:

            - ``'linear'`` (default) - Linear specification: :math:`\tilde{c} = c`.

            - ``'log'`` - Log-linear specification: :math:`\tilde{c} = \log c`.

    seed : `int, optional`
        Passed to :class:`numpy.random.RandomState` to seed the random number generator before data are simulated. By
        default, a seed is not passed to the random number generator.

    Attributes
    ----------
    product_formulations : `tuple`
        :class:`Formulation` configurations for :math:`X_1`, :math:`X_2`, and :math:`X_3`, respectively.
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
    rho : `ndarray`
        Parameters that measure within nesting group correlation, :math:`\rho`.
    product_data : `recarray`
        Synthetic product data that were loaded or simulated during initialization, except for synthetic prices and
        shares, which are computed by :meth:`Simulation.solve`.
    agent_data : `recarray`
        Synthetic agent data that were loaded or simulated during initialization.
    integration : `Integration`
        :class:`Integration` configuration for how any nodes and weights were built during initialization.
    products : `Products`
        Product data structured as :class:`Products`, which consists of data taken from :attr:`Simulation.product_data`
        along with matrices build according to :attr:`Simulation.product_formulations`.
    agents : `Agents`
        Agent data structured as :class:`Agents`, which consists of data taken from :attr:`Simulation.agent_data` or
        built by :attr:`Simulation.integration` along with any demographics formulated by
        :attr:`Simulation.agent_formulation`.
    unique_market_ids : `ndarray`
        Unique market IDs in product and agent data.
    unique_nesting_ids : `ndarray`
        Unique nesting IDs in product data.
    xi : `ndarray`
        Unobserved demand-side product characteristics, :math:`\xi`, that were simulated during initialization.
    omega : `ndarray`
        Unobserved supply-side product characteristics, :math:`\omega`, that were simulated during initialization.
    costs : `ndarray`
        Marginal costs, :math:`c`, that were simulated during initialization.
    costs_type : `str`
        The specification according to which :attr:`Simulation.costs` were simulated during initialization.
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
        Number of demand-side instruments, :math:`M_D`, which is the number of excluded demand-side instruments plus
        :math:`K_1 - K_1^p`.
    MS : `int`
        Number of supply-side instruments, :math:`M_S`, which is the number of excluded supply-side instruments plus
        :math:`K_3`.
    ED : `int`
        Number of absorbed demand-side fixed effects, :math:`E_D`, which is always zero because simulations do not
        support fixed effect absorption.
    ES : `int`
        Number of absorbed supply-side fixed effects, :math:`E_S`, which is always zero because simulations do not
        support fixed effect absorption.
    H : `int`
        Number of nesting groups, :math:`H`.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    beta: Array
    sigma: Array
    gamma: Array
    pi: Array
    rho: Array
    product_data: RecArray
    agent_data: Optional[RecArray]
    integration: Optional[Integration]
    xi: Array
    omega: Array
    costs: Array
    costs_type: str
    _linear_parameters: LinearParameters
    _nonlinear_parameters: NonlinearParameters

    def __init__(
            self, product_formulations: Sequence[Optional[Formulation]], beta: Any, sigma: Any, gamma: Any,
            product_data: Mapping, agent_formulation: Optional[Formulation] = None, pi: Optional[Any] = None,
            agent_data: Optional[Mapping] = None, integration: Optional[Integration] = None, rho: Optional[Any] = None,
            xi_variance: float = 1, omega_variance: float = 1, correlation: float = 0.9, costs_type: str = 'linear',
            seed: Optional[int] = None) -> None:
        """Load or simulate all data except for synthetic prices and shares."""

        # keep track of long it takes to initialize the simulation
        output("Initializing the simulation ...")
        start_time = time.time()

        # validate the product formulations
        if not isinstance(product_formulations, collections.Sequence) or len(product_formulations) != 3:
            raise TypeError("product_formulations must be a tuple of three formulations.")
        if not all(f is None or isinstance(f, Formulation) for f in product_formulations):
            raise TypeError("Each formulation in product_formulations must be None or a Formulation instance.")
        if product_formulations[0] is None:
            raise ValueError("The formulation for X1 must be specified.")
        if product_formulations[2] is None:
            raise ValueError("The formulation for X3 must be specified.")
        if any(f._absorbed_terms for f in product_formulations if f is not None):
            raise ValueError("product_formulations do not support fixed effect absorption in simulations.")

        # validate the agent formulation
        if agent_formulation is not None:
            if not isinstance(agent_formulation, Formulation):
                raise TypeError("agent_formulation must be None or a Formulation instance.")
            if agent_formulation._absorbed_terms:
                raise ValueError("agent_formulation does not support fixed effect absorption.")

        # load IDs
        market_ids = extract_matrix(product_data, 'market_ids')
        firm_ids = extract_matrix(product_data, 'firm_ids')
        nesting_ids = extract_matrix(product_data, 'nesting_ids')
        clustering_ids = extract_matrix(product_data, 'clustering_ids')
        if market_ids is None:
            raise KeyError("product_data must have a market_ids field.")
        if firm_ids is None:
            raise KeyError("product_data must have a firm_ids field.")
        if market_ids.shape[1] > 1:
            raise ValueError("The market_ids field of product_data must be one-dimensional.")

        # load ownership matrices
        ownership = extract_matrix(product_data, 'ownership')

        # seed the random number generator
        state = np.random.RandomState(seed)

        # load or simulate exogenous product variables in sorted order so that a seed always furnishes the same draws
        numerical_mapping: Data = {}
        categorical_mapping: Data = {}
        for formulation in product_formulations:
            if formulation is not None:
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

        # identify demand-side numerical variables
        demand_names = set(numerical_mapping) & product_formulations[0]._names
        if product_formulations[1] is not None:
            demand_names |= set(numerical_mapping) & product_formulations[1]._names
        if not demand_names:
            raise ValueError("The formulations for X1 and X2 must have at least one non-price numerical variable.")

        # identify supply-side numerical variables
        supply_names = set(numerical_mapping) & product_formulations[2]._names
        if not supply_names:
            raise ValueError("The formulation for X3 must have at least one numerical variable.")

        # identify numerical variables that are only on either the demand or supply side
        only_demand_names = demand_names - supply_names
        only_supply_names = supply_names - demand_names

        # construct excluded instruments
        instrument_data = {
            'market_ids': market_ids,
            'firm_ids': firm_ids,
            **numerical_mapping
        }
        demand_instruments = np.c_[
            build_matrix(Formulation(' + '.join(['0'] + sorted(only_supply_names))), numerical_mapping),
            build_blp_instruments(Formulation(' + '.join(['0'] + sorted(demand_names))), instrument_data)
        ]
        supply_instruments = np.c_[
            build_matrix(Formulation(' + '.join(['0'] + sorted(only_demand_names))), numerical_mapping),
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
        if nesting_ids is not None:
            product_data_mapping['nesting_ids'] = (nesting_ids, np.object)
        if clustering_ids is not None:
            product_data_mapping['clustering_ids'] = (clustering_ids, np.object)
        if ownership is not None:
            product_data_mapping['ownership'] = (ownership, options.dtype)

        # supplement the mapping with exogenous product variables
        variable_mapping = {k: (v, options.dtype) for k, v in {**numerical_mapping, **categorical_mapping}.items()}
        invalid_names = set(variable_mapping) & set(product_data_mapping)
        if invalid_names:
            raise NameError(f"These names in product_formulations are invalid: {list(invalid_names)}.")
        self.product_data = structure_matrices({**product_data_mapping, **variable_mapping})

        # structure product data
        products = Products(product_formulations, self.product_data)

        # if there are only linear characteristics, agents should not be specified
        K2 = products.X2.shape[1]
        if K2 == 0:
            if agent_formulation is not None or agent_data is not None or integration is not None:
                raise ValueError(
                    "Since X2 is not formulated, none of agent_formulation, agent_data, and integration should be "
                    "specified."
                )
            self.integration = self.agent_data = None
        else:
            # determine the number of agents by loading market IDs or by building them along with nodes and weights
            if integration is not None:
                if not isinstance(integration, Integration):
                    raise ValueError("integration must be None or an Integration instance.")
                agent_market_ids, nodes, weights = integration._build_many(K2, np.unique(market_ids))
            elif agent_data is not None:
                agent_market_ids = extract_matrix(agent_data, 'market_ids')
                nodes = weights = None
            else:
                raise ValueError("At least one of agent_data and integration must be specified.")

            # load or simulate agent variables (in sorted order so that seeds give rise to the same draws)
            agent_numerical_mapping: Data = {}
            if agent_formulation is not None:
                for name in sorted(agent_formulation._names - set(agent_numerical_mapping)):
                    variable = extract_matrix(agent_data, name) if agent_data is not None else None
                    if variable is None:
                        variable = state.uniform(size=agent_market_ids.size).astype(options.dtype)
                    agent_numerical_mapping[name] = variable

            # structure agent data fields as a mapping
            agent_data_mapping = {'market_ids': (agent_market_ids, np.object)}
            if nodes is not None:
                agent_data_mapping['nodes'] = (nodes, options.dtype)
            if weights is not None:
                agent_data_mapping['weights'] = (weights, options.dtype)

            # supplement the mapping with agent variables
            agent_variable_mapping = {k: (v, options.dtype) for k, v in agent_numerical_mapping.items()}
            invalid_names = set(agent_variable_mapping) & set(agent_data_mapping)
            if invalid_names:
                raise NameError(f"These names in agent_formulation are invalid: {list(invalid_names)}.")
            self.integration = integration
            self.agent_data = structure_matrices({**agent_data_mapping, **agent_variable_mapping})

        # structure agent data
        agents = Agents(products, agent_formulation, self.agent_data, integration)

        # initialize the underlying economy
        super().__init__(product_formulations, agent_formulation, products, agents)

        # validate that all exogenous characteristics in X2 are also in X1
        for column_formulation in self._X2_formulations:
            if 'prices' not in column_formulation.names and column_formulation not in self._X1_formulations:
                raise ValueError(f"'{column_formulation}' in the formulation for X2 is not in the formulation for X1.")

        # validate parameters
        self._linear_parameters = LinearParameters(self, beta, gamma)
        self._nonlinear_parameters = NonlinearParameters(self, sigma, pi, rho)
        self.beta = self._linear_parameters.beta
        self.gamma = self._linear_parameters.gamma
        self.sigma = self._nonlinear_parameters.sigma
        self.pi = self._nonlinear_parameters.pi
        self.rho = self._nonlinear_parameters.rho

        # simulate xi and omega
        covariance = correlation * np.sqrt(xi_variance * omega_variance)
        covariances = [[xi_variance, covariance], [covariance, omega_variance]]
        try:
            errors = state.multivariate_normal([0, 0], covariances, self.N, check_valid='raise').astype(options.dtype)
        except ValueError:
            raise ValueError("xi_variance, omega_variance, and correlation must give a positive-semidefinite matrix.")
        self.xi = errors[:, [0]]
        self.omega = errors[:, [1]]

        # compute marginal costs
        if costs_type not in {'linear', 'log'}:
            raise ValueError("costs_type must be 'linear' or 'log'.")
        self.costs_type = costs_type
        self.costs = self.products.X3 @ self.gamma + self.omega
        if costs_type == 'log':
            self.costs = np.exp(self.costs)

        # output information about the initialized simulation
        output(f"Initialized the simulation after {format_seconds(time.time() - start_time)}.")
        output("")
        output(self)

    def __str__(self) -> str:
        """Supplement general formatted information with other information about parameters."""
        sections = [[super().__str__()], ["Linear Parameters:", self._linear_parameters.format()]]
        if self.K2 > 0 or self.H > 0:
            sections.append(["Nonlinear Parameters:", self._nonlinear_parameters.format()])
        return "\n\n".join("\n".join(s) for s in sections)

    def solve(
            self, firms_index: int = 0, prices: Optional[Any] = None, iteration: Optional[Iteration] = None,
            error_behavior: str = 'raise') -> SimulationResults:
        r"""Compute synthetic prices and shares.

        Prices and shares are computed by iterating market-by-market over the :math:`\zeta`-markup equation from
        :ref:`Morrow and Skerlos (2011) <ms11>`,

        .. math:: p^* \leftarrow c + \zeta(p^*).

        .. note::

           This method supports :func:`parallel` processing. If multiprocessing is used, market-by-market computation of
           prices and shares will be distributed among the processes.

        Parameters
        ----------
        firms_index : `int, optional`
            Column index of the firm IDs in the `firm_ids` field of `product_data` in :class:`Simulation` that defines
            which firms produce which products. If an `ownership` field was specified, the corresponding stack of
            ownership matrices will be used. By default, unchanged firm IDs are used.
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

                - ``'raise'`` (default) - Raise an exception.

                - ``'warn'`` - Uses the last computed prices and shares. If the fixed point routine fails to converge,
                  these are the last prices and shares computed by the routine. If there are other issues, these are the
                  starting prices and their associated shares.

        Returns
        -------
        `SimulationResults`
            :class:`SimulationResults` of the solved simulation.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        errors: List[Error] = []

        # keep track of long it takes to solve for prices and shares
        output("Computing synthetic prices and shares ...")
        start_time = time.time()

        # validate the firms index
        self._validate_firms_index(firms_index)

        # choose or validate initial prices
        if prices is None:
            prices = self.costs
        else:
            prices = np.c_[np.asarray(prices, options.dtype)]
            if prices.shape != (self.N, 1):
                raise ValueError(f"prices must None or a {self.N}-vector.")

        # configure or validate integration
        if iteration is None:
            iteration = Iteration('simple', {'tol': 1e-12})
        elif not isinstance(iteration, Iteration):
            raise ValueError("iteration must be None or an Iteration.")

        # validate error behavior
        if error_behavior not in {'raise', 'warn'}:
            raise ValueError("error_behavior must be 'raise' or 'warn'.")

        # compute a baseline delta that will be updated when shares and prices are changed
        delta = self.products.X1 @ self.beta + self.xi

        # define a factory for solving simulation markets
        def market_factory(s: Hashable) -> Tuple[SimulationMarket, Array, Array, Iteration, int]:
            """Build a market along with arguments used to compute prices and shares."""
            assert prices is not None and iteration is not None
            market_s = SimulationMarket(self, s, self.sigma, self.pi, self.rho, self.beta, delta)
            costs_s = self.costs[self._product_market_indices[s]]
            prices_s = prices[self._product_market_indices[s]]
            return market_s, costs_s, prices_s, iteration, firms_index

        # compute prices and shares market-by-market
        iteration_mapping: Dict[Hashable, int] = {}
        evaluation_mapping: Dict[Hashable, int] = {}
        synthetic_prices = np.full_like(self.products.prices, np.nan)
        synthetic_shares = np.full_like(self.products.shares, np.nan)
        generator = output_progress(
            generate_items(self.unique_market_ids, market_factory, SimulationMarket.solve), self.T, start_time
        )
        for t, (prices_t, shares_t, errors_t, iterations_t, evaluations_t) in generator:
            synthetic_prices[self._product_market_indices[t]] = prices_t
            synthetic_shares[self._product_market_indices[t]] = shares_t
            errors.extend(errors_t)
            iteration_mapping[t] = iterations_t
            evaluation_mapping[t] = evaluations_t

        # handle any errors
        if errors:
            if error_behavior == 'raise':
                raise exceptions.MultipleErrors(errors)
            assert error_behavior == 'warn'
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # structure the results
        results = SimulationResults(
            self, firms_index, synthetic_prices, synthetic_shares, start_time, time.time(), iteration_mapping,
            evaluation_mapping
        )
        output(f"Computed synthetic prices and shares after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results
