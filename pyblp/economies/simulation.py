"""Economy-level simulation of synthetic BLP data."""

import collections.abc
import time
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .economy import Economy
from .. import exceptions
from .. import options
from ..configurations.formulation import Formulation
from ..configurations.integration import Integration
from ..configurations.iteration import Iteration
from ..markets.simulation_market import SimulationMarket
from ..parameters import Parameters
from ..primitives import Agents, Products
from ..results.simulation_results import SimulationResults
from ..utilities.algebra import precisely_compute_eigenvalues
from ..utilities.basics import (
    Array, Bounds, Error, Groups, SolverStats, RecArray, extract_matrix, format_seconds, generate_items, output,
    output_progress, structure_matrices
)


class Simulation(Economy):
    r"""Simulation of data in BLP-type models.

    Any data left unspecified are simulated during initialization. Simulated prices and shares can be replaced by
    :meth:`Simulation.replace_endogenous` with equilibrium values that are consistent with true parameters. Less
    commonly, simulated exogenous variables can be replaced instead by :meth:`Simulation.replace_exogenous`. To choose
    your own prices, refer to the first note in :meth:`Simulation.replace_endogenous`. Simulations
    are typically used for two purposes:

        1. Solving for equilibrium prices and shares under more complicated counterfactuals than is possible with
           :meth:`ProblemResults.compute_prices` and :meth:`ProblemResults.compute_shares`. For example, this class
           can be initialized with estimated parameters, structural errors, and marginal costs from a
           :meth:`ProblemResults`, but with changed data (fewer products, new products, different characteristics, etc.)
           and :meth:`Simulation.replace_endogenous` can be used to compute the corresponding prices and shares.

        2. Simulation of BLP-type models from scratch. For example, a model with fixed true parameters can be simulated
           many times, converted into problems with :meth:`SimulationResults.to_problem`, and solved with
           :meth:`Problem.solve` to evaluate in a Monte Carlo study how well the true parameters can be recovered.

    If data for variables (used to formulate product characteristics in :math:`X_1`, :math:`X_2`, and :math:`X_3`, as
    well as agent demographics, :math:`d`, and endogenous prices and market shares :math:`p` and :math:`s`) are not
    provided, the values for each unspecified variable are drawn independently from the standard uniform distribution.
    In each market :math:`t`, market shares are divided by the number of products in the market :math:`J_t`. Typically,
    :meth:`Simulation.replace_endogenous` is used to replace prices and shares with equilibrium values that are
    consistent with true parameters.

    If data for unobserved demand-and supply-side product characteristics, :math:`\xi` and :math:`\omega`, are not
    provided, they are by default drawn from a mean-zero bivariate normal distribution.

    After variables are loaded or simulated, any unspecified integration nodes and weights, :math:`\nu` and :math:`w`,
    are constructed according to a specified :class:`Integration` configuration.

    Parameters
    ----------
    product_formulations : `Formulation or sequence of Formulation`
        :class:`Formulation` configuration or a sequence of up to three :class:`Formulation` configurations for the
        matrix of demand-side linear product characteristics, :math:`X_1`, for the matrix of demand-side nonlinear
        product characteristics, :math:`X_2`, and for the matrix of supply-side characteristics, :math:`X_3`,
        respectively. If the formulation for :math:`X_2` is not specified or is ``None``, the logit (or nested logit)
        model will be simulated.

        The ``shares`` variable should not be included in the formulations for :math:`X_1` or :math:`X_2`. If ``shares``
        is included in the formulation for :math:`X_3` and ``product_data`` does not include ``shares``, one will likely
        want to set ``constant_costs=False`` in :meth:`Simulation.replace_endogenous`.

        The ``prices`` variable should not be included in the formulation for :math:`X_3`, but it should be included in
        the formulation for :math:`X_1` or :math:`X_2` (or both). Variables that cannot be loaded from ``product_data``
        will be drawn from independent standard uniform distributions. Unlike in :class:`Problem`, fixed effect
        absorption is not supported during simulation.

        .. warning::

           Characteristics that involve prices, :math:`p`, or shares, :math:`s`, should always be formulated with the
           ``prices`` and ``shares`` variables, respectively. If another name is used, :class:`Simulation` will not
           understand that the characteristic is endogenous. For example, to include a :math:`p^2` characteristic,
           include ``I(prices**2)`` in a formula instead of manually constructing and including a ``prices_squared``
           variable.

    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The convenience function
        :func:`build_id_data` can be used to construct the following required ID data:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object`) - IDs that associate products with firms.

        Custom ownership matrices can be specified as well:

            - **ownership** : (`numeric, optional`) - Custom stacked :math:`J_t \times J_t` ownership or product holding
              matrices, :math:`\mathscr{H}`, for each market :math:`t`, which can be built with :func:`build_ownership`.
              By default, standard ownership matrices are built only when they are needed to reduce memory usage. If
              specified, there should be as many columns as there are products in the market with the most products.
              Rightmost columns in markets with fewer products will be ignored.

        .. note::

           The ``ownership`` field can either be a matrix or can be broken up into multiple one-dimensional fields with
           column index suffixes that start at zero. For example, if there are three products in each market, a
           ``ownership`` field with three columns can be replaced by three one-dimensional fields: ``ownership0``,
           ``ownership1``, and ``ownership2``.

        It may be convenient to define IDs for different products:

            - **product_ids** (`object, optional`) - IDs that identify products within markets. There can be multiple
              columns.

        To simulate a nested logit or random coefficients nested logit (RCNL) model, nesting groups must be specified:

            - **nesting_ids** (`object, optional`) - IDs that associate products with nesting groups. When these IDs are
              specified, ``rho`` must be specified as well.

        Along with ``market_ids``, ``firm_ids``, ``product_ids``, and ``nesting_ids``, the names of any additional
        fields can typically be used as variables in ``product_formulations``. However, there are a few variable names
        such as ``'X1'``, which are reserved for use by :class:`Products`.

    beta : `array-like`
        Vector of demand-side linear parameters, :math:`\beta`. Elements correspond to columns in :math:`X_1`, which
        is formulated by ``product_formulations``.
    sigma : `array-like, optional`
        Lower-triangular Cholesky root of the covariance matrix for unobserved taste heterogeneity, :math:`\Sigma`. Rows
        and columns correspond to columns in :math:`X_2`, which is formulated by ``product_formulations``. If
        :math:`X_2` is not formulated, this should not be specified, since the logit model will be simulated.
    pi : `array-like, optional`
        Parameters that measure how agent tastes vary with demographics, :math:`\Pi`. Rows correspond to the same
        product characteristics as in ``sigma``. Columns correspond to columns in :math:`d`, which is formulated by
        ``agent_formulation``. If :math:`d` is not formulated, this should not be specified.
    gamma : `array-like, optional`
        Vector of supply-side linear parameters, :math:`\gamma`. Elements correspond to columns in :math:`X_3`, which
        is formulated by ``product_formulations``. If :math:`X_3` is not formulated, this should not be specified.
    rho : `array-like, optional`
        Parameters that measure within nesting group correlation, :math:`\rho`. If this is a scalar, it corresponds to
        all groups defined by the ``nesting_ids`` field of ``product_data``. If this is a vector, it must have :math:`H`
        elements, one for each nesting group. Elements correspond to group IDs in the sorted order of
        :attr:`Simulation.unique_nesting_ids`. If nesting IDs are not specified, this should not be specified either.
    agent_formulation : `Formulation, optional`
        :class:`Formulation` configuration for the matrix of observed agent characteristics called demographics,
        :math:`d`, which will only be included in the model if this formulation is specified. Any variables that cannot
        be loaded from ``agent_data`` will be drawn from independent standard uniform distributions.
    agent_data : `structured array-like, optional`
        Each row corresponds to an agent. Markets can have differing numbers of agents. Since simulated agents are only
        used if there are demand-side nonlinear product characteristics, agent data should only be specified if
        :math:`X_2` is formulated in ``product_formulations``. If agent data are specified, market IDs are required:

            - **market_ids** : (`object, optional`) - IDs that associate agents with markets. The set of distinct IDs
              should be the same as the set in ``product_data``. If ``integration`` is specified, there must be at least
              as many rows in each market as the number of nodes and weights that are built for the market.

        If ``integration`` is not specified, the following fields are required:

            - **weights** : (`numeric, optional`) - Integration weights, :math:`w`, for integration over agent choice
              probabilities.

            - **nodes** : (`numeric, optional`) - Unobserved agent characteristics called integration nodes,
              :math:`\nu`. If there are more than :math:`K_2` columns (the number of demand-side nonlinear product
              characteristics), only the first :math:`K_2` will be used. If any columns of ``sigma`` are fixed at zero,
              only the first few columns of these nodes will be used.

        The convenience function :func:`build_integration` can be useful when constructing custom nodes and weights.

        .. note::

           If ``nodes`` has multiple columns, it can be specified as a matrix or broken up into multiple one-dimensional
           fields with column index suffixes that start at zero. For example, if there are three columns of nodes, a
           ``nodes`` field with three columns can be replaced by three one-dimensional fields: ``nodes0``, ``nodes1``,
           and ``nodes2``.

        It may be convenient to define IDs for different agents:

            - **agent_ids** (`object, optional`) - IDs that identify agents within markets. There can be multiple of the
              same ID within a market.

        Along with ``market_ids`` and ``agent_ids``, the names of any additional fields can typically be used as
        variables in ``agent_formulation``. The exception is the name ``'demographics'``, which is reserved for use by
        :class:`Agents`.

        In addition to standard demographic variables :math:`d_{it}`, it is also possible to specify product-specific
        demographics :math:`d_{ijt}`. A typical example is geographic distance of agent :math:`i` from product
        :math:`j`. If ``agent_formulation`` has, for example, ``'distance'``, instead of including a single
        ``'distance'`` field in ``agent_data``, one should instead include ``'distance0'``, ``'distance1'``,
        ``'distance2'`` and so on, where the index corresponds to the order in which products appear within market in
        ``product_data``. For example, ``'distance5'`` should measure the distance of agents to the fifth product within
        the market, as ordered in ``product_data``. The last index should be the number of products in the largest
        market, minus one. For markets with fewer products than this maximum number, latter columns will be ignored.

        Finally, by default each agent :math:`i` in market :math:`t` is faced with the same choice set of product
        :math:`j`, but it is possible to specify agent-specific availability :math:`a_{ijt}` much in the same way that
        product-specific demographics are specified. To do so, the following field can be specified:

            - **availability** : (`numeric, optional`) - Agent-specific product availability, :math:`a`. Choice
              probabilities in :eq:`probabilities` are modified according to

              .. math:: s_{ijt} = \frac{a_{ijt} \exp V_{ijt}}{1 + \sum_{k \in J_t} a_{ijt} \exp V_{ikt}},

              and similarly for the nested logit model and consumer surplus calculations. By default, all
              :math:`a_{ijt} = 1`. To have a product :math:`j` be unavailable to agent :math:`i`, set
              :math:`a_{ijt} = 0`.

              Agent-specific availability is specified in the same way that product-specific demographics are specified.
              In ``agent_data``, one can include ``'availability0'``, ``'availability1'``, ``'availability2'``, and so
              on, where the index corresponds to the order in which products appear within market in ``product_data``.
              The last index should be the number of products in the largest market, minus one. For markets with fewer
              products than this maximum number, latter columns will be ignored.

    integration : `Integration, optional`
        :class:`Integration` configuration for how to build nodes and weights for integration over agent choice
        probabilities, which will replace any ``nodes`` and ``weights`` fields in ``agent_data``. This configuration is
        required if ``nodes`` and ``weights`` in ``agent_data`` are not specified. It should not be specified if
        :math:`X_2` is not formulated in ``product_formulations``.

        If this configuration is specified, :math:`K_2` columns of nodes (the number of demand-side nonlinear product
        characteristics) will be built. However, if ``sigma`` is left unspecified or is specified with columns fixed at
        zero, fewer columns will be used.

    xi : `array-like, optional`
        Unobserved demand-side product characteristics, :math:`\xi`. By default, if :math:`X_3` is formulated, each pair
        of unobserved characteristics in this vector and :math:`\omega` is drawn from a mean-zero bivariate normal
        distribution. This must be specified if :math:`X_3` is not formulated or if ``omega`` is specified.
    omega : `array-like, optional`
        Unobserved supply-side product characteristics, :math:`\omega`. By default, if :math:`X_3` is formulated, each
        pair of unobserved characteristics in this vector and :math:`\xi` is drawn from a mean-zero bivariate normal
        distribution. This must be specified if :math:`X_3` is formulated and ``xi`` is specified. It is ignored if
        :math:`X_3` is not formulated.
    xi_variance : `float, optional`
        Variance of :math:`\xi`. The default value is ``1.0``. This is ignored if ``xi`` or ``omega`` is specified.
    omega_variance : `float, optional`
        Variance of :math:`\omega`. The default value is ``1.0``. This is ignored if ``xi`` or ``omega`` is specified.
    correlation : `float, optional`
        Correlation between :math:`\xi` and :math:`\omega`. The default value is ``0.9``. This is ignored if ``xi`` or
        ``omega`` is specified.
    rc_types : `sequence of str, optional`
        Random coefficient types:

            - ``'linear'`` (default) - The random coefficient is as defined in :eq:`mu`.

            - ``'log'`` - The random coefficient's column in :eq:`mu` is exponentiated before being pre-multiplied by
              :math:`X_2`. It will take on values bounded from below by zero.

            - ``'logit'`` - The random coefficient's column in :eq:`mu` is passed through the inverse logit function
              before being pre-multiplied by :math:`X_2`. It will take on values bounded from below by zero and above by
              one.

        The list should have as many strings as there are columns in :math:`X_2`. Each string determines the type of the
        random coefficient on the corresponding product characteristic in :math:`X_2`.

        A typical example of when to use ``'log'`` is to have a lognormal coefficient on prices. Implementing this
        typically involves having an ``I(-prices)`` in the formulation for :math:`X_2`, and instead of including
        ``prices`` in :math:`X_1`, including a ``1`` in the ``agent_formulation``. Then the corresponding coefficient in
        :math:`\Pi` will serve as the mean parameter for the lognormal random coefficient on negative
        prices, :math:`-p_{jt}`.

    epsilon_scale : `float, optional`
        Factor by which the Type I Extreme Value idiosyncratic preference term, :math:`\epsilon_{ijt}`, is scaled. By
        default, :math:`\epsilon_{ijt}` is not scaled. The typical use of this parameter is to approximate the pure
        characteristics model of :ref:`references:Berry and Pakes (2007)` by choosing a value smaller than ``1.0``. As
        this scaling factor approaches zero, the model approaches the pure characteristics model in which there is no
        idiosyncratic preference term.

        For more information about choosing this parameter and estimating models where it is smaller than ``1.0``, refer
        to the same argument in :meth:`Problem.solve`. In some situations, it may be easier to solve simulations with
        small epsilon scaling factors by using :meth:`Simulation.replace_exogenous` rather than
        :meth:`Simulation.replace_endogenous`.

    costs_type : `str, optional`
        Specification of the marginal cost function :math:`\tilde{c} = f(c)` in :eq:`costs`. The following
        specifications are supported:

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
    product_data : `recarray`
        Synthetic product data that were loaded or simulated during initialization. Typically,
        :meth:`Simulation.replace_endogenous` is used replace prices and shares with equilibrium values that are
        consistent with true parameters. The :func:`data_to_dict` function can be used to convert this into a more
        usable data type.
    agent_data : `recarray`
        Synthetic agent data that were loaded or simulated during initialization. The :func:`data_to_dict` function can
        be used to convert this into a more usable data type.
    integration : `Integration`
        :class:`Integration` configuration for how any nodes and weights were built during initialization.
    products : `Products`
        Product data structured as :class:`Products`, which consists of data taken from :attr:`Simulation.product_data`
        along with matrices build according to :attr:`Simulation.product_formulations`. The :func:`data_to_dict`
        function can be used to convert this into a more usable data type.
    agents : `Agents`
        Agent data structured as :class:`Agents`, which consists of data taken from :attr:`Simulation.agent_data` or
        built by :attr:`Simulation.integration` along with any demographics formulated by
        :attr:`Simulation.agent_formulation`. The :func:`data_to_dict` function can be used to convert this into a more
        usable data type.
    unique_market_ids : `ndarray`
        Unique market IDs in product and agent data.
    unique_firm_ids : `ndarray`
        Unique firm IDs in product data.
    unique_nesting_ids : `ndarray`
        Unique nesting IDs in product data.
    unique_product_ids : `ndarray`
        Unique product IDs in product data.
    unique_agent_ids : `ndarray`
        Unique agent IDs in agent data.
    beta : `ndarray`
        Demand-side linear parameters, :math:`\beta`.
    sigma : `ndarray`
        Cholesky root of the covariance matrix for unobserved taste heterogeneity, :math:`\Sigma`.
    gamma : `ndarray`
        Supply-side linear parameters, :math:`\gamma`.
    pi : `ndarray`
        Parameters that measures how agent tastes vary with demographics, :math:`\Pi`.
    rho : `ndarray`
        Parameters that measure within nesting group correlation, :math:`\rho`.
    xi : `ndarray`
        Unobserved demand-side product characteristics, :math:`\xi`.
    omega : `ndarray`
        Unobserved supply-side product characteristics, :math:`\omega`.
    rc_types : `list of str`
        Random coefficient types.
    epsilon_scale : `float`
        Factor by which the Type I Extreme Value idiosyncratic preference term, :math:`\epsilon_{ijt}`, is scaled.
    costs_type : `str`
        Functional form of the marginal cost function :math:`\tilde{c} = f(c)`.
    T : `int`
        Number of markets, :math:`T`.
    N : `int`
        Number of products across all markets, :math:`N`.
    F : `int`
        Number of firms across all markets, :math:`F`.
    I : `int`
        Number of agents across all markets, :math:`I`.
    K1 : `int`
        Number of demand-side linear product characteristics, :math:`K_1`.
    K2 : `int`
        Number of demand-side nonlinear product characteristics, :math:`K_2`.
    K3 : `int`
        Number of supply-side characteristics, :math:`K_3`.
    D : `int`
        Number of demographic variables, :math:`D`.
    MD : `int`
        Number of demand-side instruments, :math:`M_D`, which is always zero because instruments are added or
        constructed in :meth:`SimulationResults.to_problem`.
    MS : `int`
        Number of supply-side instruments, :math:`M_S`, which is always zero because  instruments are added or
        constructed in :meth:`SimulationResults.to_problem`.
    MC : `int`
        Number of covariance instruments, :math:`M_C`.
    ED : `int`
        Number of absorbed dimensions of demand-side fixed effects, :math:`E_D`, which is always zero because
        simulations do not support fixed effect absorption.
    ES : `int`
        Number of absorbed dimensions of supply-side fixed effects, :math:`E_S`, which is always zero because
        simulations do not support fixed effect absorption.
    H : `int`
        Number of nesting groups, :math:`H`.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    product_data: RecArray
    agent_data: Optional[RecArray]
    integration: Optional[Integration]
    beta: Array
    sigma: Array
    gamma: Array
    pi: Array
    rho: Array
    xi: Array
    omega: Optional[Array]
    _parameters: Parameters

    def __init__(
            self, product_formulations: Union[Formulation, Sequence[Optional[Formulation]]], product_data: Mapping,
            beta: Any, sigma: Optional[Any] = None, pi: Optional[Any] = None, gamma: Optional[Any] = None,
            rho: Optional[Any] = None, agent_formulation: Optional[Formulation] = None,
            agent_data: Optional[Mapping] = None, integration: Optional[Integration] = None, xi: Optional[Any] = None,
            omega: Optional[Any] = None, xi_variance: float = 1, omega_variance: float = 1, correlation: float = 0.9,
            rc_types: Optional[Sequence[str]] = None, epsilon_scale: float = 1.0, costs_type: str = 'linear',
            seed: Optional[int] = None) -> None:
        """Load or simulate all data except for synthetic prices and shares."""

        # keep track of long it takes to initialize the simulation
        output("Initializing the simulation ...")
        start_time = time.time()

        # validate and normalize product formulations
        if isinstance(product_formulations, Formulation):
            product_formulations = [product_formulations]
        elif isinstance(product_formulations, collections.abc.Sequence) and len(product_formulations) <= 3:
            product_formulations = list(product_formulations)
        else:
            raise TypeError("product_formulations must be a Formulation instance or a sequence of up to three of them.")
        if any(f._absorbed_terms for f in product_formulations if f is not None):
            raise ValueError("product_formulations do not support fixed effect absorption in simulations.")
        product_formulations.extend([None] * (3 - len(product_formulations)))

        # validate the agent formulation
        if agent_formulation is not None:
            if not isinstance(agent_formulation, Formulation):
                raise TypeError("agent_formulation must be None or a Formulation instance.")
            if agent_formulation._absorbed_terms:
                raise ValueError("agent_formulation does not support fixed effect absorption.")

        # load IDs and ownership matrices
        market_ids = extract_matrix(product_data, 'market_ids')
        firm_ids = extract_matrix(product_data, 'firm_ids')
        nesting_ids = extract_matrix(product_data, 'nesting_ids')
        product_ids = extract_matrix(product_data, 'product_ids')
        clustering_ids = extract_matrix(product_data, 'clustering_ids')
        ownership = extract_matrix(product_data, 'ownership')
        if market_ids is None:
            raise KeyError("product_data must have a market_ids field.")
        if firm_ids is None:
            raise KeyError("product_data must have a firm_ids field.")
        if market_ids.shape[1] > 1:
            raise ValueError("The market_ids field of product_data must be one-dimensional.")
        if firm_ids.shape[1] > 1:
            raise ValueError("The firm_ids field of product_data must be one-dimensional.")

        # seed the random number generator
        state = np.random.RandomState(seed)

        # load or simulate endogenous variables
        market_groups = Groups(market_ids)
        shares = extract_matrix(product_data, 'shares')
        prices = extract_matrix(product_data, 'prices')
        if shares is None:
            shares = state.uniform(size=market_ids.size) / market_groups.expand(market_groups.counts)
        if prices is None:
            prices = state.uniform(size=market_ids.size)

        # load or simulate product variables in sorted order so that a seed always gives the same draws
        product_mapping = {
            'market_ids': (market_ids, np.object_),
            'firm_ids': (firm_ids, np.object_),
            'nesting_ids': (nesting_ids, np.object_),
            'product_ids': (product_ids, np.object_),
            'clustering_ids': (clustering_ids, np.object_),
            'ownership': (ownership, options.dtype),
            'shares': (shares, options.dtype),
            'prices': (prices, options.dtype),
        }
        for formulation in product_formulations:
            if formulation is None:
                continue
            for name in sorted(formulation._names - set(product_mapping)):
                variable = extract_matrix(product_data, name)
                if variable is None:
                    variable = state.uniform(size=market_ids.size)
                elif variable.shape[1] > 1:
                    raise ValueError(f"The {name} variable has a field in product_data with more than one column.")
                variable_dtype = options.dtype if np.issubdtype(variable.dtype, np.number) else np.object_
                product_mapping[name] = (variable, variable_dtype)

        # structure product data
        self.product_data = structure_matrices(product_mapping)
        products = Products(product_formulations, self.product_data, instruments=False)

        # load or build agent data
        agent_mapping = None
        if products.X2.shape[1] > 0:
            # determine the number of agents by loading market IDs or by building them along with nodes and weights
            if integration is not None:
                if not isinstance(integration, Integration):
                    raise ValueError("integration must be None or an Integration instance.")
                agent_market_ids, nodes, weights = integration._build_many(products.X2.shape[1], np.unique(market_ids))
                agent_ids = availability = None
            elif agent_data is not None:
                agent_market_ids = extract_matrix(agent_data, 'market_ids')
                agent_ids = extract_matrix(agent_data, 'agent_ids')
                nodes = extract_matrix(agent_data, 'nodes')
                weights = extract_matrix(agent_data, 'weights')
                availability = extract_matrix(agent_data, 'availability')
            else:
                raise ValueError("At least one of agent_data or integration must be specified.")

            # load or simulate agent variables in sorted order so that a seed always gives the same draws
            agent_mapping = {
                'market_ids': (agent_market_ids, np.object_),
                'agent_ids': (agent_ids, np.object_),
                'nodes': (nodes, options.dtype),
                'weights': (weights, options.dtype),
                'availability': (availability, options.dtype),
            }
            if agent_formulation is not None:
                for name in sorted(agent_formulation._names - set(agent_mapping)):
                    matrix = extract_matrix(agent_data, name) if agent_data is not None else None
                    if matrix is None:
                        agent_mapping[name] = (state.uniform(size=agent_market_ids.size), options.dtype)
                    else:
                        variable_dtype = options.dtype if np.issubdtype(matrix.dtype, np.number) else np.object_
                        if matrix.shape[1] == 1:
                            agent_mapping[name] = (matrix, variable_dtype)
                        else:
                            for index, variable in enumerate(matrix.T):
                                agent_mapping[f'{name}{index}'] = (variable, variable_dtype)

        # structure agent data
        self.integration = integration
        self.agent_data = structure_matrices(agent_mapping) if agent_mapping is not None else None
        agents = Agents(products, agent_formulation, self.agent_data)

        # initialize the underlying economy
        super().__init__(product_formulations, agent_formulation, products, agents, rc_types, epsilon_scale, costs_type)

        # load or simulate the structural errors
        self.xi = xi
        self.omega = omega
        if self.xi is not None:
            self.xi = np.c_[np.asarray(self.xi, options.dtype)]
            if self.xi.shape != (self.N, 1):
                raise ValueError(f"xi must be a vector with {self.N} elements.")
        if self.omega is not None:
            self.omega = np.c_[np.asarray(self.omega, options.dtype)]
            if self.omega.shape != (self.N, 1):
                raise ValueError(f"omega must be a vector with {self.N} elements.")
        if self.xi is None and self.omega is None and self.K3 > 0:
            covariance = correlation * np.sqrt(xi_variance * omega_variance)
            covariances = np.array([[xi_variance, covariance], [covariance, omega_variance]], options.dtype)
            self._require_psd(covariances, "the covariance matrix from xi_variance, omega_variance, and correlation")
            xi_and_omega = state.multivariate_normal([0, 0], covariances, self.N, check_valid='ignore')
            self.xi = xi_and_omega[:, [0]].astype(options.dtype)
            self.omega = xi_and_omega[:, [1]].astype(options.dtype)
        if self.xi is None:
            raise ValueError("xi must be specified if X3 is not formulated or omega is specified.")
        if self.omega is None and self.K3 > 0:
            raise ValueError("omega must be specified if X3 is formulated and xi is specified.")

        # validate parameters
        self._parameters = Parameters(self, sigma, pi, rho, beta, gamma)
        self.sigma = self._parameters.sigma
        self.pi = self._parameters.pi
        self.rho = self._parameters.rho
        self.beta = self._parameters.beta
        self.gamma = self._parameters.gamma

        # output information about the initialized simulation
        output(f"Initialized the simulation after {format_seconds(time.time() - start_time)}.")
        output("")
        output(self)

    def __str__(self) -> str:
        """Supplement general formatted information with other information about parameters."""
        return "\n\n".join([super().__str__(), self._parameters.format("True Values")])

    def replace_endogenous(
            self, costs: Optional[Any] = None, prices: Optional[Any] = None, iteration: Optional[Iteration] = None,
            constant_costs: bool = True, compute_gradients: bool = True, compute_hessians: bool = True,
            error_behavior: str = 'raise') -> SimulationResults:
        r"""Replace simulated prices and market shares with equilibrium values that are consistent with true parameters.

        This method is the standard way of solving the simulation. Prices and market shares are computed in each market
        by iterating over the :math:`\zeta`-markup contraction in :eq:`zeta_contraction`:

        .. math:: p \leftarrow c + \zeta(p).

        .. note::

           To not replace prices, pass the desired prices to ``prices`` and use an :class:`Iteration` configuration with
           ``method='return'``. This just uses the iteration "routine" that simply returns the the starting values,
           which are ``prices``.

           Using this same fake iteration routine and not setting prices will result in a simulation under perfect
           (instead of Bertrand) competition because the default starting values for the iteration routine are marginal
           costs.

        .. note::

           This method supports :func:`parallel` processing. If multiprocessing is used, market-by-market computation of
           prices and shares will be distributed among the processes.

        Parameters
        ----------
        costs : `array-like, optional`
            Marginal costs, :math:`c`. By default, :math:`c = X_3\gamma + \omega` if ``costs_type`` was ``'linear'`` in
            :class:`Simulation` (the default), and the exponential of this if it was ``'log'``. Marginal costs must be
            specified if :math:`X_3` was not formulated in :class:`Simulation`. If marginal costs depend on prices
            through market shares, they will be updated to reflect different prices during each iteration of the
            routine.
        prices : `array-like, optional`
            Prices at which the fixed point iteration routine will start. By default, ``costs``, are used as starting
            values.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem. By default,
            ``Iteration('simple', {'atol': 1e-12})`` is used.
        constant_costs : `bool, optional`
            Whether to assume that marginal costs, :math:`c`, remain constant as equilibrium prices and shares change.
            By default this is ``True``, which means that firms treat marginal costs as constant (equal to ``costs``)
            when setting prices. If set to ``False``, marginal costs will be allowed to adjust if ``shares`` was
            included in the formulation for :math:`X_3`. When simulating fake data, it likely makes more sense to set
            this to ``False`` since otherwise arbitrary ``shares`` simulated by :class:`Simulation` will be used in
            marginal costs.
        compute_gradients : `bool, optional`
            Whether to compute profit gradients to verify first order conditions. This is by default ``True``. Setting
            it to ``False`` will slightly speed up computation, but first order conditions will not be reported.
        compute_hessians : `bool, optional`
            Whether to compute profit Hessians to verify second order conditions. This is by default ``True``. Setting
            it to ``False`` will slightly speed up computation, but second order conditions will not be reported.
        error_behavior : `str, optional`
            How to handle errors when computing prices and shares. For example, the fixed point routine may not converge
            if the effects of nonlinear parameters on price overwhelm the linear parameter on price, which should be
            sufficiently negative. The following behaviors are supported:

                - ``'raise'`` (default) - Raise an exception.

                - ``'warn'`` - Use the last computed prices and shares. If the fixed point routine fails to converge,
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

        # keep track of long it takes to replace endogenous variables
        output("Replacing prices and shares ...")
        start_time = time.time()

        # load or compute marginal costs, which may be updated if shares enter into X3
        if costs is not None:
            costs = np.c_[np.asarray(costs, options.dtype)]
            if costs.shape != (self.N, 1):
                raise ValueError(f"costs must be {self.N}-vector.")
        elif self.K3 == 0:
            raise ValueError("costs must be specified if X3 was not formulated.")
        else:
            costs = self.products.X3 @ self.gamma + self.omega
            if self.costs_type == 'log':
                costs = np.exp(costs)

        # choose or validate initial prices
        if prices is None:
            prices = costs
        else:
            prices = np.c_[np.asarray(prices, options.dtype)]
            if prices.shape != (self.N, 1):
                raise ValueError(f"prices must None or a {self.N}-vector.")

        # validate other settings
        iteration = self._coerce_optional_prices_iteration(iteration)
        self._validate_error_behavior(error_behavior)

        # compute a baseline delta that will be updated when shares and prices are replaced
        delta = self.products.X1 @ self.beta + self.xi

        def market_factory(s: Hashable) -> Tuple[SimulationMarket, Array, Array, Iteration, bool, bool, bool]:
            """Build a market along with arguments used to compute prices and shares."""
            assert costs is not None and prices is not None and iteration is not None
            market_s = SimulationMarket(
                self, s, self._parameters, self.sigma, self.pi, self.rho, self.beta, self.gamma, delta
            )
            costs_s = costs[self._product_market_indices[s]]
            prices_s = prices[self._product_market_indices[s]]
            return market_s, costs_s, prices_s, iteration, constant_costs, compute_gradients, compute_hessians

        # compute prices and market shares market-by-market, also collecting potentially updated delta and costs
        data_override = {
            'prices': np.zeros_like(self.products.prices),
            'shares': np.zeros_like(self.products.shares)
        }
        true_delta = np.zeros_like(delta)
        true_costs = np.zeros_like(costs)
        iteration_stats: Dict[Hashable, SolverStats] = {}
        profit_gradients: Optional[Dict[Hashable, Dict[Hashable, Array]]] = {} if compute_gradients else None
        profit_gradient_norms: Optional[Dict[Hashable, Dict[Hashable, Array]]] = {} if compute_gradients else None
        profit_hessians: Optional[Dict[Hashable, Dict[Hashable, Array]]] = {} if compute_hessians else None
        profit_hessian_eigenvalues: Optional[Dict[Hashable, Dict[Hashable, Array]]] = {} if compute_hessians else None
        generator = generate_items(self.unique_market_ids, market_factory, SimulationMarket.compute_endogenous)
        generator = output_progress(generator, self.T, start_time)
        for t, (prices_t, shares_t, delta_t, costs_t, stats_t, gradients_t, hessians_t, errors_t) in generator:
            data_override['prices'][self._product_market_indices[t]] = prices_t
            data_override['shares'][self._product_market_indices[t]] = shares_t
            true_delta[self._product_market_indices[t]] = delta_t
            true_costs[self._product_market_indices[t]] = costs_t
            iteration_stats[t] = stats_t
            errors.extend(errors_t)

            # compute profit gradient norms to check first order conditions
            if compute_gradients:
                assert gradients_t is not None
                assert profit_gradients is not None and profit_gradient_norms is not None
                profit_gradients[t] = gradients_t
                profit_gradient_norms[t] = {}
                for f, profit_gradient in gradients_t.items():
                    profit_gradient_norms[t][f] = np.nan
                    if profit_gradient.size > 0:
                        with np.errstate(invalid='ignore'):
                            profit_gradient_norms[t][f] = np.abs(profit_gradient).max()

            # compute profit Hessian eigenvalues to check second order conditions
            if compute_hessians:
                assert hessians_t is not None
                assert profit_hessians is not None and profit_hessian_eigenvalues is not None
                profit_hessians[t] = hessians_t
                profit_hessian_eigenvalues[t] = {}
                for f, profit_hessian in hessians_t.items():
                    profit_hessian_eigenvalues[t][f], successful = precisely_compute_eigenvalues(profit_hessian)
                    if not successful:
                        errors.append(exceptions.ProfitHessianEigenvaluesError(profit_hessian))

        # structure the results
        self._handle_errors(errors, error_behavior)
        results = SimulationResults(
            self, data_override, true_delta, true_costs, start_time, time.time(), iteration_stats, profit_gradients,
            profit_gradient_norms, profit_hessians, profit_hessian_eigenvalues
        )
        output(f"Replaced prices and shares after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    def replace_exogenous(
            self, X1_name: str, X3_name: Optional[str] = None, delta: Optional[Any] = None,
            iteration: Optional[Iteration] = None, fp_type: str = 'safe_linear',
            shares_bounds: Optional[Tuple[Any, Any]] = (1e-300, None), error_behavior: str = 'raise') -> (
            SimulationResults):
        r"""Replace exogenous product characteristics with values that are consistent with true parameters.

        This method implements a less common way of solving the simulation. It may be preferable to
        :meth:`Simulation.replace_endogenous` when for some reason it is desirable to retain the prices and market
        shares from :class:`Simulation`, which are assumed to be in equilibrium. For example, it can be helpful when
        approximating the pure characteristics model of :ref:`references:Berry and Pakes (2007)` by setting a small
        ``epsilon_scale`` value in :class:`Simulation`.

        For this method of solving the simulation to be used, there must be an exogenous product characteristic
        :math:`v` that shows up only in :math:`X_1^\text{ex}`, and if there is a supply side, another product
        characteristic :math:`w` that shows up only in :math:`X_3^\text{ex}`. These characteristics will be replaced
        with values that are consistent with true parameters.

        First, the mean utility :math:`\delta` is computed in each market by iterating over the contraction in
        :eq:`contraction` and :math:`(\delta - \xi - X_1 \beta)\beta_v^{-1}` is added to the :math:`v` from
        :class:`Simulation`. Here, :math:`\beta_v` is the linear parameter in :math:`\beta` on :math:`v`.

        With a supply side, the marginal cost function :math:`\tilde{c}` is computed according to :eq:`eta` and
        :eq:`costs` and :math:`(\tilde{c} - \omega - X_3 \gamma)\gamma_w^{-1}` is added to the :math:`w` from
        :class:`Simulation`. Here, :math:`\gamma_w` is the linear parameter in :math:`\gamma` on :math:`w`.

        .. note::

           This method supports :func:`parallel` processing. If multiprocessing is used, market-by-market computation of
           prices and shares will be distributed among the processes.

        Parameters
        ----------
        X1_name : `str`
            The name of the variable :math:`v` in :math:`X_1^\text{ex}` that will be replaced. It should show up only
            once in the formulation for :math:`X_1` from :class:`Simulation` and it should not be transformed in any
            way.
        X3_name : `str, optional`
            The name of the variable :math:`w` in :math:`X_3^\text{ex}` that will be replaced. It should show up only
            once in the formulation for :math:`X_3` from :class:`Simulation` and it should not be transformed in any
            way. This will only be used if there is a supply side.
        delta : `array-like, optional`
            Initial values for the mean utility, :math:`\delta`, which the fixed point iteration routine will start
            at. By default, the solution to the logit model in :eq:`logit_delta` is used. If there is a nesting
            structure, solution to the nested logit model in :eq:`nested_logit_delta` under the initial ``rho`` is used
            instead.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration for how to solve the fixed point problem used to compute
            :math:`\delta` in each market. This configuration is only relevant if there are nonlinear parameters, since
            :math:`\delta` can be estimated analytically in the logit model. By default,
            ``Iteration('squarem', {'atol': 1e-14})`` is used. For more information, refer to the same argument in
            :meth:`Problem.solve`.
        fp_type : `str, optional`
            Configuration for the type of contraction mapping used to compute :math:`\delta`. For information about
            the different types, refer to the same argument in :meth:`Problem.solve`.
        shares_bounds : `tuple, optional`
            Configuration for :math:`s_{jt}(\delta, \theta)` bounds of the form ``(lb, ub)``, in which both ``lb`` and
            ``ub`` are floats or ``None``. By default, simulated shares are bounded from below by ``1e-300``. This is
            only relevant if ``fp_type`` is ``'safe_linear'`` or ``'linear'``. Bounding shares in the contraction does
            nothing with a nonlinear fixed point. For more information, refer to :meth:`Problem.solve`.
        error_behavior : `str, optional`
            How to handle errors when computing :math:`\delta` and :math:`\tilde{c}`. The following behaviors are
            supported:

                - ``'raise'`` (default) - Raise an exception.

                - ``'warn'`` - Use the last computed :math:`\delta` and :math:`\tilde{c}`. If the fixed point routine
                  fails to converge, these are the last :math:`\delta` and the associated :math:`\tilde{c}` by the
                  routine. If there are other issues, these are the starting :math:`\delta` values and their associated
                  :math:`\tilde{c}`.

        Returns
        -------
        `SimulationResults`
            :class:`SimulationResults` of the solved simulation.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        errors: List[Error] = []

        # keep track of long it takes to solve for exogenous product characteristics
        output("Replacing exogenous product characteristics ...")
        start_time = time.time()

        # validate that it's possible to solve for the exogenous variables
        try:
            X1_index = next(i for i, f in enumerate(self._X1_formulations) if X1_name == f)
        except StopIteration:
            raise ValueError("X1_name must be an untransformed variable in the formulation for X1.")
        if sum(X1_name in f.names for f in self._X1_formulations) > 1:
            raise ValueError("X1_name must appear only once in the formulation for X1.")
        if any(X1_name in f.names for f in self._X2_formulations + self._X3_formulations):
            raise ValueError("X1_name must appear only in the formulation for X1.")
        if X1_name == 'prices':
            raise ValueError("X1_name must be an exogenous variable.")
        if self.beta[X1_index] == 0:
            raise ValueError("The parameter in beta on X1_name cannot be zero.")
        X3_index = None
        if self.K3 > 0:
            try:
                X3_index = next(i for i, f in enumerate(self._X3_formulations) if X3_name == f)
            except StopIteration:
                raise ValueError("X3_name must be an untransformed variable in the formulation for X3.")
            if all(X3_name != f for f in self._X3_formulations):
                raise ValueError("X3_name must be an untransformed variable in the formulation for X3.")
            if sum(X3_name in f.names for f in self._X3_formulations) > 1:
                raise ValueError("X3_name must appear only once in the formulation for X3.")
            if any(X3_name in f.names for f in self._X1_formulations + self._X2_formulations):
                raise ValueError("X3_name must appear only in the formulation for X3.")
            if X3_name == 'shares':
                raise ValueError("X3_name must be an exogenous variable.")
            if self.gamma[X3_index] == 0:
                raise ValueError("The parameter in gamma on X3_name cannot be zero.")

        # choose or validate the initial delta
        if delta is None:
            delta = self._compute_logit_delta(self.rho)
        else:
            delta = np.c_[np.asarray(delta, options.dtype)]
            if delta.shape != (self.N, 1):
                raise ValueError(f"delta must None or a {self.N}-vector.")

        # validate other settings
        iteration = self._coerce_optional_delta_iteration(iteration)
        shares_bounds = self._coerce_optional_bounds(shares_bounds, 'shares_bounds')
        self._validate_fp_type(fp_type)
        self._validate_error_behavior(error_behavior)

        def market_factory(s: Hashable) -> Tuple[SimulationMarket, Array, Iteration, str, Bounds]:
            """Build a market along with arguments used to compute delta and marginal costs."""
            assert delta is not None and iteration is not None and shares_bounds is not None
            market_s = SimulationMarket(self, s, self._parameters, self.sigma, self.pi, self.rho, self.beta)
            delta_s = delta[self._product_market_indices[s]]
            return market_s, delta_s, iteration, fp_type, shares_bounds

        # compute delta and marginal costs market-by-market
        true_delta = np.zeros_like(self.xi)
        true_tilde_costs = None if self.omega is None else np.zeros_like(self.omega)
        iteration_stats: Dict[Hashable, SolverStats] = {}
        generator = generate_items(self.unique_market_ids, market_factory, SimulationMarket.compute_exogenous)
        for t, (delta_t, tilde_costs_t, iteration_stats_t, errors_t) in output_progress(generator, self.T, start_time):
            true_delta[self._product_market_indices[t]] = delta_t
            if true_tilde_costs is not None:
                true_tilde_costs[self._product_market_indices[t]] = tilde_costs_t
            iteration_stats[t] = iteration_stats_t
            errors.extend(errors_t)

        # compute the exogenous variables, ignoring any numerical errors here that carry over from market computation
        data_override: Dict[str, Array] = {}
        with np.errstate(all='ignore'):
            data_override[X1_name] = (
                self.products.X1[:, [X1_index]] +
                (true_delta - self.xi - self.products.X1 @ self.beta) / self.beta[X1_index]
            )
            if X3_name is not None:
                assert true_tilde_costs is not None
                data_override[X3_name] = (
                    self.products.X3[:, [X3_index]] +
                    (true_tilde_costs - self.omega - self.products.X3 @ self.gamma) / self.gamma[X3_index]
                )

        # compute non-transformed marginal costs
        true_costs = true_tilde_costs
        if self.costs_type == 'log' and true_costs is not None:
            true_costs = np.exp(true_costs)

        # structure the results
        self._handle_errors(errors, error_behavior)
        results = SimulationResults(
            self, data_override, true_delta, true_costs, start_time, time.time(), iteration_stats
        )
        output(f"Replaced exogenous product characteristics after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    @staticmethod
    def _validate_error_behavior(error_behavior: str) -> None:
        """Validate that a specified error behavior is supported."""
        if error_behavior not in {'raise', 'warn'}:
            raise ValueError("error_behavior must be 'raise' or 'warn'.")
