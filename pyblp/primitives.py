"""Primitive data structures that constitute the foundation of the BLP model."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import options
from .configurations.formulation import ColumnFormulation, Formulation
from .configurations.integration import Integration
from .utilities.basics import Array, Data, RecArray, extract_matrix, structure_matrices


class Products(object):
    r"""Product data structured as a record array.

    Attributes in addition to the ones below are the variables underlying :math:`X_1`, :math:`X_2`, and :math:`X_3`.

    Attributes
    ----------
    market_ids : `ndarray`
        IDs that associate products with markets.
    firm_ids : `ndarray`
        IDs that associate products with firms. Any columns after the first represent changes such as mergers.
    demand_ids : `ndarray`
        IDs used to create demand-side fixed effects.
    supply_ids : `ndarray`
        IDs used to create supply-side fixed effects.
    nesting_ids : `ndarray`
        IDs that associate products with nesting groups.
    clustering_ids : `ndarray`
        IDs used to compute clustered standard errors.
    ownership : `ndarray`
        Stacked :math:`J_t \times J_t` ownership matrices, :math:`O`, for each market :math:`t`. Each stack is
        associated with a `firm_ids` column.
    shares : `ndarray`
        Market shares, :math:`s`.
    ZD : `ndarray`
        Demand-side instruments, :math:`Z_D`.
    ZS : `ndarray`
        Supply-side instruments, :math:`Z_S`.
    X1 : `ndarray`
        Linear product characteristics, :math:`X_1`.
    X2 : `ndarray`
        Nonlinear product characteristics, :math:`X_2`.
    X3 : `ndarray`
        Cost product characteristics, :math:`X_3`.
    prices : `ndarray`
        Product prices, :math:`p`.

    """

    market_ids: Array
    firm_ids: Array
    demand_ids: Array
    supply_ids: Array
    nesting_ids: Array
    ownership: Array
    shares: Array
    ZD: Array
    ZS: Array
    X1: Array
    X2: Array
    X3: Array
    prices: Array

    def __new__(cls, product_formulations: Sequence[Optional[Formulation]], product_data: Mapping) -> RecArray:
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
        X2_formulations: List[ColumnFormulation] = []
        X2_data: Data = {}
        if product_formulations[1] is not None:
            X2, X2_formulations, X2_data = product_formulations[1]._build_matrix(product_data)
            if 'shares' in X2_data:
                raise NameError("shares cannot be included in the formulation for X2.")

        # check that prices are in X1 or X2
        if 'prices' not in X1_data and 'prices' not in X2_data:
            raise NameError("prices must be included in at least one of formulations for X1 or X2.")

        # build X3
        X3 = None
        X3_formulations: List[ColumnFormulation] = []
        X3_data: Data = {}
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
        demand_ids = None
        supply_ids = None
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
        product_mapping: Dict[Union[str, tuple], Tuple[Optional[Array], Any]] = {}
        product_mapping.update({
            'market_ids': (market_ids, np.object),
            'firm_ids': (firm_ids, np.object),
            'demand_ids': (demand_ids, np.object),
            'supply_ids': (supply_ids, np.object),
            'nesting_ids': (nesting_ids, np.object),
            'clustering_ids': (clustering_ids, np.object),
            'ownership': (ownership, options.dtype),
            'shares': (shares, options.dtype),
            'ZD': (ZD, options.dtype),
            'ZS': (ZS, options.dtype)
        })
        product_mapping.update({
            (tuple(X1_formulations), 'X1'): (X1, options.dtype),
            (tuple(X2_formulations), 'X2'): (X2, options.dtype),
            (tuple(X3_formulations), 'X3'): (X3, options.dtype)
        })

        # structure and validate variables underlying X1, X2, and X3
        underlying_data = {k: (v, options.dtype) for k, v in {**X1_data, **X2_data, **X3_data}.items()}
        invalid_names = set(underlying_data) & {k if isinstance(k, str) else k[1] for k in product_mapping}
        if invalid_names:
            raise NameError(f"These reserved names in product_formulations are invalid: {list(invalid_names)}.")

        # structure products
        return structure_matrices({**product_mapping, **underlying_data})


class Agents(object):
    r"""Agent data structured as a record array.

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

    market_ids: Array
    weights: Array
    nodes: Array
    demographics: Array

    def __new__(
            cls, products: RecArray, agent_formulation: Optional[Formulation] = None,
            agent_data: Optional[Mapping] = None, integration: Optional[Integration] = None) -> RecArray:
        """Structure agent data."""

        # data structures may be empty
        market_ids = None
        weights = None
        nodes = None
        demographics = None
        demographics_formulations: List[ColumnFormulation] = []

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
        else:
            # validate the formulation and build demographics
            if agent_formulation is not None:
                if not isinstance(agent_formulation, Formulation):
                    raise TypeError("agent_formulation must be None or a Formulation instance.")
                if agent_data is None:
                    raise ValueError("Since agent_formulation is specified, agent_data must be specified as well.")
                if agent_formulation._absorbed_terms:
                    raise ValueError("agent_formulation does not support fixed effect absorption.")
                demographics, demographics_formulations = agent_formulation._build_matrix(agent_data)[:2]

            # load market IDs
            if agent_data is not None:
                market_ids = extract_matrix(agent_data, 'market_ids')
                if market_ids is None:
                    raise KeyError("agent_data must have a market_ids field.")
                if market_ids.shape[1] > 1:
                    raise ValueError("The market_ids field of agent_data must be one-dimensional.")
                if set(np.unique(products.market_ids)) != set(np.unique(market_ids)):
                    raise ValueError("The market_ids field of agent_data must have the same IDs as product data.")

            # build nodes and weights
            if integration is not None:
                if not isinstance(integration, Integration):
                    raise ValueError("integration must be None or an Integration instance.")
                loaded_market_ids = market_ids
                market_ids, nodes, weights = integration._build_many(K2, np.unique(products.market_ids))

                # delete rows of demographics if there are too many
                if demographics is not None:
                    assert loaded_market_ids is not None
                    demographics_list: List[Array] = []
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
                if agent_data is None:
                    raise ValueError("Since integration is None, agent_data must be specified.")
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
        return structure_matrices({
            'market_ids': (market_ids, np.object),
            'weights': (weights, options.dtype),
            'nodes': (nodes, options.dtype),
            (tuple(demographics_formulations), 'demographics'): (demographics, options.dtype)
        })
