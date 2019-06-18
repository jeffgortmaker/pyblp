"""Economy underlying the BLP model."""

import abc
import functools
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence

import pandas as pd
import numpy as np

from .. import options
from ..configurations.formulation import Formulation
from ..primitives import Container
from ..utilities.algebra import precisely_identify_collinearity, precisely_identify_psd
from ..utilities.basics import Array, RecArray, StringRepresentation, format_table


class Economy(Container, StringRepresentation):
    """An abstract economy underlying the BLP model."""

    product_formulations: Sequence[Optional[Formulation]]
    agent_formulation: Optional[Formulation]
    unique_market_ids: Array
    unique_firm_ids: Array
    unique_nesting_ids: Array
    T: int
    N: int
    F: int
    I: int
    K1: int
    K2: int
    K3: int
    D: int
    MD: int
    MS: int
    ED: int
    ES: int
    H: int
    _product_market_indices: Dict[Hashable, Array]
    _agent_market_indices: Dict[Hashable, Array]
    _max_J: int
    _max_I: int
    _absorb_demand_ids: Optional[functools.partial]
    _absorb_supply_ids: Optional[functools.partial]

    @abc.abstractmethod
    def __init__(
            self, product_formulations: Sequence[Optional[Formulation]], agent_formulation: Optional[Formulation],
            products: RecArray, agents: RecArray) -> None:
        """Store information about formulations and data. Any fixed effects should be absorbed after initialization."""

        # store data and formulations
        super().__init__(products, agents)
        self.product_formulations = product_formulations
        self.agent_formulation = agent_formulation

        # identify unique markets and nests
        self.unique_market_ids = np.unique(self.products.market_ids).flatten()
        self.unique_firm_ids = np.unique(self.products.firm_ids).flatten()
        self.unique_nesting_ids = np.unique(self.products.nesting_ids).flatten()

        # count dimensions
        self.N = self.products.shape[0]
        self.T = self.unique_market_ids.size
        self.F = self.unique_firm_ids.size
        self.I = self.agents.shape[0] if self.products.X2.shape[1] > 0 else 0
        self.K1 = self.products.X1.shape[1]
        self.K2 = self.products.X2.shape[1]
        self.K3 = self.products.X3.shape[1]
        self.D = self.agents.demographics.shape[1]
        self.MD = self.products.ZD.shape[1]
        self.MS = self.products.ZS.shape[1]
        self.ED = self.products.demand_ids.shape[1]
        self.ES = self.products.supply_ids.shape[1]
        self.H = self.unique_nesting_ids.size

        # identify market indices
        s = pd.Series(self.products.market_ids.flatten())
        self._product_market_indices: Dict[Hashable, Array] = {k: v for k, v in s.groupby(s).indices.items()}
        s = pd.Series(self.agents.market_ids.flatten())
        self._agent_market_indices: Dict[Hashable, Array] = {k: v for k, v in s.groupby(s).indices.items()}

        # identify the largest number of products and agents in a market
        self._max_J = max(i.size for i in self._product_market_indices.values())
        self._max_I = max(i.size for i in self._agent_market_indices.values())

        # construct fixed effect absorption functions
        self._absorb_demand_ids = self._absorb_supply_ids = None
        if self.ED > 0:
            assert product_formulations[0] is not None
            self._absorb_demand_ids = functools.partial(product_formulations[0]._build_absorb(self.products.demand_ids))
        if self.ES > 0:
            assert product_formulations[2] is not None
            self._absorb_supply_ids = functools.partial(product_formulations[2]._build_absorb(self.products.supply_ids))

    def __str__(self) -> str:
        """Format economy information as a string."""
        return "\n\n".join([self._format_dimensions(), self._format_formulations()])

    def _format_dimensions(self) -> str:
        """Format information about the nonzero dimensions of the economy as a string."""
        header: List[str] = []
        values: List[str] = []
        for key in ['T', 'N', 'F', 'I', 'K1', 'K2', 'K3', 'D', 'MD', 'MS', 'ED', 'ES', 'H']:
            value = getattr(self, key)
            if value > 0:
                header.append(f" {key} ")
                values.append(str(value))
        return format_table(header, values, title="Dimensions")

    def _format_formulations(self) -> str:
        """Formation information about the formulations of the economy as a string."""

        # construct the data
        named_formulations = [
            (self._X1_formulations, "X1: Linear Characteristics"),
            (self._X2_formulations, "X2: Nonlinear Characteristics"),
            (self._X3_formulations, "X3: Cost Characteristics"),
            (self._demographics_formulations, "d: Demographics")
        ]
        data: List[List[str]] = []
        for formulations, name in named_formulations:
            if any(formulations):
                data.append([name] + [str(f) for f in formulations])

        # construct the header
        max_formulations = max(len(r[1:]) for r in data)
        header = ["Column Indices:"] + [f" {i} " for i in range(max_formulations)]

        # format the table
        return format_table(header, *data, title="Formulations")

    def _validate_shares(self) -> None:
        """Validate the integrity of product shares."""
        shares = self.products.shares
        if (shares <= 0).any() or (shares >= 1).any():
            raise ValueError("Product shares must be between zero and one, exclusive.")
        bad_markets = [t for t, i in self._product_market_indices.items() if shares[i].sum() >= 1]
        if bad_markets:
            raise ValueError(f"Shares in the following markets do not sum to less than one: {bad_markets}.")

    def _detect_collinearity(self) -> None:
        """Detect any collinearity issues in product data matrices."""

        # skip collinearity checking when it is disabled via zero tolerances
        if max(options.collinear_atol, options.collinear_rtol) <= 0:
            return

        # collect labels for columns of matrices that will be checked for collinearity issues
        matrix_labels = {
            'X1': [str(f) for f in self._X1_formulations],
            'X2': [str(f) for f in self._X2_formulations],
            'X3': [str(f) for f in self._X3_formulations],
            'ZD': [str(f) for f in self._X1_formulations if 'prices' not in f.names],
            'ZS': [str(f) for f in self._X3_formulations]
        }
        matrix_labels.update({
            'ZD': [f'demand_instruments{i}' for i in range(self.MD - len(matrix_labels['ZD']))] + matrix_labels['ZD'],
            'ZS': [f'demand_instruments{i}' for i in range(self.MD - len(matrix_labels['ZS']))] + matrix_labels['ZS']
        })

        # check each matrix for collinearity
        for name, labels in matrix_labels.items():
            collinear, successful = precisely_identify_collinearity(self.products[name])
            common_message = "To disable collinearity checks, set options.collinear_atol = options.collinear_rtol = 0."
            if (self.ED > 0 and name in {'X1', 'ZD'}) or (self.ES > 0 and name in {'X3', 'ZS'}):
                common_message = f"Absorbed fixed effects may be creating collinearity problems. {common_message}"
            if not successful:
                raise ValueError(
                    f"Failed to compute the QR decomposition of {name} while checking for collinearity issues. "
                    f"{common_message}"
                )
            if collinear.any():
                collinear_labels = ", ".join(l for l, c in zip(labels, collinear) if c)
                raise ValueError(
                    f"Detected collinearity issues with [{collinear_labels}] and at least one other column in {name}. "
                    f"{common_message}"
                )

    @staticmethod
    def _detect_psd(matrix: Array, name: str) -> None:
        """Detect whether a matrix is PSD."""
        psd, successful = precisely_identify_psd(matrix)
        common_message = "To disable PSD checks, set options.psd_atol = options.psd_rtol = numpy.inf."
        if not successful:
            raise ValueError(f"Failed to compute the SVD of {name} while checking that it is PSD. {common_message}")
        if not psd:
            raise ValueError(f"{name} must be a PSD matrix. {common_message}")

    def _validate_name(self, name: str) -> None:
        """Validate that a name corresponds to a variable in X1, X2, or X3."""
        formulations = self._X1_formulations + self._X2_formulations + self._X3_formulations
        names = {n for f in formulations for n in f.names}
        if name not in names:
            raise NameError(f"The name '{name}' is not one of the underlying variables, {list(sorted(names))}.")

    def _coerce_optional_firm_ids(self, firm_ids: Optional[Any], market_ids: Optional[Array] = None) -> Array:
        """Coerce optional array-like firm IDs into a column vector and validate it. By default, assume that firm IDs
        are for all markets.
        """
        if firm_ids is not None:
            firm_ids = np.c_[np.asarray(firm_ids, options.dtype)]
            rows = self.N
            if market_ids is not None:
                rows = sum(i.size for t, i in self._product_market_indices.items() if t in market_ids)
            if firm_ids.shape != (rows, 1):
                raise ValueError(f"firm_ids must be None or a {rows}-vector.")
        return firm_ids

    def _coerce_optional_ownership(self, ownership: Optional[Any], market_ids: Optional[Array] = None) -> Array:
        """Coerce optional array-like ownership matrices into a stacked matrix and validate it. By default, assume that
        ownership matrices are for all markets.
        """
        if ownership is not None:
            ownership = np.c_[np.asarray(ownership, options.dtype)]
            rows = self.N
            columns = self._max_J
            if market_ids is not None:
                rows = sum(i.size for t, i in self._product_market_indices.items() if t in market_ids)
                columns = max(i.size for t, i in self._product_market_indices.items() if t in market_ids)
            if ownership.shape != (rows, columns):
                raise ValueError(f"ownership must be None or a {rows} by {columns} matrix.")
        return ownership

    def _compute_true_X1(self, data_override: Optional[Mapping] = None, index: Optional[Array] = None) -> Array:
        """Compute X1 or columns of X1 without any absorbed demand-side fixed effects."""
        if index is None:
            index = np.ones(self.K1, np.bool)
        if self.ED == 0 and not data_override:
            return self.products.X1[:, index]
        columns = []
        for include, formulation in zip(index, self._X1_formulations):
            if include:
                column = formulation.evaluate(self.products, data_override)
                columns.append(np.broadcast_to(column, (self.N, 1)).astype(options.dtype))
        return np.column_stack(columns)

    def _compute_true_X3(
            self, data_override: Optional[Mapping] = None, index: Optional[Array] = None) -> Array:
        """Compute X3 or columns of X3 without any absorbed supply-side fixed effects."""
        if index is None:
            index = np.ones(self.K3, np.bool)
        if self.ES == 0 and not data_override:
            return self.products.X3[:, index]
        columns = []
        for include, formulation in zip(index, self._X3_formulations):
            if include:
                columns.append(formulation.evaluate(self.products, data_override) * np.ones((self.N, 1)))
        return np.column_stack(columns)
