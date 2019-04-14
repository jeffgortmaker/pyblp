"""Economy underlying the BLP model."""

import abc
import collections
import functools
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence

import numpy as np

from .. import options
from ..configurations.formulation import Formulation
from ..primitives import Container
from ..utilities.algebra import precisely_identify_collinearity
from ..utilities.basics import Array, RecArray, StringRepresentation, TableFormatter


class Economy(Container, StringRepresentation):
    """An abstract economy underlying the BLP model."""

    product_formulations: Sequence[Optional[Formulation]]
    agent_formulation: Optional[Formulation]
    unique_market_ids: Array
    unique_firm_ids: Array
    unique_nesting_ids: Array
    N: int
    T: int
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
        self._product_market_indices = {t: np.where(self.products.market_ids.flat == t) for t in self.unique_market_ids}
        self._agent_market_indices = {t: np.where(self.agents.market_ids.flat == t) for t in self.unique_market_ids}

        # identify the largest number of products and agents in a market
        self._max_J = max(v[0].size for v in self._product_market_indices.values())
        self._max_I = max(v[0].size for v in self._agent_market_indices.values())

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

        # associate dimensions and formulations with names
        dimension_mapping = collections.OrderedDict([
            ("N", self.N),
            ("T", self.T),
            ("F", self.F),
            ("I", self.I),
            ("K1", self.K1),
            ("K2", self.K2),
            ("K3", self.K3),
            ("D", self.D),
            ("MD", self.MD),
            ("MS", self.MS),
            ("ED", self.ED),
            ("ES", self.ES),
            ("H", self.H)
        ])
        formulation_mapping = collections.OrderedDict([
            ("X1: Linear Characteristics", self._X1_formulations),
            ("X2: Nonlinear Characteristics", self._X2_formulations),
            ("X3: Cost Characteristics", self._X3_formulations),
            ("d: Demographics", self._demographics_formulations)
        ])

        # build a dimensions section
        dimension_widths = [max(len(k) + 2, len(str(d))) for k, d in dimension_mapping.items() if d > 0]
        dimension_formatter = TableFormatter(dimension_widths)
        dimension_section = [
            "Dimensions:",
            dimension_formatter.line(),
            dimension_formatter([k for k, d in dimension_mapping.items() if d > 0], underline=True),
            dimension_formatter([d for k, d in dimension_mapping.items() if d > 0]),
            dimension_formatter.line()
        ]

        # build a formulations section
        formulation_header = ["Column Indices:"]
        formulation_widths = [max(len(formulation_header[0]), max(map(len, formulation_mapping.keys())))]
        for index in range(max(map(len, formulation_mapping.values()))):
            formulation_header.append(str(index))
            column_width = 5
            for formulation in formulation_mapping.values():
                if len(formulation) > index:
                    column_width = max(column_width, len(str(formulation[index])))
            formulation_widths.append(column_width)
        formulation_formatter = TableFormatter(formulation_widths)
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

    def _validate_shares(self) -> None:
        """Validate the integrity of product shares."""
        shares = self.products.shares
        if (shares <= 0).any() or (shares >= 1).any():
            raise ValueError("Product shares must be between zero and one, exclusive.")
        bad_markets = [t for t, indices in self._product_market_indices.items() if shares[indices].sum() >= 1]
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
            collinear, successful = precisely_identify_collinearity(
                self.products[name], options.collinear_atol, options.collinear_rtol
            )
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

    def _validate_name(self, name: str) -> None:
        """Validate that a name corresponds to a variable in X1, X2, or X3."""
        formulations = self._X1_formulations + self._X2_formulations + self._X3_formulations
        names = {n for f in formulations for n in f.names}
        if name not in names:
            raise NameError(f"The name '{name}' is not one of the underlying variables, {list(sorted(names))}.")

    def _coerce_optional_firm_ids(self, firm_ids: Optional[Any]) -> Array:
        """Coerce optional array-like firm IDs into a column vector and validate it."""
        if firm_ids is not None:
            firm_ids = np.c_[np.asarray(firm_ids, options.dtype)]
            if firm_ids.shape != (self.N, 1):
                raise ValueError(f"firm_ids must be None or a {self.N}-vector.")
        return firm_ids

    def _coerce_optional_ownership(self, ownership: Optional[Any]) -> Array:
        """Coerce optional array-like firm IDs into a column vector and validate it."""
        if ownership is not None:
            ownership = np.c_[np.asarray(ownership, options.dtype)]
            if ownership.shape != (self.N, self._max_J):
                raise ValueError(f"ownership must be None or a {self.N} by {self._max_J} matrix.")
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
