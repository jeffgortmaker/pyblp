"""Economy underlying the BLP model."""

import abc
import collections.abc
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .. import exceptions, options
from ..configurations.formulation import Formulation, Absorb
from ..configurations.iteration import Iteration
from ..primitives import Container
from ..utilities.algebra import precisely_identify_collinearity, precisely_identify_singularity, precisely_identify_psd
from ..utilities.basics import (
    Array, Bounds, Error, Groups, RecArray, StringRepresentation, format_number, format_table, get_indices, output, warn
)


class Economy(Container, StringRepresentation):
    """An abstract economy underlying the BLP model."""

    product_formulations: Sequence[Optional[Formulation]]
    agent_formulation: Optional[Formulation]
    rc_types: List[str]
    epsilon_scale: float
    costs_type: str
    unique_market_ids: Array
    unique_firm_ids: Array
    unique_nesting_ids: Array
    unique_product_ids: Array
    unique_agent_ids: Array
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
    MC: int
    ED: int
    ES: int
    H: int
    _market_indices: Dict[Hashable, int]
    _product_market_indices: Dict[Hashable, Array]
    _agent_market_indices: Dict[Hashable, Array]
    _max_J: int
    _max_I: int
    _absorb_demand_ids: Optional[Absorb]
    _absorb_supply_ids: Optional[Absorb]

    @abc.abstractmethod
    def __init__(
            self, product_formulations: Sequence[Optional[Formulation]], agent_formulation: Optional[Formulation],
            products: RecArray, agents: RecArray, rc_types: Optional[Sequence[str]], epsilon_scale: float,
            costs_type: str) -> None:
        """Store information about formulations and data. Any fixed effects should be absorbed after initialization."""

        # store data and formulations
        super().__init__(products, agents)
        self.product_formulations = product_formulations
        self.agent_formulation = agent_formulation

        # identify unique markets, nests, products, and agents
        self.unique_market_ids = np.unique(self.products.market_ids.flatten())
        self.unique_firm_ids = np.unique(self.products.firm_ids.flatten())
        self.unique_nesting_ids = np.unique(self.products.nesting_ids.flatten())
        self.unique_product_ids = np.unique(self.products.product_ids.flatten())
        self.unique_agent_ids = np.unique(self.agents.agent_ids.flatten())

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
        self.MC = self.products.ZC.shape[1]
        self.ED = self.products.demand_ids.shape[1]
        self.ES = self.products.supply_ids.shape[1]
        self.H = self.unique_nesting_ids.size

        # identify market indices
        self._market_indices = {t: i for i, t in enumerate(self.unique_market_ids)}
        self._product_market_indices = get_indices(self.products.market_ids)
        self._agent_market_indices = get_indices(self.agents.market_ids)

        # identify the largest number of products and agents in a market
        self._max_J = max(i.size for i in self._product_market_indices.values())
        self._max_I = max(i.size for i in self._agent_market_indices.values())

        # construct fixed effect absorption functions
        self._absorb_demand_ids = self._absorb_supply_ids = None
        if self.ED > 0:
            assert product_formulations[0] is not None
            self._absorb_demand_ids = product_formulations[0]._build_absorb(self.products.demand_ids)
        if self.ES > 0:
            assert product_formulations[2] is not None
            self._absorb_supply_ids = product_formulations[2]._build_absorb(self.products.supply_ids)

        # validate random coefficient types
        if rc_types is None:
            self.rc_types = ['linear'] * self.K2
        else:
            if not isinstance(rc_types, collections.abc.Sequence):
                raise TypeError("rc_types must be None or a sequence.")
            if len(rc_types) != self.K2:
                raise ValueError(f"rc_types must be None or a sequence of length {self.K2}.")
            if any(d not in {'linear', 'log', 'logit'} for d in rc_types):
                raise TypeError("rc_types must be None or a sequence of 'linear', 'log', or 'logit' strings.")
            self.rc_types = list(rc_types)

        # validate the scale of epsilon
        if not isinstance(epsilon_scale, (int, float)) or epsilon_scale <= 0:
            raise ValueError("epsilon_scale must be a positive float.")
        if epsilon_scale != 1 and self.H > 0:
            raise ValueError("epsilon_scale must equal 1 when there are nesting groups.")
        self.epsilon_scale = float(epsilon_scale)

        # validate the type of marginal costs
        if costs_type not in {'linear', 'log'}:
            raise ValueError("costs_type must be 'linear' or 'log'.")
        self.costs_type = costs_type

    def __str__(self) -> str:
        """Format economy information as a string."""
        return "\n\n".join([self._format_dimensions(), self._format_formulations()])

    def _format_dimensions(self) -> str:
        """Format information about the nonzero dimensions of the economy as a string."""
        header: List[str] = []
        values: List[str] = []
        for key in ['T', 'N', 'F', 'I', 'K1', 'K2', 'K3', 'D', 'MD', 'MS', 'MC', 'ED', 'ES', 'H']:
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
            (self._X3_formulations, f"X3: {self.costs_type.title()} Cost Characteristics"),
            (self._demographics_formulations, "d: Demographics")
        ]
        data: List[List[str]] = []
        for formulations, name in named_formulations:
            if any(formulations):
                data.append([name] + [str(f) for f in formulations])

        # construct the header
        max_formulations = max(len(r[1:]) for r in data)
        header = ["Column Indices:"] + [f" {i} " for i in range(max_formulations)]

        return format_table(header, *data, title="Formulations")

    def _detect_collinearity(self, added_exogenous: bool) -> None:
        """Detect any collinearity issues in product data matrices."""

        # skip collinearity checking when it is disabled via zero tolerances
        if max(options.collinear_atol, options.collinear_rtol) <= 0:
            return

        # collect labels for columns of matrices that will be checked for collinearity issues
        matrix_labels = {
            'X1': [str(f) for f in self._X1_formulations],
            'X2': [str(f) for f in self._X2_formulations],
            'X3': [str(f) for f in self._X3_formulations],
            'ZD': [str(f) for f in self._X1_formulations if 'prices' not in f.names] if added_exogenous else [],
            'ZS': [str(f) for f in self._X3_formulations if 'shares' not in f.names] if added_exogenous else [],
        }
        matrix_labels.update({
            'ZD': [f'demand_instruments{i}' for i in range(self.MD - len(matrix_labels['ZD']))] + matrix_labels['ZD'],
            'ZS': [f'supply_instruments{i}' for i in range(self.MS - len(matrix_labels['ZS']))] + matrix_labels['ZS'],
            'ZC': [f'covariance_instruments{i}' for i in range(self.MC)],
        })

        # check each matrix for collinearity
        for name, labels in matrix_labels.items():
            collinear, successful = precisely_identify_collinearity(self.products[name])
            common_message = "To disable collinearity checks, set options.collinear_atol = options.collinear_rtol = 0."
            if (self.ED > 0 and name in {'X1', 'ZD'}) or (self.ES > 0 and name in {'X3', 'ZS'}):
                common_message = f"Absorbed fixed effects may be creating collinearity problems. {common_message}"
            if not successful:
                warn(
                    f"Failed to compute the QR decomposition of {name} while checking for collinearity issues. "
                    f"{common_message}"
                )
            if collinear.any():
                collinear_labels = ", ".join(l for l, c in zip(labels, collinear) if c)
                warn(
                    f"Detected collinearity issues with [{collinear_labels}] and at least one other column in {name}. "
                    f"{common_message}"
                )

    @staticmethod
    def _detect_singularity(matrix: Array, name: str) -> None:
        """Detect any singularity issues in a matrix."""
        singular, successful, condition = precisely_identify_singularity(matrix)
        common_message = "To disable singularity checks, set options.singular_tol = numpy.inf."
        if not successful:
            warn(f"Failed to compute the condition number of {name} while checking for singularity. {common_message}")
        if singular:
            prefix = "nearly " if condition < np.inf else ""
            warn(
                f"Detected that {name} is {prefix}singular with condition number {format_number(condition).strip()}. "
                f"{common_message}"
            )

    @staticmethod
    def _require_psd(matrix: Array, name: str) -> None:
        """Require that a matrix is PSD."""
        psd, successful = precisely_identify_psd(matrix)
        common_message = "To disable PSD checks, set options.psd_atol = options.psd_rtol = numpy.inf."
        if not successful:
            raise ValueError(f"Failed to compute the SVD of {name} while checking that it is PSD. {common_message}")
        if not psd:
            raise ValueError(f"{name} must be a PSD matrix. {common_message}")

    @staticmethod
    def _handle_errors(errors: List[Error], error_behavior: str = 'raise') -> None:
        """Either raise or output information about any errors."""
        if errors:
            if error_behavior == 'raise':
                raise exceptions.MultipleErrors(errors)
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

    def _validate_name(self, name: Optional[str], none_valid: bool = True) -> None:
        """Validate that a name is either None or corresponds to a variable in X1, X2, or X3."""
        if name is None and none_valid:
            return
        formulations = self._X1_formulations + self._X2_formulations + self._X3_formulations
        names = {n for f in formulations for n in f.names}
        if name not in names:
            raise NameError(f"'{name}' is not None or one of the underlying variables, {list(sorted(names))}.")

    def _validate_product_ids_index(self, product_ids_index: int) -> None:
        """Validate that a product IDs index is valid."""
        if not isinstance(product_ids_index, int) or product_ids_index < 0:
            raise ValueError("The product IDs index must be a non-negative int.")
        if self.products.product_ids.size == 0:
            raise ValueError("Since the product IDs index is not None, product_data must have product_ids.")
        max_index = self.products.product_ids.shape[1] - 1
        if not 0 <= product_ids_index <= max_index:
            raise ValueError(f"The product IDs index should be at most {max_index}.")

    def _coerce_optional_firm_ids(self, firm_ids: Optional[Any], market_ids: Optional[Array] = None) -> Array:
        """Coerce optional array-like firm IDs into a column vector and validate it. By default, assume that firm IDs
        are for all markets.
        """
        if firm_ids is None:
            return None
        firm_ids = np.c_[np.asarray(firm_ids, np.object_)]
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
        if ownership is None:
            return None

        ownership = np.c_[np.asarray(ownership, options.dtype)]

        rows = self.N
        columns = self._max_J
        if market_ids is not None:
            rows = columns = 0
            for t in market_ids:
                size = self._product_market_indices[t].size
                rows += size
                columns = max(columns, size)

        if ownership.shape != (rows, columns):
            raise ValueError(f"ownership must be None or a {rows} by {columns} matrix.")

        return ownership

    @staticmethod
    def _coerce_optional_delta_iteration(iteration: Optional[Iteration]) -> Iteration:
        """Validate or choose a default configuration for iterating over the mean utility."""
        if iteration is None:
            iteration = Iteration('squarem', {'atol': 1e-14})
        elif not isinstance(iteration, Iteration):
            raise TypeError("iteration must be None or an Iteration instance.")
        return iteration

    @staticmethod
    def _coerce_optional_prices_iteration(iteration: Optional[Iteration]) -> Iteration:
        """Validate or choose a default configuration for iteration over prices."""
        if iteration is None:
            iteration = Iteration('simple', {'atol': 1e-12})
        elif not isinstance(iteration, Iteration):
            raise ValueError("iteration must be None or an Iteration.")
        return iteration

    def _validate_fp_type(self, fp_type: str) -> None:
        """Validate that the delta fixed point type is supported."""
        if fp_type not in {'safe_linear', 'linear', 'safe_nonlinear', 'nonlinear'}:
            raise ValueError("fp_type must be 'safe_linear', 'linear', 'safe_nonlinear', or 'nonlinear'.")
        if fp_type in {'safe_nonlinear', 'nonlinear'} and self.epsilon_scale != 1:
            raise ValueError("When epsilon_scale is not 1, fp_type must be 'safe_linear' or 'linear'.")

    @staticmethod
    def _coerce_optional_bounds(bounds: Optional[Tuple[Any, Any]], name: str) -> Bounds:
        """Validate or choose default bounds for some object."""
        if bounds is None:
            return -np.inf, +np.inf
        if len(bounds) != 2:
            raise ValueError(f"{name} must be a tuple of the form (lb, ub).")
        bounds = (np.asarray(bounds[0], options.dtype), np.asarray(bounds[1], options.dtype))
        bounds[0][np.isnan(bounds[0])] = -np.inf
        bounds[1][np.isnan(bounds[1])] = +np.inf
        if bounds[0].size != 1:
            raise ValueError(f"The lower bound in {name} must be None or a float.")
        if bounds[1].size != 1:
            raise ValueError(f"The upper bound in {name} must be None or a float.")
        if bounds[0] > bounds[1]:
            raise ValueError(f"The lower bound in {name} cannot be larger than the upper bound.")
        return bounds

    def _compute_true_X1(self, data_override: Optional[Mapping] = None, index: Optional[Array] = None) -> Array:
        """Compute X1 or columns of X1 without any absorbed demand-side fixed effects."""
        if index is None:
            index = np.ones(self.K1, np.bool_)
        if self.ED == 0 and not data_override:
            return self.products.X1[:, index]

        # compute X1 column-by-column
        columns = []
        for include, formulation in zip(index, self._X1_formulations):
            if include:
                column = formulation.evaluate(self.products, data_override)
                columns.append(np.broadcast_to(column, (self.N, 1)).astype(options.dtype))

        return np.column_stack(columns)

    def _compute_true_X3(self, data_override: Optional[Mapping] = None, index: Optional[Array] = None) -> Array:
        """Compute X3 or columns of X3 without any absorbed supply-side fixed effects."""
        if index is None:
            index = np.ones(self.K3, np.bool_)
        if self.ES == 0 and not data_override:
            return self.products.X3[:, index]

        # compute X3 column-by-column
        columns = []
        for include, formulation in zip(index, self._X3_formulations):
            if include:
                columns.append(formulation.evaluate(self.products, data_override) * np.ones((self.N, 1)))

        return np.column_stack(columns)

    def _compute_logit_delta(self, rho: Array) -> Array:
        """Compute the mean utility that solves the simple logit (or nested logit) model."""
        log_shares = np.log(self.products.shares)

        # compute delta market-by-market
        delta = log_shares.copy()
        for t, indices_t in self._product_market_indices.items():
            shares_t = self.products.shares[indices_t]
            log_outside_share_t = np.log(1 - shares_t.sum())
            delta[indices_t] -= log_outside_share_t
            if self.H > 0:
                log_shares_t = log_shares[indices_t]
                groups_t = Groups(self.products.nesting_ids[indices_t])
                log_group_shares_t = np.log(groups_t.expand(groups_t.sum(shares_t)))
                if rho.size == 1:
                    rho_t = np.full_like(shares_t, float(rho))
                else:
                    rho_t = groups_t.expand(rho[np.searchsorted(self.unique_nesting_ids, groups_t.unique)])
                delta[indices_t] -= rho_t * (log_shares_t - log_group_shares_t)

        # delta needs to be scaled if the error term is scaled
        if self.epsilon_scale != 1:
            assert self.H == 0
            delta *= self.epsilon_scale

        return delta
