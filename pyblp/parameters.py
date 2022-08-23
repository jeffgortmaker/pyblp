"""Parameters underlying the BLP model."""

import abc
from typing import Any, Type, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union

import numpy as np

from . import options
from .configurations.formulation import ColumnFormulation
from .utilities.algebra import vech
from .utilities.basics import Array, Bounds, format_number, format_se, format_table


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .economies.economy import Economy  # noqa
    from .markets.market import Market  # noqa
    from .primitives import Container


class Parameter(abc.ABC):
    """Information about a single parameter."""

    location: Sequence
    value: Optional[float]

    def __init__(self, location: Sequence, bounds: Bounds) -> None:
        """Store the information and determine whether the parameter is fixed or unfixed."""
        self.location = location
        self.value = bounds[0][location] if bounds[0][location] == bounds[1][location] else None


class Coefficient(Parameter):
    """Information about a single coefficient parameter in sigma, pi, beta, or gamma."""

    @abc.abstractmethod
    def get_product_formulation(self, container: 'Container') -> ColumnFormulation:
        """Get the product formulation associated with the parameter."""

    @abc.abstractmethod
    def get_product_characteristic(self, market: 'Market') -> Array:
        """Get the product characteristic associated with the parameter."""


class NonlinearCoefficient(Coefficient):
    """Information about a single nonlinear parameter in sigma or pi."""

    def get_product_formulation(self, container: 'Container') -> ColumnFormulation:
        """Get the product formulation associated with the parameter."""
        return container._X2_formulations[self.location[0]]

    def get_product_characteristic(self, market: 'Market') -> Array:
        """Get the product characteristic associated with the parameter."""
        return market.products.X2[:, [self.location[0]]]

    def get_rc_type(self, market: 'Market') -> str:
        """Get the random coefficient type associated with the parameter."""
        return market.parameters.rc_types[self.location[0]]

    @abc.abstractmethod
    def get_agent_characteristic(self, market: 'Market') -> Array:
        """Get the agent characteristic associated with the parameter."""


class SigmaParameter(NonlinearCoefficient):
    """Information about a single parameter in sigma."""

    def get_agent_characteristic(self, market: 'Market') -> Array:
        """Get the agent characteristic associated with the parameter."""
        return market.agents.nodes[:, [self.location[1]]]


class PiParameter(NonlinearCoefficient):
    """Information about a single parameter in pi."""

    def get_agent_characteristic(self, market: 'Market') -> Array:
        """Get the agent characteristic associated with the parameter."""
        if len(market.agents.demographics.shape) == 3:
            return market.agents.demographics[:, self.location[1]]
        return market.agents.demographics[:, [self.location[1]]]


class RhoParameter(Parameter):
    """Information about a single parameter in rho."""

    @abc.abstractmethod
    def get_group_associations(self, market: 'Market') -> Array:
        """Get an indicator for which groups are associated with the parameter."""


class AllGroupsRhoParameter(RhoParameter):
    """Information about a rho parameter for all groups."""

    def get_group_associations(self, market: 'Market') -> Array:
        """Get an indicator for all groups."""
        return np.ones((market.groups.group_count, 1), options.dtype)


class OneGroupRhoParameter(RhoParameter):
    """Information about a rho parameter for a single group."""

    def get_group_associations(self, market: 'Market') -> Array:
        """Get an indicator for the group associated with the parameter."""
        group_associations = np.zeros((market.groups.group_count, 1), options.dtype)
        group_associations[market.groups.unique == market.unique_nesting_ids[self.location[0]]] = 1
        return group_associations


class LinearCoefficient(Coefficient):
    """Information about a single linear parameter in beta or gamma."""

    def get_product_formulation(self, container: 'Container') -> ColumnFormulation:
        """Get the product formulation associated with the parameter."""
        return container._X2_formulations[self.location[0]]

    def get_product_characteristic(self, market: 'Market') -> Array:
        """Get the product characteristic associated with the parameter."""
        x = self.get_product_formulation(market).evaluate(market.products)
        return np.broadcast_to(x, (market.products.shape[0], 1)).astype(options.dtype)


class BetaParameter(LinearCoefficient):
    """Information about a single linear parameter in beta."""

    def get_product_formulation(self, container: 'Container') -> ColumnFormulation:
        """Get the product formulation associated with the parameter."""
        return container._X1_formulations[self.location[0]]


class GammaParameter(LinearCoefficient):
    """Information about a single linear parameter in gamma."""

    def get_product_formulation(self, container: 'Container') -> ColumnFormulation:
        """Get the product formulation associated with the parameter."""
        return container._X3_formulations[self.location[0]]


class Parameters(object):
    """Information about sigma, pi, rho, beta, and gamma."""

    sigma_labels: List[str]
    pi_labels: List[str]
    rho_labels: List[str]
    beta_labels: List[str]
    gamma_labels: List[str]
    theta_labels: List[str]
    rc_types: List[str]
    sigma: Array
    sigma_squared: Array
    pi: Array
    rho: Array
    beta: Array
    gamma: Array
    sigma_bounds: Bounds
    pi_bounds: Bounds
    rho_bounds: Bounds
    beta_bounds: Bounds
    gamma_bounds: Bounds
    diagonal_sigma: bool
    nonzero_sigma_index: Array
    alpha_index: Array
    endogenous_gamma_index: Array
    eliminated_beta_index: Array
    eliminated_gamma_index: Array
    eliminated_alpha_index: Array
    eliminated_endogenous_gamma_index: Array
    fixed: List[Parameter]
    unfixed: List[Parameter]
    eliminated: List[Parameter]
    P: int
    any_bounds: bool

    def __init__(
            self, economy: 'Economy', sigma: Optional[Any] = None, pi: Optional[Any] = None, rho: Optional[Any] = None,
            beta: Optional[Any] = None, gamma: Optional[Any] = None, sigma_bounds: Optional[Tuple[Any, Any]] = None,
            pi_bounds: Optional[Tuple[Any, Any]] = None, rho_bounds: Optional[Tuple[Any, Any]] = None,
            beta_bounds: Optional[Tuple[Any, Any]] = None, gamma_bounds: Optional[Tuple[Any, Any]] = None,
            bounded: bool = False, allow_linear_nans: bool = False, check_alpha: bool = True) -> None:
        """Coerce parameters into usable formats before storing information about fixed (equal bounds) and unfixed
        (unequal bounds) elements of sigma, pi, rho, beta, and gamma. Also store information about eliminated
        (concentrated out) parameters in beta and gamma. If allow_linear_nans is True, allow null linear parameters in
        order to denote those parameters that will be concentrated out. If check_alpha is True, check that alpha isn't
        concentrated out when a supply side is included.
        """

        # store labels
        self.sigma_labels = [str(f) for f in economy._X2_formulations]
        self.pi_labels = [str(f) for f in economy._demographics_formulations]
        self.rho_labels = [str(i) for i in economy.unique_nesting_ids]
        self.beta_labels = [str(f) for f in economy._X1_formulations]
        self.gamma_labels = [str(f) for f in economy._X3_formulations]

        # store types
        self.rc_types = economy.rc_types

        # validate and store parameters
        self.sigma = self.initialize_matrix("sigma", "X2 was formulated", sigma, [(economy.K2, economy.K2)])
        self.pi = self.initialize_matrix("pi", "demographics were formulated", pi, [(economy.K2, economy.D)])
        self.rho = self.initialize_matrix("rho", "nesting IDs were specified", rho, [(economy.H, 1), (1, 1)])
        self.beta = self.initialize_matrix("beta", "X1 was formulated", beta, [(economy.K1, 1)], allow_linear_nans)
        self.gamma = self.initialize_matrix("gamma", "X3 was formulated", gamma, [(economy.K3, 1)], allow_linear_nans)

        # fill the upper triangle of sigma with zeros
        self.sigma[np.triu_indices(economy.K2, 1)] = 0

        # construct sigma squared (the underlying covariance matrix)
        self.sigma_squared = self.sigma @ self.sigma.T

        # identify whether sigma is a diagonal matrix
        self.diagonal_sigma = not (np.tril(self.sigma, k=-1) != 0).any()

        # identify the index of nonzero columns in sigma
        self.nonzero_sigma_index = np.any(self.sigma != 0, axis=0)

        # identify the index of alpha in beta
        self.alpha_index = np.zeros_like(self.beta, np.bool_)
        for k, formulation in enumerate(economy._X1_formulations):
            if 'prices' in formulation.names:
                self.alpha_index[k] = True

        # identify the index of analogous parameters in gamma
        self.endogenous_gamma_index = np.zeros_like(self.gamma, np.bool_)
        for k, formulation in enumerate(economy._X3_formulations):
            if 'shares' in formulation.names:
                self.endogenous_gamma_index[k] = True

        # identify eliminated indexes
        self.eliminated_beta_index = np.isnan(self.beta)
        self.eliminated_gamma_index = np.isnan(self.gamma)
        self.eliminated_alpha_index = np.isnan(self.beta) & self.alpha_index
        self.eliminated_endogenous_gamma_index = np.isnan(self.gamma) & self.endogenous_gamma_index

        # there should be at least as many integration node columns as nonzero sigma columns
        if economy.agents.nodes.shape[1] < self.nonzero_sigma_index.sum():
            raise ValueError(
                f"The number of columns of integration nodes, {economy.agents.nodes.shape[1]}, is smaller than the "
                f"number of columns in sigma with at least one nonzero parameter, {self.nonzero_sigma_index.sum()}."
            )

        # alpha cannot be concentrated out if there's a supply side
        if check_alpha and economy.K3 > 0:
            for formulation, eliminated in zip(economy._X1_formulations, self.eliminated_beta_index.flatten()):
                if 'prices' in formulation.names and eliminated:
                    raise ValueError(
                        f"A supply side was specified, so alpha should not be concentrated out. That is, initial "
                        f"values should be specified for all parameters in beta on X1 characteristics involving prices."
                    )

        # validate and store parameter bounds
        self.sigma_bounds = self.initialize_bounds("sigma", self.sigma, sigma_bounds, bounded)
        self.pi_bounds = self.initialize_bounds("pi", self.pi, pi_bounds, bounded)
        self.rho_bounds = self.initialize_bounds("rho", self.rho, rho_bounds, bounded)
        self.beta_bounds = self.initialize_bounds("beta", self.beta, beta_bounds, bounded)
        self.gamma_bounds = self.initialize_bounds("gamma", self.gamma, gamma_bounds, bounded)

        # identify the type of rho parameter that has been specified (either for all groups or just one)
        rho_type = OneGroupRhoParameter if self.rho.size > 1 else AllGroupsRhoParameter

        # store information about fixed and unfixed parameters
        self.fixed: List[Parameter] = []
        self.unfixed: List[Parameter] = []
        self.eliminated: List[Parameter] = []
        self.store(SigmaParameter, zip(*np.tril_indices_from(self.sigma)), self.sigma_bounds)
        self.store(PiParameter, np.ndindex(self.pi.shape), self.pi_bounds)
        self.store(rho_type, np.ndindex(self.rho.shape), self.rho_bounds)
        self.store(BetaParameter, np.ndindex(self.beta.shape), self.beta_bounds, self.eliminated_beta_index)
        self.store(GammaParameter, np.ndindex(self.gamma.shape), self.gamma_bounds, self.eliminated_gamma_index)

        # count the number of unfixed parameters
        self.P = len(self.unfixed)

        # define default bounds
        if bounded:
            for parameter in self.unfixed:
                location = parameter.location
                if isinstance(parameter, SigmaParameter) and sigma_bounds is None and location[0] == location[1]:
                    self.sigma_bounds[0][location] = min(0, self.sigma[location])
                elif isinstance(parameter, RhoParameter) and rho_bounds is None:
                    self.rho_bounds[0][location] = min(0, self.rho[location])
                    self.rho_bounds[1][location] = max(0.99, self.rho[location])

        # identify whether there are any bounds
        self.any_bounds = np.isfinite(self.compress_bounds()).any()

        # define labels for theta
        self.theta_labels = self.compress_labels()

    @staticmethod
    def initialize_matrix(
            name: str, condition_name: str, values: Optional[Any], shapes: Sequence[Tuple[int, int]],
            allow_nans: bool = False) -> Array:
        """Validate and structure a parameter matrix, which can be a number of different shapes."""

        # allow the matrix to be all nans if it is to be entirely concentrated out
        matrix = np.full(shapes[0], np.nan, options.dtype)
        if allow_nans and values is None:
            return matrix

        # validate the matrix
        if values is not None:
            matrix = np.c_[np.asarray(values, options.dtype)]
        if (values is not None and matrix.size > 0) != all(r * c > 0 for r, c in shapes):
            raise ValueError(f"{name} should be specified only when {condition_name}.")
        if matrix.shape not in shapes:
            shape_names = " or ".join(f"{r} by {c}" for r, c in shapes)
            raise ValueError(f"{name} must be {shape_names}.")
        if not allow_nans and np.isnan(matrix).any():
            raise ValueError(f"{name} should not have any null values.")
        return matrix

    @staticmethod
    def initialize_bounds(name: str, matrix: Array, bound_values: Optional[Tuple[Any, Any]], bounded: bool) -> Bounds:
        """Validate and structure parameter bounds."""

        # by default, initialize non-binding bounds
        bounds = (np.full_like(matrix, -np.inf, options.dtype), np.full_like(matrix, +np.inf, options.dtype))

        # validate any specified bounds
        if matrix.size > 0 and bound_values is not None:
            if len(bound_values) != 2:
                raise ValueError(f"{name}_bounds must be a tuple of the form (lb, ub).")
            bounds = (
                np.c_[np.asarray(bound_values[0], options.dtype)], np.c_[np.asarray(bound_values[1], options.dtype)]
            )
            bounds[0][np.isnan(bounds[0])] = -np.inf
            bounds[1][np.isnan(bounds[1])] = +np.inf
            if bounds[0].shape != matrix.shape:
                raise ValueError(f"The lower bound in {name}_bounds does not have the same shape as {name}.")
            if bounds[1].shape != matrix.shape:
                raise ValueError(f"The upper bound in {name}_bounds does not have the same shape as {name}.")
            with np.errstate(invalid='ignore'):
                if ((matrix < bounds[0]) | (matrix > bounds[1])).any():
                    raise ValueError(f"{name} must be within its bounds.")

        # keep only bounds that fix parameters if bounding isn't supported
        if not bounded:
            unfixed = bounds[0] != bounds[1]
            bounds[0][unfixed] = -np.inf
            bounds[1][unfixed] = +np.inf

        # fix parameters set to zero
        zeros_index = matrix == 0
        bounds[0][zeros_index] = bounds[1][zeros_index] = 0
        return bounds

    def store(
            self, parameter_type: Type[Parameter], locations: Iterable[Tuple[int, int]], bounds: Bounds,
            eliminated_index: Optional[Array] = None) -> None:
        """Store fixed, unfixed, and eliminated parameters in lists."""
        for location in locations:
            parameter = parameter_type(location, bounds)
            if eliminated_index is not None and eliminated_index[location]:
                self.eliminated.append(parameter)
            elif parameter.value is None:
                self.unfixed.append(parameter)
            else:
                self.fixed.append(parameter)

    def format(self, title: str) -> str:
        """Format fixed and unfixed parameter values as a string."""
        return self.format_theta_parameters(
            title, self.sigma, self.pi, self.rho, self.beta, self.gamma, self.sigma_squared
        )

    def format_lower_bounds(self, title: str) -> str:
        """Format lower bounds for fixed and unfixed parameter values as a string."""
        return self.format_theta_parameters(
            title, self.sigma_bounds[0], self.pi_bounds[0], self.rho_bounds[0], self.beta_bounds[0],
            self.gamma_bounds[0]
        )

    def format_upper_bounds(self, title: str) -> str:
        """Format upper bounds for fixed and unfixed parameter values as a string."""
        return self.format_theta_parameters(
            title, self.sigma_bounds[1], self.pi_bounds[1], self.rho_bounds[1], self.beta_bounds[1],
            self.gamma_bounds[1]
        )

    def format_theta_parameters(
            self, title: str, sigma_like: Array, pi_like: Array, rho_like: Array, beta_like: Array,
            gamma_like: Array, sigma_squared_like: Optional[Array] = None) -> str:
        """Format fixed and unfixed parameter-like values as a string. Skip sections of parameters without any that
        are in theta.
        """
        items = [
            (NonlinearCoefficient, lambda: self.format_nonlinear_coefficients(
                title, sigma_like, pi_like, sigma_squared_like
            )),
            (RhoParameter, lambda: self.format_rho(title, rho_like)),
            (BetaParameter, lambda: self.format_beta(title, beta_like)),
            (GammaParameter, lambda: self.format_gamma(title, gamma_like))
        ]
        return "\n\n".join(f() for t, f in items if any(isinstance(p, t) for p in self.fixed + self.unfixed))

    def format_estimates(
            self, title: str, sigma: Array, pi: Array, rho: Array, beta: Array, gamma: Array, sigma_squared: Array,
            sigma_se: Array, pi_se: Array, rho_se: Array, beta_se: Array, gamma_se: Array,
            sigma_squared_se: Array) -> str:
        """Format all estimates and their standard errors as a string."""
        items = [
            (sigma, lambda: self.format_nonlinear_coefficients(
                title, sigma, pi, sigma_squared, sigma_se, pi_se, sigma_squared_se
            )),
            (rho, lambda: self.format_rho(title, rho, rho_se)),
            (beta, lambda: self.format_beta(title, beta, beta_se)),
            (gamma, lambda: self.format_gamma(title, gamma, gamma_se))
        ]
        return "\n\n".join(f() for e, f in items if e.size > 0)

    def format_rho(self, title: str, rho_like: Array, rho_se_like: Optional[Array] = None) -> str:
        """Format a vector (and optional standard errors) of the same size as rho as a string."""
        header = self.rho_labels if rho_like.size > 1 else ["All Groups"]
        return self.format_vector(f"Rho {title}", RhoParameter, header, rho_like, rho_se_like)

    def format_beta(self, title: str, beta_like: Array, beta_se_like: Optional[Array] = None) -> str:
        """Format a vector (and optional standard errors) of the same size as beta as a string."""
        return self.format_vector(f"Beta {title}", BetaParameter, self.beta_labels, beta_like, beta_se_like)

    def format_gamma(self, title: str, gamma_like: Array, gamma_se_like: Optional[Array] = None) -> str:
        """Format a vector (and optional standard errors) of the same size as gamma as a string."""
        return self.format_vector(f"Gamma {title}", BetaParameter, self.gamma_labels, gamma_like, gamma_se_like)

    def format_vector(
            self, title: str, parameter_type: Type[Union[RhoParameter, LinearCoefficient]], header: List[str],
            vector: Array, vector_se: Optional[Array] = None) -> str:
        """Format a vector (and optional standard errors) as a string."""
        data = [[format_number(x) for x in vector]]
        if vector_se is not None:
            fixed_indices = {p.location[0] for p in self.fixed if isinstance(p, parameter_type)}
            data.append(["" if i in fixed_indices else format_se(x) for i, x in enumerate(vector_se)])
        return format_table(header, *data, title=title)

    def format_nonlinear_coefficients(
            self, title: str, sigma_like: Array, pi_like: Array, sigma_squared_like: Optional[Array] = None,
            sigma_se: Optional[Array] = None, pi_se: Optional[Array] = None,
            sigma_squared_se: Optional[Array] = None) -> str:
        """Format matrices (and optional standard errors) of the same size as sigma and pi as a string."""

        # determine whether a types column is necessary
        rc_types_column = any(d != 'linear' for d in self.rc_types)

        # only add sigma squared columns if the covariance matrix has off-diagonal terms
        if self.diagonal_sigma:
            sigma_squared_like = sigma_squared_se = None

        # construct the header
        line_indices: Set[int] = {0} if rc_types_column else set()
        header = (["Types:"] if rc_types_column else []) + ["Sigma:"] + self.sigma_labels
        if sigma_squared_like is not None:
            line_indices.add(len(header) - 1)
            header.extend(["Sigma Squared:"] + self.sigma_labels)
        if self.pi_labels:
            line_indices.add(len(header) - 1)
            header.extend(["Pi:"] + self.pi_labels)

        # construct the data
        data: List[List[str]] = []
        for row_index, (row_rc_type, row_label) in enumerate(zip(self.rc_types, self.sigma_labels)):
            # add a row of values
            values_row = ([row_rc_type.title()] if rc_types_column else []) + [row_label]
            for column_index in range(row_index + 1):
                values_row.append(format_number(sigma_like[row_index, column_index]))
            values_row.extend([""] * (sigma_like.shape[1] - row_index - 1))
            if sigma_squared_like is not None:
                values_row.append(row_label)
                for column_index in range(sigma_squared_like.shape[1]):
                    values_row.append(format_number(sigma_squared_like[row_index, column_index]))
            if pi_like.shape[1] > 0:
                values_row.append(row_label)
                for column_index in range(pi_like.shape[1]):
                    values_row.append(format_number(pi_like[row_index, column_index]))
            data.append(values_row)

            # only add a row of standard errors if standard errors are specified
            if sigma_se is None:
                continue

            # determine which columns in this row correspond to unfixed parameters
            relevant_unfixed = {p for p in self.unfixed if p.location[0] == row_index}
            unfixed_sigma_indices = {p.location[1] for p in relevant_unfixed if isinstance(p, SigmaParameter)}
            unfixed_pi_indices = {p.location[1] for p in relevant_unfixed if isinstance(p, PiParameter)}

            # add a row of standard errors
            se_row = (2 if rc_types_column else 1) * [""]
            for column_index in range(row_index + 1):
                se = sigma_se[row_index, column_index]
                se_row.append(format_se(se) if column_index in unfixed_sigma_indices else "")
            se_row.extend([""] * (sigma_like.shape[1] - row_index - 1))
            if sigma_squared_se is not None:
                se_row.append("")
                for column_index in range(sigma_squared_se.shape[1]):
                    se = sigma_squared_se[row_index, column_index]
                    se_row.append(format_se(se))
            if pi_se is not None and pi_se.shape[1] > 0:
                se_row.append("")
                for column_index in range(pi_se.shape[1]):
                    se = pi_se[row_index, column_index]
                    se_row.append(format_se(se) if column_index in unfixed_pi_indices else "")
            data.append(se_row)

            # add a blank row to separate the standard errors from the next row of values
            if row_index < sigma_like.shape[1] - 1:
                data.append([""] * len(se_row))

        return format_table(header, *data, title=f"Nonlinear Coefficient {title}", line_indices=line_indices)

    def compress(self) -> Array:
        """Compress the initial parameters into theta."""
        items = [
            (SigmaParameter, self.sigma),
            (PiParameter, self.pi),
            (RhoParameter, self.rho),
            (BetaParameter, self.beta),
            (GammaParameter, self.gamma),
        ]
        return np.r_[[v[p.location] for t, v in items for p in self.unfixed if isinstance(p, t)]]

    def compress_bounds(self) -> List[Tuple[float, float]]:
        """Compress parameter bounds into a list of (lb, ub) tuples for theta."""
        items = [
            (SigmaParameter, self.sigma_bounds),
            (PiParameter, self.pi_bounds),
            (RhoParameter, self.rho_bounds),
            (BetaParameter, self.beta_bounds),
            (GammaParameter, self.gamma_bounds),
        ]
        return [(l[p.location], u[p.location]) for t, (l, u) in items for p in self.unfixed if isinstance(p, t)]

    def compress_labels(self) -> List[str]:
        """Compress labels into a list of labels for theta."""
        items = [
            (SigmaParameter, np.array([[f'{k1} x {k2}' for k2 in self.sigma_labels] for k1 in self.sigma_labels])),
            (PiParameter, np.array([[f'{k1} x {k2}' for k2 in self.pi_labels] for k1 in self.sigma_labels])),
            (RhoParameter, np.c_[np.array(self.rho_labels)]),
            (BetaParameter, np.c_[np.array(self.beta_labels)]),
            (GammaParameter, np.c_[np.array(self.gamma_labels)]),
        ]
        return [v[p.location] for t, v in items for p in self.unfixed if isinstance(p, t)]

    def expand(self, theta_like: Array, nullify: bool = False) -> Tuple[Array, Array, Array, Array, Array]:
        """Recover matrices of the same size as parameter matrices from a vector of the same size as theta. By default,
        fill elements corresponding to fixed parameters with their fixed values. Always fill concentrated out parameters
        with nulls.
        """
        sigma_like = np.full_like(self.sigma, np.nan)
        pi_like = np.full_like(self.pi, np.nan)
        rho_like = np.full_like(self.rho, np.nan)
        beta_like = np.full_like(self.beta, np.nan)
        gamma_like = np.full_like(self.gamma, np.nan)
        items = [
            (SigmaParameter, sigma_like),
            (PiParameter, pi_like),
            (RhoParameter, rho_like),
            (BetaParameter, beta_like),
            (GammaParameter, gamma_like),
        ]

        # fill values of unfixed parameters
        for parameter, value in zip(self.unfixed, theta_like):
            for parameter_type, values in items:
                if isinstance(parameter, parameter_type):
                    values[parameter.location] = value
                    break

        # if they aren't null, fill values of fixed parameters
        if not nullify:
            sigma_like[np.triu_indices_from(sigma_like, 1)] = 0
            for parameter in self.fixed:
                for parameter_type, values in items:
                    if isinstance(parameter, parameter_type):
                        values[parameter.location] = parameter.value
                        break

        return sigma_like, pi_like, rho_like, beta_like, gamma_like

    def extract_sigma_vector_covariances(self, theta_covariances: Array) -> Array:
        """Extract the sub-matrix of covariances for vech(sigma) from a full covariance matrix for theta."""
        assert theta_covariances.shape == (self.P, self.P)

        # identify indices in theta for elements in vech(sigma), with NaN representing fixed elements
        sigma_indices = self.expand(np.arange(self.P), nullify=True)[0]
        sigma_vector_indices = vech(sigma_indices)

        # extract corresponding rows and columns from the theta covariances, taking zeros for fixed elements
        padded_covariances = np.pad(theta_covariances, pad_width=(0, 1), mode='constant', constant_values=0)
        indices = np.nan_to_num(sigma_vector_indices, nan=self.P).astype(np.int64)
        return padded_covariances[indices, :][:, indices]
