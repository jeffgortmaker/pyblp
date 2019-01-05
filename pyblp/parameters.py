"""Parameters underlying the BLP model."""

import abc
from typing import Any, Type, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union

import numpy as np

from . import options
from .configurations.formulation import ColumnFormulation
from .utilities.basics import Array, Bounds, Groups, TableFormatter, format_number, format_se


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .economies.economy import Economy  # noqa
    from .markets.market import Market  # noqa


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
    def get_product_formulation(self, economy_or_market: Union['Economy', 'Market']) -> ColumnFormulation:
        """Get the product formulation associated with the parameter."""

    @abc.abstractmethod
    def get_product_characteristic(self, economy_or_market: Union['Economy', 'Market']) -> Array:
        """Get the product characteristic associated with the parameter."""


class NonlinearCoefficient(Coefficient):
    """Information about a single nonlinear parameter in sigma or pi."""

    def get_product_formulation(self, economy_or_market: Union['Economy', 'Market']) -> ColumnFormulation:
        """Get the product formulation associated with the parameter."""
        return economy_or_market._X2_formulations[self.location[0]]

    def get_product_characteristic(self, economy_or_market: Union['Economy', 'Market']) -> Array:
        """Get the product characteristic associated with the parameter."""
        return economy_or_market.products.X2[:, [self.location[0]]]

    @abc.abstractmethod
    def get_agent_characteristic(self, economy_or_market: Union['Economy', 'Market']) -> Array:
        """Get the agent characteristic associated with the parameter."""


class SigmaParameter(NonlinearCoefficient):
    """Information about a single parameter in sigma."""

    def get_agent_characteristic(self, economy_or_market: Union['Economy', 'Market']) -> Array:
        """Get the agent characteristic associated with the parameter."""
        return economy_or_market.agents.nodes[:, [self.location[1]]]


class PiParameter(NonlinearCoefficient):
    """Information about a single parameter in pi."""

    def get_agent_characteristic(self, economy_or_market: Union['Economy', 'Market']) -> Array:
        """Get the agent characteristic associated with the parameter."""
        return economy_or_market.agents.demographics[:, [self.location[1]]]


class RhoParameter(Parameter):
    """Information about a single parameter in rho."""

    @abc.abstractmethod
    def get_group_associations(self, groups: Groups) -> Array:
        """Get an indicator for which groups are associated with the parameter."""


class AllGroupsRhoParameter(RhoParameter):
    """Information about a rho parameter for all groups."""

    def get_group_associations(self, groups: Groups) -> Array:
        """Get an indicator for all groups."""
        return np.ones((groups.group_count, 1), options.dtype)


class OneGroupRhoParameter(RhoParameter):
    """Information about a rho parameter for a single group."""

    def get_group_associations(self, groups: Groups) -> Array:
        """Get an indicator for the group associated with the parameter."""
        group_associations = np.zeros((groups.group_count, 1), options.dtype)
        group_associations[self.location] = 1
        return group_associations


class LinearCoefficient(Coefficient):
    """Information about a single linear parameter in beta or gamma."""

    def get_product_characteristic(self, economy_or_market: Union['Economy', 'Market']) -> Array:
        """Get the product characteristic associated with the parameter."""
        x = self.get_product_formulation(economy_or_market).evaluate(economy_or_market.products)
        return np.broadcast_to(x, (economy_or_market.products.shape[0], 1)).astype(options.dtype)


class BetaParameter(LinearCoefficient):
    """Information about a single linear parameter in beta."""

    def get_product_formulation(self, economy_or_market: Union['Economy', 'Market']) -> ColumnFormulation:
        """Get the product formulation associated with the parameter."""
        return economy_or_market._X1_formulations[self.location[0]]


class GammaParameter(LinearCoefficient):
    """Information about a single linear parameter in gamma."""

    def get_product_formulation(self, economy_or_market: Union['Economy', 'Market']) -> ColumnFormulation:
        """Get the product formulation associated with the parameter."""
        return economy_or_market._X3_formulations[self.location[0]]


class Parameters(object):
    """Information about sigma, pi, rho, beta, and gamma."""

    sigma_labels: List[str]
    pi_labels: List[str]
    rho_labels: List[str]
    beta_labels: List[str]
    gamma_labels: List[str]
    sigma: Array
    pi: Array
    rho: Array
    beta: Array
    gamma: Array
    sigma_bounds: Bounds
    pi_bounds: Bounds
    rho_bounds: Bounds
    beta_bounds: Bounds
    gamma_bounds: Bounds
    alpha_index: Array
    eliminated_alpha_index: Array
    eliminated_beta_index: Array
    eliminated_gamma_index: Array
    alpha_indices: List[int]
    eliminated_alpha_indices: List[int]
    eliminated_beta_indices: List[int]
    eliminated_gamma_indices: List[int]
    fixed: List[Parameter]
    unfixed: List[Parameter]
    P: int

    def __init__(
            self, economy: 'Economy', sigma: Optional[Any] = None, pi: Optional[Any] = None, rho: Optional[Any] = None,
            beta: Optional[Any] = None, gamma: Optional[Any] = None, sigma_bounds: Optional[Tuple[Any, Any]] = None,
            pi_bounds: Optional[Tuple[Any, Any]] = None, rho_bounds: Optional[Tuple[Any, Any]] = None,
            beta_bounds: Optional[Tuple[Any, Any]] = None, gamma_bounds: Optional[Tuple[Any, Any]] = None,
            bounded: bool = False, allow_linear_nans: bool = False) -> None:
        """Coerce parameters into usable formats before storing information about fixed (equal bounds) and unfixed
        (unequal bounds) elements of sigma, pi, rho, beta, and gamma. For unfixed parameters, verify that values have
        been chosen such that choice probability computation is unlikely to overflow. If bounds are unspecified,
        determine reasonable bounds as well. If allow_linear_nans is True, allow null linear parameters in order to
        denote those parameters that will be concentrated out.
        """

        # store labels
        self.sigma_labels = [str(f) for f in economy._X2_formulations]
        self.pi_labels = [str(f) for f in economy._demographics_formulations]
        self.rho_labels = [str(i) for i in economy.unique_nesting_ids]
        self.beta_labels = [str(f) for f in economy._X1_formulations]
        self.gamma_labels = [str(f) for f in economy._X3_formulations]

        # validate and store parameters
        self.sigma = self.initialize_matrix("sigma", "X2 was formulated", sigma, [(economy.K2, economy.K2)])
        self.pi = self.initialize_matrix("pi", "demographics were formulated", pi, [(economy.K2, economy.D)])
        self.rho = self.initialize_matrix("rho", "nesting IDs were specified", rho, [(economy.H, 1), (1, 1)])
        self.beta = self.initialize_matrix("beta", "X1 was formulated", beta, [(economy.K1, 1)], allow_linear_nans)
        self.gamma = self.initialize_matrix("gamma", "X3 was formulated", gamma, [(economy.K3, 1)], allow_linear_nans)

        # identify the index of alpha in beta
        self.alpha_index = np.zeros_like(self.beta, np.bool)
        for k, formulation in enumerate(economy._X1_formulations):
            if 'prices' in formulation.names:
                self.alpha_index[k] = True

        # identify eliminated indexes
        self.eliminated_alpha_index = np.isnan(self.beta) & self.alpha_index
        self.eliminated_beta_index = np.isnan(self.beta)
        self.eliminated_gamma_index = np.isnan(self.gamma)

        # fill the lower triangle of sigma with zeros
        self.sigma[np.tril_indices(economy.K2, -1)] = 0

        # alpha cannot be concentrated out if there's a supply side
        if economy.K3 > 0:
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
        self.store_parameters(SigmaParameter, zip(*np.triu_indices_from(self.sigma)), self.sigma_bounds)
        self.store_parameters(PiParameter, np.ndindex(self.pi.shape), self.pi_bounds)
        self.store_parameters(rho_type, np.ndindex(self.rho.shape), self.rho_bounds)
        self.store_parameters(BetaParameter, zip(*np.where(~self.eliminated_beta_index)), self.beta_bounds)
        self.store_parameters(GammaParameter, zip(*np.where(~self.eliminated_gamma_index)), self.gamma_bounds)

        # count the number of unfixed parameters
        self.P = len(self.unfixed)

        # skip overflow checks and bound computation if parameters aren't bounded
        if not bounded:
            return

        # identify which parameters need default bounds
        unbounded_parameters = []
        bounds_mapping = {
            SigmaParameter: sigma_bounds,
            PiParameter: pi_bounds,
            RhoParameter: rho_bounds
        }
        for parameter in self.unfixed:
            for parameter_type, bounds in bounds_mapping.items():
                if isinstance(parameter, parameter_type):
                    if bounds is None:
                        unbounded_parameters.append(parameter)
                    break

        # compute default bounds for unbounded parameters such that conditional on reasonable values for all other
        #   parameters, choice probability computation is unlikely to require overflow safety precautions
        mu_norm = self.compute_mu_norm(economy)
        mu_max = np.log(np.finfo(np.float64).max)
        for parameter in unbounded_parameters:
            location = parameter.location
            if isinstance(parameter, NonlinearCoefficient):
                v_norm = np.abs(parameter.get_agent_characteristic(economy)).max()
                x_norm = np.abs(parameter.get_product_characteristic(economy)).max()
                additional_mu_norm = self.compute_mu_norm(economy, eliminate_parameter=parameter)
                with np.errstate(divide='ignore'):
                    bound = self.normalize_default_bound(max(0, mu_max - additional_mu_norm) / v_norm / x_norm)
                if isinstance(parameter, SigmaParameter):
                    lb = min(self.sigma[location], -bound if location[0] != location[1] else 0)
                    ub = max(self.sigma[location], +bound)
                    self.sigma_bounds[0][location], self.sigma_bounds[1][location] = lb, ub
                else:
                    assert isinstance(parameter, PiParameter)
                    lb = min(self.pi[location], -bound)
                    ub = max(self.pi[location], +bound)
                    self.pi_bounds[0][location], self.pi_bounds[1][location] = lb, ub
            else:
                assert isinstance(parameter, RhoParameter)
                lb = min(self.rho[location], 0)
                ub = max(self.rho[location], self.normalize_default_bound(1 - min(1, mu_norm / mu_max)))
                self.rho_bounds[0][location], self.rho_bounds[1][location] = lb, ub

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
        if matrix.size > 0 and bound_values is not None and bounded:
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

        # fix parameters set to zero
        zeros = np.where(matrix == 0)
        bounds[0][zeros] = bounds[1][zeros] = 0
        return bounds

    def store_parameters(
            self, parameter_type: Type[Parameter], locations: Iterable[Tuple[int, int]], bounds: Bounds) -> None:
        """Store fixed and unfixed parameters in lists."""
        for location in locations:
            parameter = parameter_type(location, bounds)
            if parameter.value is None:
                self.unfixed.append(parameter)
            else:
                self.fixed.append(parameter)

    def compute_mu_norm(
            self, economy: 'Economy', eliminate_parameter: Optional[NonlinearCoefficient] = None) -> float:
        """Compute the infinity norm of mu under initial parameters, optionally eliminating the contribution of a
        parameter.
        """

        # zero out any parameter that is to be eliminated
        sigma = self.sigma.copy()
        pi = self.pi.copy()
        if isinstance(eliminate_parameter, SigmaParameter):
            sigma[eliminate_parameter.location] = 0
        elif isinstance(eliminate_parameter, PiParameter):
            pi[eliminate_parameter.location] = 0

        # compute the norm by computing mu in each market
        norm = 0
        if economy.K2 > 0:
            for t in economy.unique_market_ids:
                coefficients_t = sigma @ economy.agents.nodes[economy._agent_market_indices[t]].T
                if economy.D > 0:
                    coefficients_t += pi @ economy.agents.demographics[economy._agent_market_indices[t]].T
                mu_t = economy.products.X2[economy._product_market_indices[t]] @ coefficients_t
                norm = max(norm, np.abs(mu_t).max())
        return norm

    @staticmethod
    def normalize_default_bound(bound: float) -> float:
        """Reduce an initial parameter bound by 1% and round it to two significant figures."""
        if not np.isfinite(bound) or bound == 0:
            return bound
        reduced = 0.99 * bound
        return np.round(reduced, 1 + int(reduced < 1) - int(np.log10(reduced)))

    def format(self, title: str) -> str:
        """Format fixed and unfixed parameter values as a string."""
        return self.format_theta_parameters(title, self.sigma, self.pi, self.rho, self.beta, self.gamma)

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
            gamma_like: Array) -> str:
        """Format fixed and unfixed parameter-like values as a string. Skip sections of parameters without any that
        are in theta.
        """
        items = [
            (NonlinearCoefficient, lambda: self.format_nonlinear_coefficients(title, sigma_like, pi_like)),
            (RhoParameter, lambda: self.format_rho(title, rho_like)),
            (BetaParameter, lambda: self.format_beta(title, beta_like)),
            (GammaParameter, lambda: self.format_gamma(title, gamma_like))
        ]
        return "\n\n".join(f() for t, f in items if any(isinstance(p, t) for p in self.fixed + self.unfixed))

    def format_estimates(
            self, title: str, sigma: Array, pi: Array, rho: Array, beta: Array, gamma: Array, sigma_se: Array,
            pi_se: Array, rho_se: Array, beta_se: Array, gamma_se: Array) -> str:
        """Format all estimates and their standard errors as a string."""
        items = [
            (sigma, lambda: self.format_nonlinear_coefficients(title, sigma, pi, sigma_se, pi_se)),
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
        lines = [f"{title}:"]

        # build the formatter
        widths = [max(len(k), options.digits + 8) for k in header]
        formatter = TableFormatter(widths)

        # build the table
        lines.extend([formatter.line(), formatter(header, underline=True)])

        # determine which indices correspond to fixed parameters
        fixed_indices = {p.location[0] for p in self.fixed if isinstance(p, parameter_type)}

        # build the content of the table
        lines.append(formatter([format_number(x) for x in vector]))
        if vector_se is not None:
            lines.append(formatter([format_se(x) if i not in fixed_indices else "" for i, x in enumerate(vector_se)]))

        # build the bottom of the table before combining the lines into one string
        lines.append(formatter.line())
        return "\n".join(lines)

    def format_nonlinear_coefficients(
            self, title: str, sigma_like: Array, pi_like: Array, sigma_se_like: Optional[Array] = None,
            pi_se_like: Optional[Array] = None) -> str:
        """Format matrices (and optional standard errors) of the same size as sigma and pi as a string."""
        lines = [f"Nonlinear Coefficient {title}:"]

        # build the formatter
        line_indices: Set[int] = set()
        header = ["Sigma:"] + self.sigma_labels
        widths = [max(map(len, header))] + [max(len(k), options.digits + 8) for k in header[1:]]
        if self.pi_labels:
            line_indices.add(len(widths) - 1)
            header.extend(["Pi:"] + self.pi_labels)
            widths.extend([widths[0]] + [max(len(k), options.digits + 8) for k in header[len(widths) + 1:]])
        formatter = TableFormatter(widths, line_indices)

        # build the top of the table
        lines.extend([formatter.line(), formatter(header, underline=True)])

        # construct the rows containing parameter information
        for row_index, row_label in enumerate(self.sigma_labels):
            # the row is a label, blanks for sigma's lower triangle, sigma values, the label again, and pi values
            values_row = [row_label] + [""] * row_index
            for column_index in range(row_index, sigma_like.shape[1]):
                values_row.append(format_number(sigma_like[row_index, column_index]))
            if pi_like.shape[1] > 0:
                values_row.append(row_label)
                for column_index in range(pi_like.shape[1]):
                    values_row.append(format_number(pi_like[row_index, column_index]))
            lines.append(formatter(values_row))

            # construct a row of standard errors for unfixed parameters
            if sigma_se_like is not None and pi_se_like is not None:
                # determine which columns in this row correspond to unfixed parameters
                relevant_unfixed = {p for p in self.unfixed if p.location[0] == row_index}
                unfixed_sigma_indices = {p.location[1] for p in relevant_unfixed if isinstance(p, SigmaParameter)}
                unfixed_pi_indices = {p.location[1] for p in relevant_unfixed if isinstance(p, PiParameter)}

                # construct a row similar to the values row without row labels and optionally with standard errors
                se_row = [""] * (1 + row_index)
                for column_index in range(row_index, sigma_se_like.shape[1]):
                    se = sigma_se_like[row_index, column_index]
                    se_row.append(format_se(se) if column_index in unfixed_sigma_indices else "")
                if pi_se_like.shape[1] > 0:
                    se_row.append("")
                    for column_index in range(pi_se_like.shape[1]):
                        se = pi_se_like[row_index, column_index]
                        se_row.append(format_se(se) if column_index in unfixed_pi_indices else "")

                # format the row of values and add an additional blank line if there is another row of values
                lines.append(formatter(se_row))
                if row_index < sigma_like.shape[1] - 1:
                    lines.append(formatter.blank())

        # build the bottom of the table before combining the lines into one string
        lines.append(formatter.line())
        return "\n".join(lines)

    def compress(self) -> Array:
        """Compress the initial parameters into theta."""
        items = [
            (SigmaParameter, self.sigma),
            (PiParameter, self.pi),
            (RhoParameter, self.rho),
            (BetaParameter, self.beta),
            (GammaParameter, self.gamma)
        ]
        return np.r_[[v[p.location] for t, v in items for p in self.unfixed if isinstance(p, t)]]

    def compress_bounds(self) -> List[Tuple[float, float]]:
        """Compress parameter bounds into a list of (lb, ub) tuples for theta."""
        items = [
            (SigmaParameter, self.sigma_bounds),
            (PiParameter, self.pi_bounds),
            (RhoParameter, self.rho_bounds),
            (BetaParameter, self.beta_bounds),
            (GammaParameter, self.gamma_bounds)
        ]
        return [(l[p.location], u[p.location]) for t, (l, u) in items for p in self.unfixed if isinstance(p, t)]

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
            (GammaParameter, gamma_like)
        ]
        for parameter, value in zip(self.unfixed, theta_like):
            for parameter_type, values in items:
                if isinstance(parameter, parameter_type):
                    values[parameter.location] = value
                    break
        if not nullify:
            sigma_like[np.tril_indices_from(sigma_like, -1)] = 0
            for parameter in self.fixed:
                for parameter_type, values in items:
                    if isinstance(parameter, parameter_type):
                        values[parameter.location] = parameter.value
                        break
        return sigma_like, pi_like, rho_like, beta_like, gamma_like
