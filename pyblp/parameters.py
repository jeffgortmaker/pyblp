"""Parameters underlying the BLP model."""

import abc
import itertools
from typing import Any, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple

import numpy as np

from . import exceptions, options
from .utilities.basics import Array, Bounds, Error, Groups, RecArray, TableFormatter, format_number, format_se


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .economies.economy import Economy  # noqa


class NonlinearParameter(abc.ABC):
    """Information about a single nonlinear parameter."""

    unbounded: bool
    location: Sequence
    value: Optional[float]

    def __init__(self, location: Sequence, bounds: Bounds, unbounded: bool) -> None:
        """Store the information and determine whether the parameter is fixed or unfixed."""
        self.location = location
        self.unbounded = unbounded
        self.value = bounds[0][location] if bounds[0][location] == bounds[1][location] else None


class RandomCoefficientParameter(NonlinearParameter):
    """Information about a single nonlinear parameter in sigma or pi."""

    def get_product_characteristic(self, products: RecArray) -> Array:
        """Get the product characteristic associated with the parameter."""
        return products.X2[:, [self.location[0]]]

    @abc.abstractmethod
    def get_agent_characteristic(self, agents: RecArray) -> Array:
        """Get the agent characteristic associated with the parameter."""
        pass


class SigmaParameter(RandomCoefficientParameter):
    """Information about a single parameter in sigma."""

    def get_agent_characteristic(self, agents: RecArray) -> Array:
        """Get the agent characteristic associated with the parameter."""
        return agents.nodes[:, [self.location[1]]]


class PiParameter(RandomCoefficientParameter):
    """Information about a single parameter in pi."""

    def get_agent_characteristic(self, agents: RecArray) -> Array:
        """Get the agent characteristic associated with the parameter."""
        return agents.demographics[:, [self.location[1]]]


class RhoParameter(NonlinearParameter):
    """Information about a single parameter in rho."""

    single: bool

    def __init__(self, location: Tuple[int, int], bounds: Bounds, unbounded: bool, single: bool) -> None:
        """Store the information along with whether there is only a single parameter for all groups."""
        super().__init__(location, bounds, unbounded)
        self.single = single

    def get_group_associations(self, groups: Groups) -> Array:
        """Get an indicator for which groups are associated with the parameter."""
        associations = np.ones((groups.unique.size, 1), options.dtype)
        if not self.single:
            associations[:] = 0
            associations[self.location] = 1
        return associations


class NonlinearParameters(object):
    """Information about sigma and pi."""

    sigma_labels: List[str]
    pi_labels: List[str]
    rho_labels: List[str]
    sigma: Array
    pi: Array
    rho: Array
    sigma_bounds: Bounds
    pi_bounds: Bounds
    rho_bounds: Bounds
    fixed: List[NonlinearParameter]
    unfixed: List[NonlinearParameter]
    P: int
    errors: List[Error]

    def __init__(
            self, economy: 'Economy', sigma: Optional[Any] = None, pi: Optional[Any] = None, rho: Optional[Any] = None,
            sigma_bounds: Optional[Tuple[Any, Any]] = None, pi_bounds: Optional[Tuple[Any, Any]] = None,
            rho_bounds: Optional[Tuple[Any, Any]] = None, bounded: bool = False) -> None:
        """Store information about fixed (equal bounds) and unfixed (unequal bounds) elements of sigma, pi, and rho.
        Also verify that parameters have been chosen such that choice probability computation is unlikely to overflow.
        If unspecified, determine reasonable bounds as well.
        """

        # store labels
        self.sigma_labels = [str(f) for f in economy._X2_formulations]
        self.pi_labels = [str(f) for f in economy._demographics_formulations]
        self.rho_labels = [str(i) for i in economy.unique_nesting_ids]

        # store the upper triangle of sigma
        self.sigma = np.full((economy.K2, economy.K2), np.nan, options.dtype)
        if sigma is None and economy.K2 > 0:
            raise ValueError("X2 was formulated, so sigma should not be None.")
        elif sigma is not None:
            self.sigma = np.c_[np.asarray(sigma, options.dtype)]
            if economy.K2 == 0 and sigma.size > 0:
                raise ValueError("X2 was not formulated, so sigma should be None.")
            if self.sigma.shape != (economy.K2, economy.K2):
                raise ValueError(f"sigma must be a {economy.K2} by {economy.K2} matrix.")
            self.sigma[np.tril_indices(economy.K2, -1)] = 0

        # store pi
        self.pi = np.full((economy.K2, 0), np.nan, options.dtype)
        if pi is None and economy.D > 0:
            raise ValueError("Demographics were formulated, so pi should not be None.")
        elif pi is not None:
            self.pi = np.c_[np.asarray(pi, options.dtype)]
            if economy.D == 0 and self.pi.size > 0:
                raise ValueError("Demographics were not formulated, so pi should be None.")
            if self.pi.shape != (economy.K2, economy.D):
                raise ValueError(f"pi must be a {economy.K2} by {economy.D} matrix.")

        # store rho
        self.rho = np.full((0, 1), np.nan, options.dtype)
        if rho is None and economy.H > 0:
            raise ValueError("Nesting IDs were specified, so rho should not be None.")
        elif rho is not None:
            self.rho = np.c_[np.asarray(rho, options.dtype)]
            if economy.H == 0 and self.rho.size > 0:
                raise ValueError("Nesting IDs were not specified, so rho should be None.")
            if self.rho.shape not in {(1, 1), (economy.H, 1)}:
                raise ValueError(f"rho must be a scalar or a {economy.H}-vector.")

        # validate sigma bounds
        self.sigma_bounds = (
            np.full_like(self.sigma, -np.inf, options.dtype), np.full_like(self.sigma, +np.inf, options.dtype)
        )
        if economy.K2 > 0 and sigma_bounds is not None and bounded:
            if len(sigma_bounds) != 2:
                raise ValueError("sigma_bounds must be a tuple of the form (lb, ub).")
            self.sigma_bounds = (
                np.c_[np.asarray(sigma_bounds[0], options.dtype)], np.c_[np.asarray(sigma_bounds[1], options.dtype)]
            )
            self.sigma_bounds[0][np.isnan(self.sigma_bounds[0])] = -np.inf
            self.sigma_bounds[1][np.isnan(self.sigma_bounds[1])] = +np.inf
            if self.sigma_bounds[0].shape != self.sigma.shape:
                raise ValueError(f"The lower bound in sigma_bounds does not have the same shape as sigma.")
            if self.sigma_bounds[1].shape != self.sigma.shape:
                raise ValueError(f"The upper bound in sigma_bounds does not have the same shape as sigma.")
            if ((self.sigma < self.sigma_bounds[0]) | (self.sigma > self.sigma_bounds[1])).any():
                raise ValueError("sigma must be within its bounds.")

        # validate pi bounds
        self.pi_bounds = (
            np.full_like(self.pi, -np.inf, options.dtype), np.full_like(self.pi, +np.inf, options.dtype)
        )
        if economy.D > 0 and pi_bounds is not None and bounded:
            if len(pi_bounds) != 2:
                raise ValueError("pi_bounds must be a tuple of the form (lb, ub).")
            self.pi_bounds = (
                np.c_[np.asarray(pi_bounds[0], options.dtype)], np.c_[np.asarray(pi_bounds[1], options.dtype)]
            )
            self.pi_bounds[0][np.isnan(self.pi_bounds[0])] = -np.inf
            self.pi_bounds[1][np.isnan(self.pi_bounds[1])] = +np.inf
            if self.pi_bounds[0].shape != self.pi.shape:
                raise ValueError(f"The lower bound in pi_bounds does not have the same shape as pi.")
            if self.pi_bounds[1].shape != self.pi.shape:
                raise ValueError(f"The upper bound in pi_bounds does not have the same shape as pi.")
            if ((self.pi < self.pi_bounds[0]) | (self.pi > self.pi_bounds[1])).any():
                raise ValueError("pi must be within its bounds.")

        # validate rho bounds
        self.rho_bounds = (
            np.full_like(self.rho, -np.inf, options.dtype), np.full_like(self.rho, +np.inf, options.dtype)
        )
        if economy.H > 0 and rho_bounds is not None and bounded:
            if len(rho_bounds) != 2:
                raise ValueError("rho_bounds must be a tuple of the form (lb, ub).")
            self.rho_bounds = (
                np.c_[np.asarray(rho_bounds[0], options.dtype)], np.c_[np.asarray(rho_bounds[1], options.dtype)]
            )
            self.rho_bounds[0][np.isnan(self.rho_bounds[0])] = -np.inf
            self.rho_bounds[1][np.isnan(self.rho_bounds[1])] = +np.inf
            if self.rho_bounds[0].shape != self.rho.shape:
                raise ValueError(f"The lower bound in rho_bounds does not have the same shape as rho.")
            if self.rho_bounds[1].shape != self.rho.shape:
                raise ValueError(f"The upper bound in rho_bounds does not have the same shape as rho.")
            if ((self.rho < self.rho_bounds[0]) | (self.rho > self.rho_bounds[1])).any():
                raise ValueError("rho must be within its bounds.")

        # set upper and lower bounds to zero for parameters that are fixed at zero
        sigma_zeros = np.where(self.sigma == 0)
        pi_zeros = np.where(self.pi == 0)
        rho_zeros = np.where(self.rho == 0)
        self.sigma_bounds[0][sigma_zeros] = self.pi_bounds[0][pi_zeros] = self.rho_bounds[0][rho_zeros] = 0
        self.sigma_bounds[1][sigma_zeros] = self.pi_bounds[1][pi_zeros] = self.rho_bounds[1][rho_zeros] = 0

        # store information about individual elements in sigma, pi, and rho
        self.fixed: List[NonlinearParameter] = []
        self.unfixed: List[NonlinearParameter] = []

        # store information for the upper triangle of sigma
        for location in zip(*np.triu_indices_from(self.sigma)):
            sigma_parameter = SigmaParameter(location, self.sigma_bounds, unbounded=sigma_bounds is None)
            parameter_list = self.unfixed if sigma_parameter.value is None else self.fixed
            parameter_list.append(sigma_parameter)

        # store information for pi
        for location in np.ndindex(self.pi.shape):
            pi_parameter = PiParameter(location, self.pi_bounds, unbounded=pi_bounds is None)
            parameter_list = self.unfixed if pi_parameter.value is None else self.fixed
            parameter_list.append(pi_parameter)

        # store information for rho
        for location in np.ndindex(self.rho.shape):
            single = self.rho.size == 1
            rho_parameter = RhoParameter(location, self.rho_bounds, unbounded=rho_bounds is None, single=single)
            parameter_list = self.unfixed if rho_parameter.value is None else self.fixed
            parameter_list.append(rho_parameter)

        # count the number of unfixed parameters
        self.P = len(self.unfixed)

        # verify that parameters have been chosen such that choice probability computation is unlikely to overflow
        self.errors: List[Error] = []
        mu_norm = self.compute_mu_norm(economy)
        mu_max = np.log(np.finfo(np.float64).max)
        if mu_norm > mu_max or (economy.H > 0 and mu_norm > mu_max * (1 - self.rho.max())):
            self.errors.append(exceptions.LargeInitialParametersError())

        # compute default bounds for each parameter such that conditional on reasonable values for all other parameters,
        #   choice probability computation is unlikely to overflow
        for parameter in self.unfixed:
            if parameter.unbounded:
                location = parameter.location
                if isinstance(parameter, RandomCoefficientParameter):
                    v_norm = np.abs(parameter.get_agent_characteristic(economy.agents)).max()
                    x_norm = np.abs(parameter.get_product_characteristic(economy.products)).max()
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

    def compute_mu_norm(self, economy: 'Economy', eliminate_parameter: Optional[NonlinearParameter] = None) -> float:
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
        """Reduce an initial parameter bound by 5% and round it to two significant figures."""
        if not np.isfinite(bound) or bound == 0:
            return bound
        reduced = 0.95 * bound
        return np.round(reduced, 1 + int(reduced < 1) - int(np.log10(reduced)))

    def format(self) -> str:
        """Format the initial sigma, pi, and rho as a string."""
        return self.format_matrices(self.sigma, self.pi, self.rho)

    def format_lower_bounds(self) -> str:
        """Format lower sigma, pi, and rho bounds as a string."""
        return self.format_matrices(self.sigma_bounds[0], self.pi_bounds[0], self.rho_bounds[0])

    def format_upper_bounds(self) -> str:
        """Format upper sigma, pi, and rho bounds as a string."""
        return self.format_matrices(self.sigma_bounds[1], self.pi_bounds[1], self.rho_bounds[1])

    def format_estimates(
            self, sigma: Array, pi: Array, rho: Array, sigma_se: Array, pi_se: Array, rho_se: Array) -> str:
        """Format sigma, pi, and rho estimates along with their standard errors as a string."""
        return self.format_matrices(sigma, pi, rho, sigma_se, pi_se, rho_se)

    def format_matrices(
            self, sigma_like: Array, pi_like: Array, rho_like: Array, sigma_se_like: Optional[Array] = None,
            pi_se_like: Optional[Array] = None, rho_se_like: Optional[Array] = None) -> str:
        """Format matrices (and optional standard errors) of the same size as sigma, pi, and rho as a string."""
        lines: List[str] = []

        # construct the primary table for sigma and pi
        if sigma_like.shape[1] > 0:
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
                    unfixed_sigma_indices = set()
                    unfixed_pi_indices = set()
                    for parameter in self.unfixed:
                        if isinstance(parameter, RandomCoefficientParameter) and parameter.location[0] == row_index:
                            if isinstance(parameter, SigmaParameter):
                                unfixed_sigma_indices.add(parameter.location[1])
                            else:
                                assert isinstance(parameter, PiParameter)
                                unfixed_pi_indices.add(parameter.location[1])

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

            # build the bottom of the table
            lines.append(formatter.line())

        # construct a table for rho
        if rho_like.size > 0:
            rho_header = ["Rho:"] + (self.rho_labels if rho_like.size > 1 else ["All Groups"])
            rho_widths = [len(rho_header[0])] + [max(len(k), options.digits + 8) for k in rho_header[1:]]
            rho_formatter = TableFormatter(rho_widths)

            # build the top of the table, replacing the top and bottom of the table for sigma and pi if necessary
            if not lines:
                lines.append(rho_formatter.line())
            elif len(lines[-1]) < len(rho_formatter.line()):
                lines[0] = lines[-1] = rho_formatter.line()
            lines.append(rho_formatter(rho_header, underline=True))

            # determine which elements correspond to unfixed parameters
            unfixed_rho_indices = set()
            for parameter in self.unfixed:
                if isinstance(parameter, RhoParameter):
                    unfixed_rho_indices.add(parameter.location[0])

            # build the content of the table
            lines.append(rho_formatter([""] + [format_number(x) for x in rho_like]))
            if rho_se_like is not None:
                se_row = [""]
                for row_index in range(rho_se_like.shape[0]):
                    se = rho_se_like[row_index]
                    se_row.append(format_se(se) if row_index in unfixed_rho_indices else "")
                lines.append(rho_formatter(se_row))

            # build the bottom of the table
            lines.append(lines[0])

        # combine the lines into one string
        return "\n".join(lines)

    def compress(self) -> Array:
        """Compress the initial sigma, pi, and rho into theta."""
        theta_values: List[float] = []
        tuples = [(self.sigma, SigmaParameter), (self.pi, PiParameter), (self.rho, RhoParameter)]
        for values, parameter_type in tuples:
            theta_values.extend(values[p.location] for p in self.unfixed if isinstance(p, parameter_type))
        return np.r_[theta_values]

    def compress_bounds(self) -> List[Tuple[float, float]]:
        """Compress sigma, pi, and rho bounds into a list of (lb, ub) tuples for theta."""
        theta_bounds: List[Tuple[float, float]] = []
        tuples = [(self.sigma_bounds, SigmaParameter), (self.pi_bounds, PiParameter), (self.rho_bounds, RhoParameter)]
        for (lb, ub), parameter_type in tuples:
            theta_bounds.extend((lb[p.location], ub[p.location]) for p in self.unfixed if isinstance(p, parameter_type))
        return theta_bounds

    def expand(self, theta_like: Array, nullify: bool = False) -> Tuple[Array, Array, Array]:
        """Recover matrices of the same size as sigma, pi, and rho from a vector of the same size as theta. By default,
        fill elements corresponding to fixed parameters with their fixed values.
        """
        sigma_like = np.full_like(self.sigma, np.nan)
        pi_like = np.full_like(self.pi, np.nan)
        rho_like = np.full_like(self.rho, np.nan)

        # set values for elements that correspond to unfixed parameters
        for parameter, value in zip(self.unfixed, theta_like):
            if isinstance(parameter, SigmaParameter):
                sigma_like[parameter.location] = value
            elif isinstance(parameter, PiParameter):
                pi_like[parameter.location] = value
            else:
                assert isinstance(parameter, RhoParameter)
                rho_like[parameter.location] = value

        # set values for elements that correspond to fixed parameters
        if not nullify:
            sigma_like[np.tril_indices_from(sigma_like, -1)] = 0
            for parameter in self.fixed:
                if isinstance(parameter, SigmaParameter):
                    sigma_like[parameter.location] = parameter.value
                elif isinstance(parameter, PiParameter):
                    pi_like[parameter.location] = parameter.value
                else:
                    assert isinstance(parameter, RhoParameter)
                    rho_like[parameter.location] = parameter.value

        # return the expanded matrices
        return sigma_like, pi_like, rho_like


class LinearParameters(object):
    """Information about beta and gamma."""

    beta_labels: List[str]
    gamma_labels: List[str]
    beta: Array
    gamma: Array

    def __init__(self, economy: 'Economy', beta: Any, gamma: Optional[Any] = None) -> None:
        """Store information about parameters in beta and gamma."""

        # store labels
        self.beta_labels = [str(f) for f in economy._X1_formulations]
        self.gamma_labels = [str(f) for f in economy._X3_formulations]

        # store beta
        self.beta = np.c_[np.asarray(beta, options.dtype)]
        if self.beta.shape != (economy.K1, 1):
            raise ValueError(f"beta must be a {economy.K1}-vector.")

        # store gamma
        self.gamma = np.full((economy.K3, 0), np.nan, options.dtype)
        if gamma is not None:
            self.gamma = np.c_[np.asarray(gamma, options.dtype)]
            if economy.K3 == 0 and self.gamma.size > 0:
                raise ValueError("X2 was not formulated, so gamma should be None.")
            if self.gamma.shape != (economy.K3, 1):
                raise ValueError(f"gamma must be a {economy.K3}-vector.")

    def format(self) -> str:
        """Format the initial beta and gamma as a string."""
        return self.format_vectors(self.beta, self.gamma)

    def format_estimates(self, beta: Array, gamma: Array, beta_se: Array, gamma_se: Array) -> str:
        """Format beta and gamma estimates along with their standard errors as a string."""
        return self.format_vectors(beta, gamma, beta_se, gamma_se)

    def format_vectors(
            self, beta_like: Array, gamma_like: Array, beta_se_like: Optional[Array] = None,
            gamma_se_like: Optional[Array] = None) -> str:
        """Format matrices (and optional standard errors) of the same size as beta and gamma as a string."""
        lines: List[str] = []

        # build the header for beta
        beta_header = ["Beta:"] + self.beta_labels
        beta_widths = [len(beta_header[0])] + [max(len(k), options.digits + 8) for k in beta_header[1:]]

        # build the header for gamma
        gamma_header = None
        gamma_widths: List[int] = []
        if self.gamma_labels:
            gamma_header = ["Gamma:"] + self.gamma_labels
            gamma_widths = [len(gamma_header[0])] + [max(len(k), options.digits + 8) for k in gamma_header[1:]]

        # build the table formatter
        widths = [max(w) for w in itertools.zip_longest(beta_widths, gamma_widths, fillvalue=0)]
        formatter = TableFormatter(widths)

        # build the table
        lines.extend([
            formatter.line(),
            formatter(beta_header, underline=True),
            formatter([""] + [format_number(x) for x in beta_like])
        ])
        if beta_se_like is not None:
            lines.append(formatter([""] + [format_se(x) for x in beta_se_like]))
        if gamma_header is not None:
            lines.extend([
                formatter.line(),
                formatter(gamma_header, underline=True),
                formatter([""] + [format_number(x) for x in gamma_like])
            ])
            if gamma_se_like is not None:
                lines.append(formatter([""] + [format_se(x) for x in gamma_se_like]))
        lines.append(formatter.line())

        # combine the lines into one string
        return "\n".join(lines)
