"""Micro moments."""

import abc
import collections
from typing import Dict, Hashable, Optional, Sequence, TYPE_CHECKING

import numpy as np

from . import options
from .utilities.basics import Array, StringRepresentation, TableFormatter, format_number


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .economies.economy import Economy  # noqa


class Moment(StringRepresentation):
    """Information about a single micro moment."""

    value: Array
    market_ids: Optional[Array]

    def __init__(self, value: float, market_ids: Optional[Sequence] = None) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        self.value = np.asarray(value, options.dtype)
        if self.value.size != 1:
            raise ValueError("The micro moment value must be a scalar.")
        self.market_ids = None
        if market_ids is not None:
            self.market_ids = np.asarray(market_ids, np.object)
            unique, counts = np.unique(self.market_ids, return_counts=True)
            duplicates = unique[counts > 1]
            if duplicates.size > 0:
                raise ValueError(f"The following market IDs are duplicated in market_ids: {duplicates}.")

    def __str__(self) -> str:
        """Format information about the micro moment as a string."""
        return f"{self._format_markets()}: {self._format_moment()}"

    def _format_markets(self) -> str:
        """Format information about the markets associated with the micro moment as a string."""
        if self.market_ids is None:
            return "Average Over All Markets"
        markets_title = "Average Over Markets" if self.market_ids.size > 1 else "Market"
        markets_list = ", ".join(f"'{t}'" for t in self.market_ids)
        return f"{markets_title} {markets_list}"

    def _format_moment(self) -> str:
        """Construct a string expression for the micro moment."""
        formatted = self._format_value()
        if self.value < 0:
            formatted = f"{formatted} + {format_number(float(self.value), sign=False)}"
        elif self.value > 0:
            formatted = f"{formatted} - {format_number(float(self.value), sign=False)}"
        return formatted

    @abc.abstractmethod
    def _format_value(self) -> str:
        """Construct a string expression for the micro moment value."""

    def _validate(self, economy: 'Economy') -> None:
        """Check that all market IDs associated with this moment are in the economy."""
        if self.market_ids is not None:
            extra_ids = set(self.market_ids) - set(economy.unique_market_ids)
            if extra_ids:
                raise ValueError(f"market_ids contains the following extra IDs: {sorted(extra_ids)}.")


class ProductsAgentsCovarianceMoment(Moment):
    r"""Configuration for micro moments that match covariances between product and agent characteristics, conditional on
    purchasing non-outside goods.

    For example, survey data can often be used to compute the covariance :math:`\sigma_{xy}` between a product
    characteristic, :math:`x_{jt}`, and an agent demographic such as income, :math:`y_{it}`, conditional on purchasing a
    non-outside good. With this covariance, a micro moment :math:`m` can approximated in market :math:`t` by

    .. math:: g_{M,mt} \approx \sum_{i=1}^{I_t} w_i(z_{it} - \bar{z}_t)(y_{it} - \bar{y}_t) - \sigma_{xy}

    where, conditional on purchasing a non-outside good, the expected value of :math:`x_{jt}` for agent :math:`i` is

    .. math:: z_{it} = \sum_{j=1}^{J_t} \frac{x_{jt}s_{jti}}{1 - s_{0ti}}.

    These market-level micro moments are averaged across a set :math:`\mathscr{T}_m` of markets in which the micro data
    used to compute :math:`\sigma_{xy}` is relevant. This gives a :math:`\bar{g}_{M,m}` in :eq:`averaged_micro_moments`.

    Parameters
    ----------
    X2_index : `int`
        Column index of :math:`x_{jt}` in the matrix of nonlinear product characteristics, :math:`X_2`. This should be
        between zero and :math:`K_2 - 1`, inclusive.
    demographics_index : `int`
        Column index of the demographic :math:`y_{it}` (which can be any demographic, not just income) in the matrix of
        agent demographics, :math:`d`. This should be between zero and :math:`D - 1`, inclusive.
    value : `float`
        Value of the covariance :math:`\sigma_{xy}` estimated from micro data.
    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments :math:`g_{M,mt}` will be averaged to get :math:`\bar{g}_{M,m}`.
        These are also the only markets in which the moments will be computed. By default, the moments are computed for
        and averaged across all markets. That is, by default, it is assumed that the specified ``value`` is on average
        the same for all markets.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    X2_index: int
    demographics_index: int

    def __init__(
            self, X2_index: int, demographics_index: int, value: float, market_ids: Optional[Sequence] = None) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        super().__init__(value, market_ids)
        if not isinstance(X2_index, int) or X2_index < 0:
            raise ValueError("X2_index must be a positive int.")
        if not isinstance(demographics_index, int) or demographics_index < 0:
            raise ValueError("demographics_index must be a positive int.")
        self.X2_index = X2_index
        self.demographics_index = demographics_index

    def _format_value(self) -> str:
        """Construct a string expression for the covariance moment."""
        return f"Cov(X2 Column {self.X2_index}, Demographic Column {self.demographics_index})"

    def _validate(self, economy: 'Economy') -> None:
        """Check that matrix indices are valid in the economy."""
        super()._validate(economy)
        if self.X2_index >= economy.K2:
            raise ValueError(f"X2_index must be between 0 and K2 = {economy.K2}, inclusive.")
        if self.demographics_index >= economy.D:
            raise ValueError(f"demographics_index must be between 0 and D = {economy.D}, inclusive.")


class Moments(object):
    """Information about a sequence of micro moments."""

    micro_moments: Sequence[Moment]
    MM: int

    def __init__(self, micro_moments: Sequence[Moment]) -> None:
        """Store information about a sequence of micro moment instances."""
        self.micro_moments = micro_moments
        self.MM = len(micro_moments)

    def format(self, title: str, values: Optional[Array] = None) -> str:
        """Format micro moments (and optionally their values) as a string."""
        lines = [f"{title}:"]

        # build the formatter
        mapping = collections.OrderedDict([
            ("Index", [str(i) for i in range(self.MM)]),
            ("Markets", [m._format_markets() for m in self.micro_moments]),
            ("Moment", [m._format_moment() for m in self.micro_moments])
        ])
        if values is not None:
            mapping["Value"] = [format_number(v) for v in values]
        widths = [max(len(k), max(map(len, v))) for k, v in mapping.items()]
        formatter = TableFormatter(widths)

        # build the top of the table
        lines.extend([formatter.line(), formatter(list(mapping.keys()), underline=True)])

        # build the content of the table
        for index in range(self.MM):
            lines.append(formatter([v[index] for v in mapping.values()]))

        # build the bottom of the table before combining the lines into one string
        lines.append(formatter.line())
        return "\n".join(lines)


class EconomyMoments(Moments):
    """Information about a sequence of micro moments in an economy."""

    market_counts: Array
    market_indices: Dict[Hashable, Array]

    def __init__(self, economy: 'Economy', micro_moments: Sequence[Moment]) -> None:
        """Validate and store information about a sequence of micro moment instances in the context of an economy."""

        # validate the moments
        if not isinstance(micro_moments, collections.Sequence):
            raise TypeError("micro_moments must be a tuple of micro moment instances.")
        for moment in micro_moments:
            if not isinstance(moment, Moment):
                raise TypeError("micro_moments must consist only of micro moment instances.")
            try:
                moment._validate(economy)
            except Exception as exception:
                raise ValueError(f"The micro moment '{moment}' is invalid.") from exception

        # store basic moment information
        super().__init__(micro_moments)

        # identify market counts and indices
        market_counts = [economy.T if m.market_ids is None else m.market_ids.size for m in micro_moments]
        self.market_counts = np.c_[np.array(market_counts)]
        self.market_indices: Dict[Hashable, Array] = {}
        for t in economy.unique_market_ids:
            market_index_t = [m.market_ids is None or t in m.market_ids for m in self.micro_moments]
            self.market_indices[t] = np.flatnonzero(market_index_t)


class MarketMoments(Moments):
    """Information about a sequence of micro moments in a market."""

    def __init__(self, economy_moments: EconomyMoments, t: Hashable) -> None:
        """Select only those micro moment instances that will be computed for this market."""
        super().__init__([m for m in economy_moments.micro_moments if m.market_ids is None or t in m.market_ids])
