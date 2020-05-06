"""Micro moments."""

import abc
import collections
from typing import Any, Dict, Hashable, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from . import options
from .utilities.basics import Array, StringRepresentation, format_number, format_table


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
            return "All"
        return ", ".join(str(t) for t in self.market_ids)

    def _format_moment(self) -> str:
        """Construct a string expression for the micro moment."""
        formatted = self._format_value()
        if self.value < 0:
            formatted = f"{formatted} + {format_number(float(self.value))[1:]}"
        elif self.value > 0:
            formatted = f"{formatted} - {format_number(float(self.value))[1:]}"
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


class DemographicExpectationMoment(Moment):
    r"""Configuration for micro moments that match expectations of demographics for agents who choose certain products.

    For example, micro data can sometimes be used to compute the mean of a demographic such as income, :math:`y_{it}`,
    for agents who choose product :math:`j`. With the value :math:`v_m` of this mean, a micro moment :math:`m` in market
    :math:`t` for agent :math:`i` can be defined by

    .. math:: g_{M,imt} = \frac{y_{it} s_{ijt}}{s_{jt}} - v_m.

    Integrals of these micro moments are approximated within and averaged across a set :math:`T_m` of markets in which
    the micro data used to compute :math:`v_m` are relevant, which gives :math:`\bar{g}_{M,m}` in
    :eq:`averaged_micro_moments`.

    Parameters
    ----------
    product_id : `object`
        ID of the product :math:`j` or ``None`` to denote the outside option :math:`j = 0`. If not ``None``, there must
        be exactly one of this ID in the ``product_ids`` field of ``product_data`` in :class:`Problem` or
        :class:`Simulation` for each market over which this micro moment will be averaged.
    demographics_index : `int`
        Column index of the demographic :math:`y_{it}` (which can be any demographic, not just income) in the matrix of
        agent demographics, :math:`d`. This should be between zero and :math:`D - 1`, inclusive.
    value : `float`
        Value :math:`v_m` of the mean estimated from micro data.
    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments will be averaged to get :math:`\bar{g}_{M,m}`. These are also
        the only markets in which the moments will be computed. By default, the moments are computed for and averaged
        across all markets. That is, by default, it is assumed that the specified ``value`` is relevant for and on
        average the same for all markets.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    product_id: Optional[Any]
    demographics_index: int

    def __init__(
            self, product_id: Optional[Any], demographics_index: int, value: float,
            market_ids: Optional[Sequence] = None) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        super().__init__(value, market_ids)
        if not isinstance(demographics_index, int) or demographics_index < 0:
            raise ValueError("demographics_index must be a positive int.")
        self.product_id = product_id
        self.demographics_index = demographics_index

    def _format_value(self) -> str:
        """Construct a string expression for the covariance moment."""
        product = "Outside" if self.product_id is None else f"'{self.product_id}'"
        return f"E[Demographic Column {self.demographics_index} | {product}]"

    def _validate(self, economy: 'Economy') -> None:
        """Check that matrix indices are valid in the economy."""
        super()._validate(economy)
        economy._validate_product_id(self.product_id, self.market_ids)
        if self.demographics_index >= economy.D:
            raise ValueError(f"demographics_index must be between 0 and D = {economy.D}, inclusive.")


class DemographicCovarianceMoment(Moment):
    r"""Configuration for micro moments that match covariances between product characteristics and demographics.

    For example, micro data can sometimes be used to compute the sample covariance between a product characteristic
    :math:`x_{jt}` of an agent's choice :math:`j`, and a demographic such as income, :math:`y_{it}`, amongst those
    agents who purchase an inside good. With the value :math:`v_m` of this sample covariance, a micro moment :math:`m`
    in market :math:`t` for agent :math:`i` can be defined by

    .. math:: g_{M,imt} = (z_{it} - \bar{z}_t)(y_{it} - \bar{y}_t) - v_m

    where :math:`\bar{z}_t = \sum_i w_{it} z_{it}`, :math:`\bar{y}_t = \sum_i w_{it} y_{it}`, and conditional on
    choosing an inside good, the expected value of :math:`x_{jt}` for agent :math:`i` is

    .. math:: z_{it} = \sum_{j \in J_t} x_{jt}s_{ij(-0)t}

    where :math:`s_{ij(-0)t}` is the probability of :math:`i` choosing :math:`j` when the outside option is removed from
    the choice set.

    Integrals of these micro moments are approximated within and averaged across a set :math:`T_m` of markets in which
    the micro data used to compute :math:`v_m` are relevant, which gives :math:`\bar{g}_{M,m}` in
    :eq:`averaged_micro_moments`.

    Parameters
    ----------
    X2_index : `int`
        Column index of :math:`x_{jt}` in the matrix of demand-side nonlinear product characteristics, :math:`X_2`. This
        should be between zero and :math:`K_2 - 1`, inclusive.
    demographics_index : `int`
        Column index of the demographic :math:`y_{it}` (which can be any demographic, not just income) in the matrix of
        agent demographics, :math:`d`. This should be between zero and :math:`D - 1`, inclusive.
    value : `float`
        Value :math:`v_m` of the sample covariance estimated from micro data.
    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments will be averaged to get :math:`\bar{g}_{M,m}`. These are also
        the only markets in which the moments will be computed. By default, the moments are computed for and averaged
        across all markets. That is, by default, it is assumed that the specified ``value`` is relevant for and on
        average the same for all markets.

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
        return f"Cov(X2 Column {self.X2_index}, Demographic Column {self.demographics_index} | Inside)"

    def _validate(self, economy: 'Economy') -> None:
        """Check that matrix indices are valid in the economy."""
        super()._validate(economy)
        if self.X2_index >= economy.K2:
            raise ValueError(f"X2_index must be between 0 and K2 = {economy.K2}, inclusive.")
        if self.demographics_index >= economy.D:
            raise ValueError(f"demographics_index must be between 0 and D = {economy.D}, inclusive.")


class DiversionProbabilityMoment(Moment):
    r"""Configuration for micro moments that match second choice probabilities of certain products for agents whose
    first choices are certain other products.

    For example, micro data can sometimes be used to compute the share of agents who would choose product :math:`k` if
    :math:`j` were removed from the choice set, out of those agents whose first choice is :math:`j`. With the value
    :math:`v_m` of this share, a micro moment :math:`m` in market :math:`t` for agent :math:`i` can be defined by

    .. math:: g_{M,imt} = \frac{s_{ik(-j)t} s_{ijt}}{s_{jt}} - v_m

    where :math:`s_{ik(-j)t}` is the probability of :math:`i` choosing :math:`k` when :math:`j` is removed from the
    choice set.

    Integrals of these micro moments are approximated within and averaged across a set :math:`T_m` of markets in which
    the micro data used to compute :math:`v_m` are relevant, which gives :math:`\bar{g}_{M,m}` in
    :eq:`averaged_micro_moments`.

    Parameters
    ----------
    product_id1 : `object`
        ID of the first choice product :math:`j`, which cannot be the outside option. There must be exactly one of this
        ID in the ``product_ids`` field of ``product_data`` in :class:`Problem` or :class:`Simulation` for each market
        over which this micro moment will be averaged.
    product_id2 : `object`
        ID of the second choice product :math:`k` or ``None`` to denote the outside option :math:`j = 0`. If not
        ``None``, there must be exactly one of this ID for each market over which this micro moment will be averaged.
    value : `float`
        Value :math:`v_m` of the share estimated from micro data.
    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments will be averaged to get :math:`\bar{g}_{M,m}`. These are also
        the only markets in which the moments will be computed. By default, the moments are computed for and averaged
        across all markets. That is, by default, it is assumed that the specified ``value`` is relevant for and on
        average the same for all markets.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    product_id1: Any
    product_id2: Optional[Any]

    def __init__(
            self, product_id1: Any, product_id2: Optional[Any], value: float,
            market_ids: Optional[Sequence] = None) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        super().__init__(value, market_ids)
        if product_id1 is None:
            raise ValueError("product_id1 cannot be None because the outside option cannot be removed.")
        self.product_id1 = product_id1
        self.product_id2 = product_id2

    def _format_value(self) -> str:
        """Construct a string expression for the covariance moment."""
        product1 = f"'{self.product_id1}'"
        product2 = "Outside" if self.product_id2 is None else f"'{self.product_id2}'"
        return f"P({product1} First, {product2} Second)"

    def _validate(self, economy: 'Economy') -> None:
        """Check that matrix indices are valid in the economy."""
        super()._validate(economy)
        economy._validate_product_id(self.product_id1, self.market_ids)
        economy._validate_product_id(self.product_id2, self.market_ids)


class DiversionCovarianceMoment(Moment):
    r"""Configuration for micro moments that match covariances between product characteristics of first and second
    choices.

    For example, survey data can sometimes be used to compute the sample covariance between a product characteristic
    :math:`x_{jt}^{(1)}` of an agent's first choice :math:`j` and either the same or a different product characteristic
    :math:`x_{kt}^{(2)}` of the agent's second choice :math:`k` if :math:`j` were removed from the choice set, amongst
    those agents whose first and second choices are both inside goods. With the value :math:`v_m` of this sample
    covariance, a micro moment :math:`m` in market :math:`t` for agent :math:`i` can be defined by

    .. math:: g_{M,imt} = (z_{it}^{(1)} - \bar{z}_t^{(1)})(z_{it}^{(2)} - \bar{z}_t^{(2)}) - v_m

    where :math:`\bar{z}_t^{(1)} = \sum_i w_{it} z_{it}^{(1)}`, :math:`\bar{z}_t^{(2)} = \sum_i w_{it} z_{it}^{(2)}`,
    and conditional on purchasing inside goods, the expected values of :math:`x_{jt}^{(1)}` and :math:`x_{kt}^{(2)}` for
    agent :math:`i` are

    .. math::

       z_{it}^{(1)} = \sum_{j \in J_t} x_{jt}^{(1)} s_{ij(-0)t}, \quad
       z_{it}^{(2)} = \sum_{j, k \in J_t} x_{kt}^{(2)} s_{ij(-0)t} s_{ik(-0,j)t}

    where :math:`s_{ij(-0)t}` is the probability of choosing :math:`j` when the outside option is removed from the
    choice set and :math:`s_{ik(-0,j)t}` is the probability of choosing :math:`k` when both the outside option and
    :math:`j` are removed from the choice set.

    Integrals of these micro moments are approximated within and averaged across a set :math:`T_m` of markets in which
    the micro data used to compute :math:`v_m` are relevant, which gives :math:`\bar{g}_{M,m}` in
    :eq:`averaged_micro_moments`.

    Parameters
    ----------
    X2_index1 : `int`
        Column index of :math:`x_{jt}^{(1)}` in the matrix of demand-side nonlinear product characteristics,
        :math:`X_2`. This should be between zero and :math:`K_2 - 1`, inclusive.
    X2_index2 : `int`
        Column index of :math:`x_{kt}^{(2)}` in the matrix of demand-side nonlinear product characteristics,
        :math:`X_2`. This should be between zero and :math:`K_2 - 1`, inclusive.
    value : `float`
        Value :math:`v_m` of the sample covariance estimated from micro data.
    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments will be averaged to get :math:`\bar{g}_{M,m}`. These are also
        the only markets in which the moments will be computed. By default, the moments are computed for and averaged
        across all markets. That is, by default, it is assumed that the specified ``value`` is relevant for and on
        average the same for all markets.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    X2_index1: int
    X2_index2: int

    def __init__(self, X2_index1: int, X2_index2: int, value: float, market_ids: Optional[Sequence] = None) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        super().__init__(value, market_ids)
        if not isinstance(X2_index1, int) or X2_index1 < 0:
            raise ValueError("X2_index1 must be a positive int.")
        if not isinstance(X2_index2, int) or X2_index2 < 0:
            raise ValueError("X2_index2 must be a positive int.")
        self.X2_index1 = X2_index1
        self.X2_index2 = X2_index2

    def _format_value(self) -> str:
        """Construct a string expression for the covariance moment."""
        return f"Cov(X2 Column {self.X2_index1} First, X2 Column {self.X2_index2} Second | Inside)"

    def _validate(self, economy: 'Economy') -> None:
        """Check that matrix indices are valid in the economy."""
        super()._validate(economy)
        if self.X2_index1 >= economy.K2:
            raise ValueError(f"X2_index1 must be between 0 and K2 = {economy.K2}, inclusive.")
        if self.X2_index2 >= economy.K2:
            raise ValueError(f"X2_index2 must be between 0 and K2 = {economy.K2}, inclusive.")


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

        # construct the leftmost part of the table that always shows up
        header = ["Index", "Markets", "Moment"]
        data: List[List[str]] = []
        for m, moment in enumerate(self.micro_moments):
            data.append([str(m), moment._format_markets(), moment._format_moment()])

        # add moment values
        if values is not None:
            header.append("Value")
            for m, value in enumerate(values):
                data[m].append(format_number(value))

        return format_table(header, *data, title=title)


class EconomyMoments(Moments):
    """Information about a sequence of micro moments in an economy."""

    market_indices: Dict[Hashable, Array]
    market_counts: Array
    pairwise_market_counts: Array

    def __init__(self, economy: 'Economy', micro_moments: Sequence[Moment]) -> None:
        """Validate and store information about a sequence of micro moment instances in the context of an economy."""

        # validate the moments
        if not isinstance(micro_moments, collections.abc.Sequence):
            raise TypeError("micro_moments must be a sequence of micro moment instances.")
        for moment in micro_moments:
            if not isinstance(moment, Moment):
                raise TypeError("micro_moments must consist only of micro moment instances.")
            try:
                moment._validate(economy)
            except Exception as exception:
                raise ValueError(f"The micro moment '{moment}' is invalid.") from exception

        # store basic moment information
        super().__init__(micro_moments)

        # identify market indices
        self.market_indices: Dict[Hashable, Array] = {}
        for t in economy.unique_market_ids:
            market_index_t = np.array([m.market_ids is None or t in m.market_ids for m in self.micro_moments])
            self.market_indices[t] = np.flatnonzero(market_index_t)

        # count the number of markets associated with moments
        self.market_counts = np.zeros((self.MM, 1), np.int)
        self.pairwise_market_counts = np.zeros((self.MM, self.MM), np.int)
        for m, moment_m in enumerate(self.micro_moments):
            market_ids_m = set(economy.unique_market_ids if moment_m.market_ids is None else moment_m.market_ids)
            self.market_counts[m] = len(market_ids_m)
            for n, moment_n in enumerate(self.micro_moments):
                market_ids_n = set(economy.unique_market_ids if moment_n.market_ids is None else moment_n.market_ids)
                self.pairwise_market_counts[m, n] = len(market_ids_m & market_ids_n)


class MarketMoments(Moments):
    """Information about a sequence of micro moments in a market."""

    def __init__(self, economy_moments: EconomyMoments, t: Hashable) -> None:
        """Select only those micro moment instances that will be computed for this market."""
        super().__init__([m for m in economy_moments.micro_moments if m.market_ids is None or t in m.market_ids])
