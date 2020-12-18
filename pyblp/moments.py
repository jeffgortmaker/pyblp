"""Micro moments."""

import abc
import collections
import functools
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from . import options
from .utilities.basics import Array, StringRepresentation, format_number, format_table


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .economies.economy import Economy  # noqa
    from .markets.market import Market  # noqa


class Moment(StringRepresentation):
    """Information about a single micro moment."""

    values: Array
    market_ids: Optional[Array]
    requires_inside: bool
    requires_eliminated: Sequence[Any]
    requires_inside_eliminated: bool

    def __init__(
            self, values: Any, market_ids: Optional[Sequence] = None, requires_inside: bool = False,
            requires_eliminated: Sequence[Any] = (), requires_inside_eliminated: bool = False) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        self.values = np.asarray(values, options.dtype)

        # validate market IDs
        if market_ids is None:
            self.market_ids = None
        else:
            self.market_ids = np.asarray(market_ids, np.object)

            # check for duplicates
            unique, counts = np.unique(self.market_ids, return_counts=True)
            duplicates = unique[counts > 1]
            if duplicates.size > 0:
                raise ValueError(f"The following market IDs are duplicated in market_ids: {duplicates}.")

            # validate shape against values
            if self.values.size not in {1, self.market_ids.size}:
                raise ValueError(
                    f"Micro moment values must be a scalar or, when market IDs are not None, have the same number of "
                    f"values as the number of market IDs."
                )

        # validate requirements
        assert not requires_inside_eliminated or requires_inside
        self.requires_inside = requires_inside
        self.requires_eliminated = requires_eliminated
        self.requires_inside_eliminated = requires_inside_eliminated

    def __str__(self) -> str:
        """Format information about the micro moment as a string."""
        return f"{self._format_markets()}: {self._format_moment()}"

    def _format_markets(self) -> str:
        """Format information about the markets associated with the micro moment as a string."""
        if self.market_ids is None:
            return "All"
        return ", ".join(str(t) for t in self.market_ids)

    @abc.abstractmethod
    def _format_moment(self) -> str:
        """Construct a string expression for the micro moment."""

    def _validate(self, economy: 'Economy') -> None:
        """Check that all market IDs associated with this moment are in the economy. If the moment is associated with
        all markets, validate the shape of its values.
        """
        if self.market_ids is not None:
            extra_ids = set(self.market_ids) - set(economy.unique_market_ids)
            if extra_ids:
                raise ValueError(f"market_ids contains the following extra IDs: {sorted(extra_ids)}.")
        elif self.values.size not in {1, economy.unique_market_ids.size}:
            raise ValueError(
                f"Micro moment values must be a scalar or, when market IDs are None, have the same number of values as"
                f"the number of distinct markets."
            )

    @abc.abstractmethod
    def _compute_agent_values(
            self, market: 'Market', delta: Array, probabilities: Array, conditionals: Optional[Array],
            inside_probabilities: Optional[Array], eliminated_probabilities: Dict[int, Array],
            inside_eliminated_sum: Optional[Array]) -> Array:
        """Compute agent-specific micro moment values, which will be aggregated up into means or covariances."""

    @abc.abstractmethod
    def _compute_agent_values_tangent(
            self, market: 'Market', p: int, delta: Array, probabilities: Array, probabilities_tangent: Array,
            inside_probabilities: Optional[Array], inside_tangent: Optional[Array],
            eliminated_tangents: Dict[int, Array], inside_eliminated_sum: Optional[Array],
            inside_eliminated_sum_tangent: Optional[Array]) -> Array:
        """Compute the tangent of agent-specific micro moments with respect to a parameter."""


class DemographicExpectationMoment(Moment):
    r"""Configuration for micro moments that match expectations of demographics for agents who choose certain products.

    For example, micro data can sometimes be used to compute the mean of a demographic such as income, :math:`y_{it}`,
    for agents who choose products in some set :math:`J`. With the value :math:`\mathscr{V}_{mt}` of this mean, a
    micro moment :math:`m` in market :math:`t` can be defined by :math:`g_{M,mt} = \mathscr{V}_{mt} - v_{mt}` where

    .. math:: v_{mt} = \frac{E[y_{it} \sum_{j \in J} s_{ijt}]}{\sum_{j \in J} s_{jt}}.

    These micro moments are averaged across a set :math:`T_m` of markets, which gives :math:`\bar{g}_{M,m}` in
    :eq:`averaged_micro_moments`.

    Parameters
    ----------
    product_ids : `sequence of object`
        IDs of the products :math:`j \in J`, which may include ``None`` to denote the outside option :math:`j = 0`. If
        there is no ``None`, at least one of these IDs should show up in the ``product_ids`` field of ``product_data``
        in :class:`Problem` or :class:`Simulation` for each market over which this micro moment will be averaged.
    demographics_index : `int`
        Column index of the demographic :math:`y_{it}` (which can be any demographic, not just income) in the matrix of
        agent demographics, :math:`d`. This should be between zero and :math:`D - 1`, inclusive.
    values : `float`
        Values :math:`\mathscr{V}_{mt}` of the statistic estimated from micro data. If a scalar is specified, then
        :math:`\mathscr{V}_{mt} = \mathscr{V}_m` is assumed to be constant across all markets in which the moment is
        relevant. Otherwise, this should have as many elements as ``market_ids``, or as the total number of markets if
        ``market_ids`` is ``None``.
    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments will be averaged to get :math:`\bar{g}_{M,m}`. These are also
        the only markets in which the moments will be computed. By default, the moments are computed for and averaged
        across all markets.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    product_ids: Sequence[Any]
    demographics_index: int

    def __init__(
            self, product_ids: Optional[Any], demographics_index: int, values: Any,
            market_ids: Optional[Sequence] = None) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        if not isinstance(product_ids, collections.abc.Sequence) or len(product_ids) == 0:
            raise ValueError("product_ids must be a sequence with at least one ID.")
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("product_ids should not have duplicates.")
        if not isinstance(demographics_index, int) or demographics_index < 0:
            raise ValueError("demographics_index must be a positive int.")
        super().__init__(values, market_ids)
        self.product_ids = product_ids
        self.demographics_index = demographics_index

    def _format_moment(self) -> str:
        """Construct a string expression for the moment."""
        products = ", ".join("Outside" if i is None else f"'{i}'" for i in self.product_ids)
        return f"E[Demographic Column {self.demographics_index} | {products}]"

    def _validate(self, economy: 'Economy') -> None:
        """Check that matrix indices are valid in the economy."""
        super()._validate(economy)
        economy._validate_product_ids(self.product_ids, self.market_ids)
        if self.demographics_index >= economy.D:
            raise ValueError(f"demographics_index must be between 0 and D = {economy.D}, inclusive.")

    def _compute_agent_values(
            self, market: 'Market', delta: Array, probabilities: Array, conditionals: Optional[Array],
            inside_probabilities: Optional[Array], eliminated_probabilities: Dict[int, Array],
            inside_eliminated_sum: Optional[Array]) -> Array:
        """Compute agent-specific micro moment values, which will be aggregated up into means or covariances."""
        shares_sum = 0
        probabilities_sum = np.zeros((market.I, 1), options.dtype)
        for product_id in self.product_ids:
            if product_id is None:
                shares_sum += 1 - market.products.shares.sum()
                probabilities_sum += 1 - probabilities.sum(axis=0, keepdims=True).T
            elif product_id in market.products.product_ids:
                j = market.get_product(product_id)
                shares_sum += market.products.shares[j]
                probabilities_sum += probabilities[[j]].T

        d = market.agents.demographics[:, [self.demographics_index]]
        return d * probabilities_sum / shares_sum

    def _compute_agent_values_tangent(
            self, market: 'Market', p: int, delta: Array, probabilities: Array, probabilities_tangent: Array,
            inside_probabilities: Optional[Array], inside_tangent: Optional[Array],
            eliminated_tangents: Dict[int, Array], inside_eliminated_sum: Optional[Array],
            inside_eliminated_sum_tangent: Optional[Array]) -> Array:
        """Compute the tangent of agent-specific micro moments with respect to a parameter."""
        shares_sum = 0
        probabilities_tangent_sum = np.zeros((market.I, 1), options.dtype)
        for product_id in self.product_ids:
            if product_id is None:
                shares_sum += 1 - market.products.shares.sum()
                probabilities_tangent_sum += -probabilities_tangent.sum(axis=0, keepdims=True).T
            elif product_id in market.products.product_ids:
                j = market.get_product(product_id)
                shares_sum += market.products.shares[j]
                probabilities_tangent_sum += probabilities_tangent[[j]].T

        d = market.agents.demographics[:, [self.demographics_index]]
        return d * probabilities_tangent_sum / shares_sum


class CharacteristicExpectationMoment(Moment):
    r"""Configuration for micro moments that match expectations of characteristics of products chosen by certain agents.

    For example, micro data can sometimes be used to compute the mean of a product characteristic :math:`x_{jt}` of an
    agent's choice :math:`j` for agents in some set :math:`I`. With the value :math:`\mathscr{V}_{mt}` of this mean, a
    micro moment :math:`m` in market :math:`t` can be defined by :math:`g_{M,mt} = \mathscr{V}_{mt} - v_{mt}` where

    .. math:: v_{mt} = E[z_{it} | i \in I]

    where conditional on choosing an inside good, the expected value of :math:`x_{jt}` for agent :math:`i` is

    .. math:: z_{it} = \sum_{j \in J_t} x_{jt}s_{ij(-0)t}

    where :math:`s_{ij(-0)t} = s_{ijt} / (1 - s_{i0t})` is the probability of :math:`i` choosing :math:`j` when the
    outside option is removed from the choice set.

    These micro moments are averaged across a set :math:`T_m` of markets, which gives :math:`\bar{g}_{M,m}` in
    :eq:`averaged_micro_moments`.

    Parameters
    ----------
    agent_ids : `sequence of object`
        IDs of the agents :math:`i \in I`. At least one of these IDs should show up in the ``agent_ids`` field of
        ``agent_data`` in :class:`Problem` or :class:`Simulation` for each market over which this micro moment will be
        averaged.
    X2_index : `int`
        Column index of :math:`x_{jt}` in the matrix of demand-side nonlinear product characteristics, :math:`X_2`. This
        should be between zero and :math:`K_2 - 1`, inclusive.
    values : `float`
        Values :math:`\mathscr{V}_{mt}` of the statistic estimated from micro data. If a scalar is specified, then
        :math:`\mathscr{V}_{mt} = \mathscr{V}_m` is assumed to be constant across all markets in which the moment is
        relevant. Otherwise, this should have as many elements as ``market_ids``, or as the total number of markets if
        ``market_ids`` is ``None``.
    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments will be averaged to get :math:`\bar{g}_{M,m}`. These are also
        the only markets in which the moments will be computed. By default, the moments are computed for and averaged
        across all markets.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    agent_ids: Sequence[Any]
    X2_index: int

    def __init__(
            self, agent_ids: Optional[Any], X2_index: int, values: Any,
            market_ids: Optional[Sequence] = None) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        if not isinstance(agent_ids, collections.abc.Sequence) or len(agent_ids) == 0:
            raise ValueError("agent_ids must be a sequence with at least one ID.")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent_ids should not have duplicates.")
        if not isinstance(X2_index, int) or X2_index < 0:
            raise ValueError("X2_index must be a positive int.")
        super().__init__(values, market_ids, requires_inside=True)
        self.agent_ids = agent_ids
        self.X2_index = X2_index

    def _format_moment(self) -> str:
        """Construct a string expression for the moment."""
        agents = ", ".join(f"'{i}'" for i in self.agent_ids)
        return f"E[X2 Column {self.X2_index} | {agents}]"

    def _validate(self, economy: 'Economy') -> None:
        """Check that matrix indices are valid in the economy."""
        super()._validate(economy)
        economy._validate_agent_ids(self.agent_ids, self.market_ids)
        if self.X2_index >= economy.K2:
            raise ValueError(f"X2_index must be between 0 and K2 = {economy.K2}, inclusive.")

    def _compute_agent_values(
            self, market: 'Market', delta: Array, probabilities: Array, conditionals: Optional[Array],
            inside_probabilities: Optional[Array], eliminated_probabilities: Dict[int, Array],
            inside_eliminated_sum: Optional[Array]) -> Array:
        """Compute agent-specific micro moment values, which will be aggregated up into means or covariances."""
        assert inside_probabilities is not None
        x = market.products.X2[:, [self.X2_index]]

        weights_sum = 0
        x_means = np.zeros((market.I, 1), options.dtype)
        for agent_id in self.agent_ids:
            if agent_id in market.agents.agent_ids:
                i = market.get_agent(agent_id)
                weights_sum += market.agents.weights[i]
                x_means[i] = inside_probabilities[:, i] @ x

        return x_means / weights_sum

    def _compute_agent_values_tangent(
            self, market: 'Market', p: int, delta: Array, probabilities: Array, probabilities_tangent: Array,
            inside_probabilities: Optional[Array], inside_tangent: Optional[Array],
            eliminated_tangents: Dict[int, Array], inside_eliminated_sum: Optional[Array],
            inside_eliminated_sum_tangent: Optional[Array]) -> Array:
        """Compute the tangent of agent-specific micro moments with respect to a parameter."""
        assert inside_tangent is not None
        x = market.products.X2[:, [self.X2_index]]

        weights_sum = 0
        x_mean_tangents = np.zeros((market.I, 1), options.dtype)
        for agent_id in self.agent_ids:
            if agent_id in market.agents.agent_ids:
                i = market.get_agent(agent_id)
                weights_sum += market.agents.weights[i]
                x_mean_tangents[i] = inside_tangent[:, i] @ x

        return x_mean_tangents / weights_sum


class DemographicCovarianceMoment(Moment):
    r"""Configuration for micro moments that match covariances between product characteristics and demographics.

    For example, micro data can sometimes be used to compute the sample covariance between a product characteristic
    :math:`x_{jt}` of an agent's choice :math:`j`, and a demographic such as income, :math:`y_{it}`, amongst those
    agents who purchase an inside good. With the value :math:`\mathscr{V}_{mt}` of this sample covariance, a micro
    moment :math:`m` in market :math:`t` can be defined by :math:`g_{M,mt} = \mathscr{V}_{mt} - v_{mt}` where

    .. math:: v_{mt} = \text{Cov}(y_{it}, z_{it})

    where conditional on choosing an inside good, the expected value of :math:`x_{jt}` for agent :math:`i` is

    .. math:: z_{it} = \sum_{j \in J_t} x_{jt}s_{ij(-0)t}

    where :math:`s_{ij(-0)t} = s_{ijt} / (1 - s_{i0t})` is the probability of :math:`i` choosing :math:`j` when the
    outside option is removed from the choice set.

    These micro moments are averaged across a set :math:`T_m` of markets, which gives :math:`\bar{g}_{M,m}` in
    :eq:`averaged_micro_moments`.

    Parameters
    ----------
    X2_index : `int`
        Column index of :math:`x_{jt}` in the matrix of demand-side nonlinear product characteristics, :math:`X_2`. This
        should be between zero and :math:`K_2 - 1`, inclusive.
    demographics_index : `int`
        Column index of the demographic :math:`y_{it}` (which can be any demographic, not just income) in the matrix of
        agent demographics, :math:`d`. This should be between zero and :math:`D - 1`, inclusive.
    values : `float`
        Values :math:`\mathscr{V}_{mt}` of the statistic estimated from micro data. If a scalar is specified, then
        :math:`\mathscr{V}_{mt} = \mathscr{V}_m` is assumed to be constant across all markets in which the moment is
        relevant. Otherwise, this should have as many elements as ``market_ids``, or as the total number of markets if
        ``market_ids`` is ``None``.
    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments will be averaged to get :math:`\bar{g}_{M,m}`. These are also
        the only markets in which the moments will be computed. By default, the moments are computed for and averaged
        across all markets.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    X2_index: int
    demographics_index: int

    def __init__(
            self, X2_index: int, demographics_index: int, values: Any, market_ids: Optional[Sequence] = None) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        if not isinstance(X2_index, int) or X2_index < 0:
            raise ValueError("X2_index must be a positive int.")
        if not isinstance(demographics_index, int) or demographics_index < 0:
            raise ValueError("demographics_index must be a positive int.")
        super().__init__(values, market_ids, requires_inside=True)
        self.X2_index = X2_index
        self.demographics_index = demographics_index

    def _format_moment(self) -> str:
        """Construct a string expression for the moment."""
        return f"Cov(X2 Column {self.X2_index}, Demographic Column {self.demographics_index})"

    def _validate(self, economy: 'Economy') -> None:
        """Check that matrix indices are valid in the economy."""
        super()._validate(economy)
        if self.X2_index >= economy.K2:
            raise ValueError(f"X2_index must be between 0 and K2 = {economy.K2}, inclusive.")
        if self.demographics_index >= economy.D:
            raise ValueError(f"demographics_index must be between 0 and D = {economy.D}, inclusive.")

    def _compute_agent_values(
            self, market: 'Market', delta: Array, probabilities: Array, conditionals: Optional[Array],
            inside_probabilities: Optional[Array], eliminated_probabilities: Dict[int, Array],
            inside_eliminated_sum: Optional[Array]) -> Array:
        """Compute agent-specific micro moment values, which will be aggregated up into means or covariances."""
        assert inside_probabilities is not None
        x = market.products.X2[:, [self.X2_index]]
        d = market.agents.demographics[:, [self.demographics_index]]
        z = inside_probabilities.T @ x
        demeaned_z = z - market.agents.weights.T @ z
        demeaned_d = d - market.agents.weights.T @ d
        return demeaned_z * demeaned_d

    def _compute_agent_values_tangent(
            self, market: 'Market', p: int, delta: Array, probabilities: Array, probabilities_tangent: Array,
            inside_probabilities: Optional[Array], inside_tangent: Optional[Array],
            eliminated_tangents: Dict[int, Array], inside_eliminated_sum: Optional[Array],
            inside_eliminated_sum_tangent: Optional[Array]) -> Array:
        """Compute the tangent of agent-specific micro moments with respect to a parameter."""
        assert inside_tangent is not None
        x = market.products.X2[:, [self.X2_index]]
        d = market.agents.demographics[:, [self.demographics_index]]
        z_tangent = inside_tangent.T @ x
        demeaned_z_tangent = z_tangent - market.agents.weights.T @ z_tangent
        demeaned_d = d - market.agents.weights.T @ d
        return demeaned_z_tangent * demeaned_d


class DiversionProbabilityMoment(Moment):
    r"""Configuration for micro moments that match second choice probabilities of certain products for agents whose
    first choices are certain other products.

    For example, micro data can sometimes be used to compute the share of agents who would choose product :math:`k` if
    :math:`j` were removed from the choice set, out of those agents whose first choice is :math:`j`. With the value
    :math:`\mathscr{V}_{mt}` of this share, a micro moment :math:`m` in market :math:`t` can be defined by
    :math:`g_{M,mt} = \mathscr{V}_{mt} - v_{mt}` where

    .. math:: v_{mt} = \frac{E[s_{ik(-j)t} s_{ijt}]}{s_{jt}}

    where :math:`s_{ik(-j)t} = s_{ijt} / (1 - s_{ijt})` is the probability of :math:`i` choosing :math:`k` when
    :math:`j` is removed from the choice set. Rearranging terms gives the equivalent definition

    .. math:: g_{M,mt} = \mathscr{V}_{mt} - \frac{s_{k(-j)t} - s_{kt}}{s_{jt}},

    which is more reminiscent of the long-run diversion ratios :math:`\bar{\mathscr{D}}_{jk}` computed by
    :meth:`ProblemResults.compute_long_run_diversion_ratios`.

    These micro moments are averaged across a set :math:`T_m` of markets, which gives :math:`\bar{g}_{M,m}` in
    :eq:`averaged_micro_moments`.

    Parameters
    ----------
    product_id1 : `object`
        ID of the first choice product :math:`j` or ``None`` to denote the outside option :math:`j = 0`. There must be
        exactly one of this ID in the ``product_ids`` field of ``product_data`` in :class:`Problem` or
        :class:`Simulation` for each market over which this micro moment will be averaged.
    product_id2 : `object`
        ID of the second choice product :math:`k` or ``None`` to denote the outside option :math:`j = 0`. If not
        ``None``, there must be exactly one of this ID for each market over which this micro moment will be averaged.
    values : `float`
        Values :math:`\mathscr{V}_{mt}` of the statistic estimated from micro data. If a scalar is specified, then
        :math:`\mathscr{V}_{mt} = \mathscr{V}_m` is assumed to be constant across all markets in which the moment is
        relevant. Otherwise, this should have as many elements as ``market_ids``, or as the total number of markets if
        ``market_ids`` is ``None``.
    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments will be averaged to get :math:`\bar{g}_{M,m}`. These are also
        the only markets in which the moments will be computed. By default, the moments are computed for and averaged
        across all markets.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    product_id1: Optional[Any]
    product_id2: Optional[Any]

    def __init__(
            self, product_id1: Any, product_id2: Optional[Any], values: Any,
            market_ids: Optional[Sequence] = None) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        if product_id1 is None and product_id2 is None:
            raise ValueError("At least one of product_id1 or product_id2 must be not None.")
        super().__init__(
            values, market_ids, requires_inside=product_id1 is None,
            requires_eliminated=[] if product_id1 is None else [product_id1]
        )
        self.product_id1 = product_id1
        self.product_id2 = product_id2

    def _format_moment(self) -> str:
        """Construct a string expression for the moment."""
        product1 = "Outside" if self.product_id1 is None else f"'{self.product_id1}'"
        product2 = "Outside" if self.product_id2 is None else f"'{self.product_id2}'"
        return f"P({product1} First, {product2} Second)"

    def _validate(self, economy: 'Economy') -> None:
        """Check that matrix indices are valid in the economy."""
        super()._validate(economy)
        economy._validate_product_ids([self.product_id1], self.market_ids)
        economy._validate_product_ids([self.product_id2], self.market_ids)

    def _compute_agent_values(
            self, market: 'Market', delta: Array, probabilities: Array, conditionals: Optional[Array],
            inside_probabilities: Optional[Array], eliminated_probabilities: Dict[int, Array],
            inside_eliminated_sum: Optional[Array]) -> Array:
        """Compute agent-specific micro moment values, which will be aggregated up into means or covariances."""

        # match the second choice probability of a certain inside good for agents who choose the outside good
        if self.product_id1 is None:
            assert inside_probabilities is not None
            k = market.get_product(self.product_id2)
            outside_share = 1 - market.products.shares.sum()
            numerator = inside_probabilities[[k]].T - market.products.shares[k]
            return numerator / outside_share

        # match the second choice probability of the outside good for agents who choose a certain inside good
        if self.product_id2 is None:
            j = market.get_product(self.product_id1)
            eliminated_outside_probabilities = 1 - eliminated_probabilities[j].sum(axis=0, keepdims=True)
            outside_share = 1 - market.products.shares.sum()
            numerator = eliminated_outside_probabilities.T - outside_share
            return numerator / market.products.shares[j]

        # match the second choice probability of a certain inside good for agents who choose a certain inside good
        j = market.get_product(self.product_id1)
        k = market.get_product(self.product_id2)
        numerator = eliminated_probabilities[j][[k]].T - market.products.shares[k]
        return numerator / market.products.shares[j]

    def _compute_agent_values_tangent(
            self, market: 'Market', p: int, delta: Array, probabilities: Array, probabilities_tangent: Array,
            inside_probabilities: Optional[Array], inside_tangent: Optional[Array],
            eliminated_tangents: Dict[int, Array], inside_eliminated_sum: Optional[Array],
            inside_eliminated_sum_tangent: Optional[Array]) -> Array:
        """Compute the tangent of agent-specific micro moments with respect to a parameter."""

        # handle the second choice probability of a certain inside good for agents who choose the outside good
        if self.product_id1 is None:
            assert inside_tangent is not None
            k = market.get_product(self.product_id2)
            outside_share = 1 - market.products.shares.sum()
            return inside_tangent[[k]].T / outside_share

        # handle the second choice probability of the outside good for agents who choose a certain inside good
        if self.product_id2 is None:
            j = market.get_product(self.product_id1)
            eliminated_outside_tangent = -eliminated_tangents[j].sum(axis=0, keepdims=True)
            return eliminated_outside_tangent.T / market.products.shares[j]

        # handle the second choice probability of a certain inside good for agents who choose a certain inside good
        j = market.get_product(self.product_id1)
        k = market.get_product(self.product_id2)
        return eliminated_tangents[j][[k]].T / market.products.shares[j]


class DiversionCovarianceMoment(Moment):
    r"""Configuration for micro moments that match covariances between product characteristics of first and second
    choices.

    For example, survey data can sometimes be used to compute the sample covariance between a product characteristic
    :math:`x_{jt}^{(1)}` of an agent's first choice :math:`j` and either the same or a different product characteristic
    :math:`x_{kt}^{(2)}` of the agent's second choice :math:`k` if :math:`j` were removed from the choice set, amongst
    those agents whose first and second choices are both inside goods. With the value :math:`\mathscr{V}_{mt}` of this
    sample covariance, a micro moment :math:`m` in market :math:`t` can be defined by
    :math:`g_{M,mt} = \mathscr{V}_{mt} - v_{mt}` where

    .. math:: v_{mt} = \text{Cov}(z_{it}^{(1)}, z_{it}^{(2)})

    where conditional on purchasing inside goods, the expected values of :math:`x_{jt}^{(1)}` and
    :math:`x_{kt}^{(2)}` for agent :math:`i` are

    .. math::

       z_{it}^{(1)} = \sum_{j \in J_t} x_{jt}^{(1)} s_{ij(-0)t}, \quad
       z_{it}^{(2)} = \sum_{j, k \in J_t} x_{kt}^{(2)} s_{ik(-0,j)t} s_{ij(-0)t}

    where :math:`s_{ij(-0)t}` is the probability of choosing :math:`j` when the outside option is removed from the
    choice set and :math:`s_{ik(-0,j)t}` is the probability of choosing :math:`k` when both the outside option and
    :math:`j` are removed from the choice set.

    These micro moments are averaged across a set :math:`T_m` of markets, which gives :math:`\bar{g}_{M,m}` in
    :eq:`averaged_micro_moments`.

    Parameters
    ----------
    X2_index1 : `int`
        Column index of :math:`x_{jt}^{(1)}` in the matrix of demand-side nonlinear product characteristics,
        :math:`X_2`. This should be between zero and :math:`K_2 - 1`, inclusive.
    X2_index2 : `int`
        Column index of :math:`x_{kt}^{(2)}` in the matrix of demand-side nonlinear product characteristics,
        :math:`X_2`. This should be between zero and :math:`K_2 - 1`, inclusive.
    values : `float`
        Values :math:`\mathscr{V}_{mt}` of the statistic estimated from micro data. If a scalar is specified, then
        :math:`\mathscr{V}_{mt} = \mathscr{V}_m` is assumed to be constant across all markets in which the moment is
        relevant. Otherwise, this should have as many elements as ``market_ids``, or as the total number of markets if
        ``market_ids`` is ``None``.
    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments will be averaged to get :math:`\bar{g}_{M,m}`. These are also
        the only markets in which the moments will be computed. By default, the moments are computed for and averaged
        across all markets.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    X2_index1: int
    X2_index2: int

    def __init__(self, X2_index1: int, X2_index2: int, values: Any, market_ids: Optional[Sequence] = None) -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        if not isinstance(X2_index1, int) or X2_index1 < 0:
            raise ValueError("X2_index1 must be a positive int.")
        if not isinstance(X2_index2, int) or X2_index2 < 0:
            raise ValueError("X2_index2 must be a positive int.")
        super().__init__(values, market_ids, requires_inside=True, requires_inside_eliminated=True)
        self.X2_index1 = X2_index1
        self.X2_index2 = X2_index2

    def _format_moment(self) -> str:
        """Construct a string expression for the moment."""
        return f"Cov(X2 Column {self.X2_index1} First, X2 Column {self.X2_index2} Second)"

    def _validate(self, economy: 'Economy') -> None:
        """Check that matrix indices are valid in the economy."""
        super()._validate(economy)
        if self.X2_index1 >= economy.K2:
            raise ValueError(f"X2_index1 must be between 0 and K2 = {economy.K2}, inclusive.")
        if self.X2_index2 >= economy.K2:
            raise ValueError(f"X2_index2 must be between 0 and K2 = {economy.K2}, inclusive.")

    def _compute_agent_values(
            self, market: 'Market', delta: Array, probabilities: Array, conditionals: Optional[Array],
            inside_probabilities: Optional[Array], eliminated_probabilities: Dict[int, Array],
            inside_eliminated_sum: Optional[Array]) -> Array:
        """Compute agent-specific micro moment values, which will be aggregated up into means or covariances."""
        assert inside_probabilities is not None and inside_eliminated_sum is not None
        x1 = market.products.X2[:, [self.X2_index1]]
        x2 = market.products.X2[:, [self.X2_index2]]
        z1 = inside_probabilities.T @ x1
        z2 = inside_eliminated_sum.T @ x2
        demeaned_z1 = z1 - market.agents.weights.T @ z1
        demeaned_z2 = z2 - market.agents.weights.T @ z2
        return demeaned_z1 * demeaned_z2

    def _compute_agent_values_tangent(
            self, market: 'Market', p: int, delta: Array, probabilities: Array, probabilities_tangent: Array,
            inside_probabilities: Optional[Array], inside_tangent: Optional[Array],
            eliminated_tangents: Dict[int, Array], inside_eliminated_sum: Optional[Array],
            inside_eliminated_sum_tangent: Optional[Array]) -> Array:
        """Compute the tangent of agent-specific micro moments with respect to a parameter."""
        assert inside_probabilities is not None and inside_tangent is not None
        assert inside_eliminated_sum is not None and inside_eliminated_sum_tangent is not None
        x1 = market.products.X2[:, [self.X2_index1]]
        x2 = market.products.X2[:, [self.X2_index2]]
        z1 = inside_probabilities.T @ x1
        z1_tangent = inside_tangent.T @ x1
        z2 = inside_eliminated_sum.T @ x2
        z2_tangent = inside_eliminated_sum_tangent.T @ x2
        demeaned_z1 = z1 - market.agents.weights.T @ z1
        demeaned_z1_tangent = z1_tangent - market.agents.weights.T @ z1_tangent
        demeaned_z2 = z2 - market.agents.weights.T @ z2
        demeaned_z2_tangent = z2_tangent - market.agents.weights.T @ z2_tangent
        return demeaned_z1_tangent * demeaned_z2 + demeaned_z1 * demeaned_z2_tangent


class CustomMoment(Moment):
    r"""Configuration for custom micro moments.

    This configuration requires values :math:`\mathscr{V}_{mt}` computed from survey data, for example. It also requires
    a function that computes these values' simulated counterparts in order to form a micro moment :math:`m` in market
    :math:`t` defined by :math:`g_{M,mt} = \mathscr{V}_{mt} - v_{mt}` where

    .. math:: v_{mt} = \sum_{i \in I_t} w_{it} v_{imt},

    a simulated integral over agent-specific micro values :math:`v_{imt}` computed according to a custom function.

    These micro moments are averaged across a set :math:`T_m` of markets, which gives :math:`\bar{g}_{M,m}` in
    :eq:`averaged_micro_moments`.

    Parameters
    ----------
    values : `float`
        Values :math:`\mathscr{V}_{mt}` of the statistic estimated from micro data. If a scalar is specified, then
        :math:`\mathscr{V}_{mt} = \mathscr{V}_m` is assumed to be constant across all markets in which the moment is
        relevant. Otherwise, this should have as many elements as ``market_ids``, or as the total number of markets if
        ``market_ids`` is ``None``.
    compute_custom : `callable`
        Function that computes :math:`v_{imt}` in a single market :math:`t`, which is of the following form::

            compute_custom(t, sigma, pi, rho, products, agents, delta, mu, probabilities) -> custom

        where

            - ``t`` is the ID of the market in which the :math:`v_{imt}` should be computed;

            - ``sigma`` is the Cholesky root of the covariance matrix for unobserved taste heterogeneity,
              :math:`\Sigma`, which will be empty if there are no such parameters;

            - ``pi`` are parameters that measure how agent tastes vary with demographics, :math:`\Pi`, which will be
              empty if there are no such parameters;

            - ``rho`` is a :math:`J_t \times 1` vector with parameters that measure within nesting group correlations
              for each product, :math:`\rho_{h(j)}`, which will be empty if there is no nesting structure;

            - ``products`` is a :class:`Products` instance containing product data for the current market;

            - ``agents`` is an :class:`Agents` instance containing agent data for the current market;

            - ``delta`` is a :math:`J_t \times 1` vector of mean utilities :math:`\delta_{jt}`;

            - ``mu`` is a :math:`J_t \times I_t` matrix of agent-specific utilities :math:`\mu_{ijt}`;

            - ``probabilities`` is a :math:`J_t \times I_t` matrix of choice probabilities :math:`s_{ijt}`; and

            - ``custom`` is an :math:`I_t \times 1`` vector of agent-specific micro values :math:`v_{imt}`.

    compute_custom_derivatives : `callable, optional`
        Function that computes :math:`\frac{\partial v_{imt}}{\partial \theta_p}` in a single market :math:`t`, which is
        of the following form::

            compute_custom_derivatives(t, sigma, pi, rho, products, agents, delta, mu, probabilities, p, derivatives) ->
            custom_derivatives

        where the first few arguments are the same as above,

            - ``p`` is the index :math:`p \in \{0, \dots, P - 1\}` of :math:`\theta_p` (for the ordering of the
              :math:`P` parameters in :math:`\theta`, see :attr:`ProblemResults.theta`),

            - ``derivatives`` is a :math:`J_t \times I_t` matrix of derivatives
              :math:`\frac{\partial s_{ijt}}{\partial \theta_p}`, and

            - ``custom_derivatives`` is an :math:`I_t \times 1` vector of agent-specific micro value derivatives
              :math:`\frac{\partial v_{imt}}{\partial \theta_p}`.

        If this function is left unspecified, you must set ``finite_differences=True`` in :meth:`Problem.solve` when
        using custom moments. This may slow down optimization and slightly reduce the numerical accuracy of standard
        errors.

        If you specify this function, to check that you have implemented derivatives correctly, you can pass
        ``optimization=Optimization('return')`` to :meth:`Problem.solve` when evaluating the gradient with
        ``finite_differences=True`` and ``finite_differences=False``. If the numerical gradient is close to the analytic
        one, this suggests that you have implemented derivatives correctly.

    market_ids : `array-like, optional`
        Distinct market IDs over which the micro moments will be averaged to get :math:`\bar{g}_{M,m}`. These are also
        the only markets in which the moments will be computed. By default, the moments are computed for and averaged
        across all markets.
    name : `str, optional`
        Name of the custom moment, which will be used when displaying information about micro moments. By default, this
        is ``"Custom"``.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    name: str
    compute_custom: functools.partial
    compute_custom_derivatives: Optional[functools.partial]

    def __init__(
            self, values: Any, compute_custom: Callable, compute_custom_derivatives: Optional[Callable] = None,
            market_ids: Optional[Sequence] = None, name: str = "Custom") -> None:
        """Validate information about the moment to the greatest extent possible without an economy instance."""
        if not callable(compute_custom):
            raise ValueError("compute_custom must be callable.")
        if compute_custom_derivatives is not None and not callable(compute_custom_derivatives):
            raise ValueError("compute_custom_derivatives must be None or callable.")
        super().__init__(values, market_ids)
        self.name = name
        self.compute_custom = functools.partial(compute_custom)
        if compute_custom_derivatives is None:
            self.compute_custom_derivatives = None
        else:
            self.compute_custom_derivatives = functools.partial(compute_custom_derivatives)

    def _format_moment(self) -> str:
        """The expression for the moment is just the specified name."""
        return self.name

    def _compute_agent_values(
            self, market: 'Market', delta: Array, probabilities: Array, conditionals: Optional[Array],
            inside_probabilities: Optional[Array], eliminated_probabilities: Dict[int, Array],
            inside_eliminated_sum: Optional[Array]) -> Array:
        """Compute agent-specific micro moment values, which will be aggregated up into means or covariances."""
        values = self.compute_custom(
            market.t, market.sigma, market.pi, market.rho, market.products, market.agents, delta, market.mu,
            probabilities
        )
        values = np.asarray(values, options.dtype)
        if values.size != market.I:
            raise ValueError("compute_custom must return a vector with as many elements as agents.")
        return values.reshape((market.I, 1))

    def _compute_agent_values_tangent(
            self, market: 'Market', p: int, delta: Array, probabilities: Array, probabilities_tangent: Array,
            inside_probabilities: Optional[Array], inside_tangent: Optional[Array],
            eliminated_tangents: Dict[int, Array], inside_eliminated_sum: Optional[Array],
            inside_eliminated_sum_tangent: Optional[Array]) -> Array:
        """Compute the tangent of agent-specific micro moments with respect to a parameter."""
        assert self.compute_custom_derivatives is not None
        derivatives = self.compute_custom_derivatives(
            market.t, market.sigma, market.pi, market.rho, market.products, market.agents, delta, market.mu,
            probabilities, p, probabilities_tangent
        )
        derivatives = np.asarray(derivatives, options.dtype)
        if derivatives.size != market.I:
            raise ValueError("compute_custom_derivatives must return a vector with as many elements as agents.")
        return derivatives.reshape((market.I, 1))


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
        header = ["Index", "Markets", "Type"]
        data: List[List[str]] = []
        for m, moment in enumerate(self.micro_moments):
            data.append([str(m), moment._format_markets(), moment._format_moment()])

        # add moment values
        if values is not None:
            header.append("Moment")
            for m, value in enumerate(values):
                data[m].append(format_number(value))

        return format_table(header, *data, title=title)


class EconomyMoments(Moments):
    """Information about a sequence of micro moments in an economy."""

    market_indices: Dict[Hashable, Array]
    market_values: Array
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

        # identify moment indices that are relevant in each market along with the associated observed micro values
        self.market_indices: Dict[Hashable, Array] = {}
        self.market_values: Dict[Hashable, Array] = {}
        for t in economy.unique_market_ids:
            indices = []
            values = []
            for m, moment in enumerate(self.micro_moments):
                market_ids_m = economy.unique_market_ids if moment.market_ids is None else moment.market_ids
                if t in market_ids_m:
                    indices.append(m)
                    values.append(moment.values if moment.values.size == 1 else moment.values[market_ids_m == t])

            self.market_indices[t] = np.array(indices, np.int)
            self.market_values[t] = np.array(values, options.dtype).flatten()

        # count the number of markets associated with moments
        self.market_counts = np.zeros(self.MM, np.int)
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
