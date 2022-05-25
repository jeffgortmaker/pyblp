"""Micro moments."""

import collections.abc
import functools
from typing import Callable, List, Optional, Sequence, Set, TYPE_CHECKING, Union

import numpy as np

from .utilities.basics import Array, StringRepresentation, format_number, format_table


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .economies.economy import Economy  # noqa


class MicroDataset(StringRepresentation):
    r"""Configuration for a micro dataset :math:`d` on which micro moments are computed.

    A micro dataset :math:`d`, often a survey, is defined by survey weights :math:`w_{dijt}`, which are used in
    :eq:`averaged_micro_moments`. For example, :math:`w_{dijt} = 1\{j \neq 0, t \in T_d\}` defines a micro dataset that
    is a selected sample of inside purchasers in a few markets :math:`T_d \subset T`, giving each market an equal
    sampling weight. Different micro datasets are independent.

    .. warning::

        Micro moments are under active development. Their API and functionality may change as development progresses.

    Parameters
    ----------
    name : str
        The unique name of the dataset, which will be used for outputting information about micro moments.
    observations : int
        The number of observations :math:`N_d` in the micro dataset.
    compute_weights : `callable`
        Function for computing survey weights :math:`w_{dijt}` in a market of the following form::

            compute_weights(t, products, agents) --> weights

        where ``t`` is the market in which to compute weights, ``products`` is the market's :class:`Products` (with
        :math:`J_t` rows), and ``agents`` is the market's :class:`Agents` (with :math:`I_t` rows), unless
        ``pyblp.options.micro_computation_chunks`` is larger than its default of ``1``, in which case ``agents`` is a
        chunk of the market's :class:`Agents`. Denoting the number of rows in ``agents`` by :math:`I`, the returned
        ``weights`` should be an array of one of the following shapes:

            - :math:`I \times J_t`: Conditions on inside purchases by assuming :math:`w_{di0t} = 0`. Rows correspond to
              agents :math:`i \in I` in the same order as ``agent_data`` in :class:`Problem` or :class:`Simulation` and
              columns correspond to inside products :math:`j \in J_t` in the same order as ``product_data`` in
              :class:`Problem` or :class:`Simulation`.

            - :math:`I \times (1 + J_t)`: The first column indexes the outside option, which can have nonzero survey
              weights :math:`w_{di0t}`.

        If the micro dataset contains second choice data, ``weights`` can have a third axis corresponding to second
        choices :math:`k` in :math:`w_{dijkt}`:

            - :math:`I \times J_t \times J_t`: Conditions on inside purchases by assuming
              :math:`w_{di0kt} = w_{dij0t} = 0`.

            - :math:`I \times (1 + J_t) \times J_t`: The first column indexes the outside option, but the second
              choice is assumed to be an inside option, :math:`w_{dij0t} = 0`.

            - :math:`I \times J_t \times (1 + J_t)`: The first index in the third axis indexes the outside option, but
              the first choice is assumed to be an inside option, :math:`w_{di0k} = 0`.

            - :math:`I \times (1 + J_t) \times (1 + J_t)`: The first column and the first index in the third axis
              index the outside option as the first and second choice.

        .. warning::

            Second choice moments can use a lot of memory, especially when :math:`J_t` is large. If this becomes an
            issue, consider setting ``pyblp.options.micro_computation_chunks`` to a value higher than its default of
            ``1``, such as the highest :math:`J_t`. This will cut down on memory usage without much affecting speed.

    market_ids : `array-like, optional`
        Distinct market IDs with nonzero survey weights :math:`w_{dijt}`. For other markets, :math:`w_{dijt} = 0`, and
        ``compute_weights`` will not be called.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    name: str
    observations: int
    compute_weights: functools.partial
    market_ids: Optional[Set]

    def __init__(
            self, name: str, observations: int, compute_weights: Callable,
            market_ids: Optional[Union[Sequence, Array]] = None) -> None:
        """Validate information to the greatest extent possible without an economy or calling the function."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not isinstance(observations, int) or observations < 1:
            raise ValueError("observations must be a positive int.")
        if not callable(compute_weights):
            raise ValueError("compute_weights must be callable.")

        self.name = name
        self.observations = observations
        self.compute_weights = functools.partial(compute_weights)

        # validate market IDs, checking for duplicates
        if market_ids is None:
            self.market_ids = None
        else:
            if isinstance(market_ids, set):
                market_ids = list(market_ids)
            market_ids = np.asarray(market_ids, np.object_)
            unique, counts = np.unique(market_ids, return_counts=True)
            duplicates = unique[counts > 1]
            if duplicates.size > 0:
                raise ValueError(f"The following market IDs are duplicated in market_ids: {duplicates}.")
            self.market_ids = set(market_ids.flatten())

    def __str__(self) -> str:
        """Format information about the dataset as a string."""
        return f"{self.name}: {self.observations} Observations in {self._format_markets(text=True)}"

    def _validate(self, economy: 'Economy') -> None:
        """Check that all market IDs associated with this dataset are in the economy."""
        if self.market_ids is not None:
            extra_ids = self.market_ids - set(economy.unique_market_ids)
            if extra_ids:
                raise ValueError(f"market_ids contains the following extra IDs: {sorted(extra_ids)}.")

    def _format_markets(self, text: bool = False) -> str:
        """Format information about the markets associated with the dataset as a string."""
        if not text:
            return "All" if self.market_ids is None else str(len(self.market_ids))
        if self.market_ids is None:
            return "All Markets"
        if len(self.market_ids) == 1:
            return f"Market '{list(self.market_ids)[0]}'"
        return f"{len(self.market_ids)} Markets"


class MicroMoment(StringRepresentation):
    r"""Configuration for a micro moment :math:`m`.

    Each micro moment :math:`m` is defined by its dataset :math:`d_m` and micro values :math:`v_{mijt}`, which are used
    in :eq:`averaged_micro_moments`. For example, a micro moment :math:`m` with :math:`v_{mijt} = y_{it}x_{jt}` matches
    the mean of an interaction between some demographic :math:`y_{it}` and some product characteristic :math:`x_{jt}`.

    .. warning::

        Micro moments are under active development. Their API and functionality may change as development progresses.

    Parameters
    ----------
    name : str
        The unique name of the micro moment, which will be used for outputting information about micro moments.
    dataset : MicroDataset
        The :class:`MicroDataset` :math:`d_m` on which the observed ``value`` was computed.
    value : float
        The observed value :math:`\bar{v}_m` in :eq:`observed_micro_value`.
    compute_values : `callable`
        Function for computing micro values :math:`v_{mijt}` (or :math:`v_{mijkt}` if the dataset :math:`d_m` contains
        second choice data) in a market of the following form::

            compute_values(t, products, agents) --> values

        where ``t`` is the market in which to compute values, ``products`` is the market's :class:`Products` (with
        :math:`J_t` rows), and ``agents`` is the market's :class:`Agents` (with :math:`I_t` rows), unless
        ``pyblp.options.micro_computation_chunks`` is larger than its default of ``1``, in which case ``agents`` is a
        chunk of the market's :class:`Agents`. The returned ``values`` should be an array of the same shape as the
        ``weights`` returned by ``compute_weights`` of ``dataset``.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    name: str
    dataset: MicroDataset
    value: float
    compute_values: functools.partial

    def __init__(self, name: str, dataset: MicroDataset, value: float, compute_values: Callable) -> None:
        """Validate information to the greatest extent possible without calling the function."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not isinstance(dataset, MicroDataset):
            raise TypeError("dataset must be a MicroDataset instance.")
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a float.")
        if not callable(compute_values):
            raise ValueError("compute_values must be callable.")

        self.name = name
        self.dataset = dataset
        self.value = value
        self.compute_values = functools.partial(compute_values)

    def __str__(self) -> str:
        """Format information about the moment as a string."""
        return f"{self.name}: {format_number(self.value)} ({self.dataset})"


class Moments(object):
    """Information about a sequence of micro moments."""

    micro_moments: Sequence[MicroMoment]
    values: Array
    MM: int

    def __init__(self, economy: 'Economy', micro_moments: Sequence[MicroMoment]) -> None:
        """Validate and store information about a sequence of micro moment instances."""
        if not isinstance(micro_moments, collections.abc.Sequence):
            raise TypeError("micro_moments must be a sequence of micro moment instances.")
        for m, moment in enumerate(micro_moments):
            if not isinstance(moment, MicroMoment):
                raise TypeError("micro_moments must consist only of micro moment instances.")
            try:
                moment.dataset._validate(economy)
            except Exception as exception:
                message = f"The micro dataset '{moment.dataset}' is invalid because of the above exception."
                raise ValueError(message) from exception
            for moment2 in micro_moments[:m]:
                if moment == moment2:
                    raise ValueError(f"There is more than one of the micro moment '{moment}'.")
                if moment.name == moment2.name:
                    raise ValueError(f"Micro moment '{moment}' has the same name as '{moment2}'.")
                if moment.dataset != moment2.dataset and moment.dataset.name == moment2.dataset.name:
                    raise ValueError(
                        f"The dataset of '{moment}' is not the same instance as that of '{moment2}', but the two "
                        f"datasets have the same name."
                    )

        self.micro_moments = micro_moments
        self.values = np.c_[[m.value for m in micro_moments]]
        self.MM = len(micro_moments)

    def format(self, title: str, values: Optional[Array] = None) -> str:
        """Format micro moments and their associated datasets as a string."""
        header = ["Observed"]
        if values is not None:
            header.extend(["Estimated", "Difference"])
        header.extend(["Moment", "Dataset", "Observations", "Markets"])

        data: List[List[str]] = []
        for m, moment in enumerate(self.micro_moments):
            row = [format_number(moment.value)]
            if values is not None:
                row.extend([format_number(values[m]), format_number(moment.value - values[m])])
            row.extend([
                moment.name,
                moment.dataset.name,
                str(moment.dataset.observations),
                moment.dataset._format_markets(),
            ])
            data.append(row)

        return format_table(header, *data, title=title)
