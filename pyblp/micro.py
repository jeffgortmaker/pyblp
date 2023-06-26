"""Micro moments."""

import collections.abc
import functools
from typing import Any, Callable, List, Optional, Sequence, Set, TYPE_CHECKING, Union

import numpy as np

from .utilities.basics import Array, StringRepresentation, format_number, format_table


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .economies.economy import Economy  # noqa


class MicroDataset(StringRepresentation):
    r"""Configuration for a micro dataset :math:`d` on which micro moments are computed.

    A micro dataset :math:`d`, often a survey, is defined by survey weights :math:`w_{dijt}`, which are used in
    :eq:`micro_moment`. For example, :math:`w_{dijt} = 1\{j \neq 0, t \in T_d\}` defines a micro dataset that is a
    selected sample of inside purchasers in a few markets :math:`T_d \subset T`, giving each market an equal sampling
    weight. Different micro datasets are independent.

    See :ref:`references:Conlon and Gortmaker (2023)` for a more in-depth discussion of the standardized framework used
    by PyBLP for incorporating micro data into BLP-style estimation.

    Parameters
    ----------
    name : `str`
        The unique name of the dataset, which will be used for outputting information about micro moments.
    observations : `int`
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

        .. warning::

            If using different lambda functions to define different ``compute_weights`` functions in a loop, any
            variables that are changing within the loop should be passed as extra arguments to the function to preserve
            their scope. For example, ``lambda t, p, a: weights[t]`` where ``weights`` is some dictionary that is
            changing in the outer loop should instead be ``lambda t, p, a, weights=weights: weights[t]``; otherwise,
            the ``weights`` in the current loop's iteration will be lost.

        .. warning::

            If using product-specific demographics, ``agents.demographics`` will be a :math:`I_t \times D \times J_t`
            array, instead of a :math:`I_t \times D` array like usual. Non-product specific demographics will be
            repeated :math:`J_t` times.

        .. note::

            Particularly when using product-specific demographics or second choices, it may be convenient to use
            ``numpy.einsum``, which handles many multiplying multi-dimensional arrays with common dimensions in an
            elegant way.

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

    eliminated_product_ids_index : `int, optional`
        This option determines whether the dataset's second choices are after only the first choice product :math:`j` is
        eliminated from the choice set, in which case this should be ``None``, the default, or if a group of products
        including the first choice product is eliminated, in which case this should be a number between ``0`` and the
        number of columns in the ``product_ids`` field of ``product_data`` minus one, inclusive. The column of
        ``product_ids`` determines the groups.
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
    eliminated_product_ids_index: Optional[int]

    def __init__(
            self, name: str, observations: int, compute_weights: Callable,
            eliminated_product_ids_index: Optional[int] = None,
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

        # validate the product IDs index
        self.eliminated_product_ids_index = eliminated_product_ids_index
        if eliminated_product_ids_index is not None:
            if not isinstance(eliminated_product_ids_index, int) or eliminated_product_ids_index < 0:
                raise ValueError("eliminated_product_ids_index must be None or a non-negative int.")

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
        """Check that all market IDs associated with this dataset are in the economy and that any eliminated product
        IDs index is valid.
        """
        if self.eliminated_product_ids_index is not None:
            economy._validate_product_ids_index(self.eliminated_product_ids_index)
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


class MicroPart(StringRepresentation):
    r"""Configuration for a micro moment part :math:`p`.

    Each micro moment part :math:`p` is defined by its dataset :math:`d_p` and micro values :math:`v_{pijt}`, which are
    used in :eq:`observed_micro_part` and :eq:`simulated_micro_part`. For example, a micro moment part :math:`p` with
    :math:`v_{pijt} = y_{it} x_{jt}` yields the mean :math:`\bar{v}_p` or expectation :math:`v_p` of an interaction
    between some demographic :math:`y_{it}` and product characteristic :math:`x_{jt}`.

    See :ref:`references:Conlon and Gortmaker (2023)` for a more in-depth discussion of the standardized framework used
    by PyBLP for incorporating micro data into BLP-style estimation.

    Parameters
    ----------
    name : `str`
        The unique name of the micro moment part, which will be used for outputting information about micro moments.
    dataset : `MicroDataset`
        The :class:`MicroDataset` :math:`d_p` on which the micro part is computed.
    compute_values : `callable`
        Function for computing micro values :math:`v_{pijt}` (or :math:`v_{pijkt}` if the dataset :math:`d_p` contains
        second choice data) in a market of the following form::

            compute_values(t, products, agents) --> values

        where ``t`` is the market in which to compute values, ``products`` is the market's :class:`Products` (with
        :math:`J_t` rows), and ``agents`` is the market's :class:`Agents` (with :math:`I_t` rows), unless
        ``pyblp.options.micro_computation_chunks`` is larger than its default of ``1``, in which case ``agents`` is a
        chunk of the market's :class:`Agents`. The returned ``values`` should be an array of the same shape as the
        ``weights`` returned by ``compute_weights`` of ``dataset``.

        .. warning::

            If using different lambda functions to define different ``compute_values`` functions in a loop, any
            variables that are changing within the loop should be passed as extra arguments to the function to preserve
            their scope. For example, ``lambda t, p, a: np.outer(a.demographics[:, d], p.X2[:, c])`` where ``d`` and
            ``c`` are indices that are changing in the outer loop should instead be
            ``lambda t, p, a, d=d, c=c: np.outer(a.demographics[:, d], p.X2[:, c])``; otherwise, the values of ``d``
            and ``c`` in the current loop's iteration will be lost.

        .. warning::

            If using product-specific demographics, ``agents.demographics`` will be a :math:`I_t \times D \times J_t`
            array, instead of a :math:`I_t \times D` array like usual. Non-product specific demographics will be
            repeated :math:`J_t` times.

        .. note::

            Particularly when using product-specific demographics or second choices, it may be convenient to use
            ``numpy.einsum``, which handles many multiplying multi-dimensional arrays with common dimensions in an
            elegant way.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """
    name: str
    dataset: MicroDataset
    compute_values: functools.partial

    def __init__(self, name: str, dataset: MicroDataset, compute_values: Callable) -> None:
        """Validate information to the greatest extent possible without calling the function."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not isinstance(dataset, MicroDataset):
            raise TypeError("dataset must be a MicroDataset instance.")
        if not callable(compute_values):
            raise ValueError("compute_values must be callable.")

        self.name = name
        self.dataset = dataset
        self.compute_values = functools.partial(compute_values)

    def __str__(self) -> str:
        """Format information about the part as a string."""
        return f"{self.name} on {self.dataset}"


class MicroMoment(StringRepresentation):
    r"""Configuration for a micro moment :math:`m`.

    Each micro moment :math:`m` matches a function :math:`f_m(v)` of one or more micro moment parts :math:`v` in
    :eq:`micro_moment`. For example, :math:`f_m(v) = v_p` with :math:`v_{pijt} = y_{it} x_{jt}` matches the mean of
    an interaction between some demographic :math:`y_{it}` and some product characteristic :math:`x_{jt}`.

    Non-simple averages such as conditional means, covariances, correlations, or regression coefficients can be matched
    by choosing an appropriate function :math:`f_m`. For example, :math:`f_m(v) = v_1 / v_2` with
    :math:`v_{1ijt} = y_{it}x_{jt}1\{j \neq 0\}` and :math:`v_{2ijt} = 1\{j \neq 0\}` matches the conditional mean of an
    interaction between :math:`y_{it}` and :math:`x_{jt}` among those who do not choose the outside option
    :math:`j = 0`.

    See :ref:`references:Conlon and Gortmaker (2023)` for a more in-depth discussion of the standardized framework used
    by PyBLP for incorporating micro data into BLP-style estimation.

    Parameters
    ----------
    name : `str`
        The unique name of the micro moment, which will be used for outputting information about micro moments.
    value : `float`
        The observed value :math:`f_m(\bar{v})`.
    parts : `MicroPart or sequence of MicroPart`
        The :class:`MicroPart` configurations on which :math:`f_m(\cdot)` depends. If this is just a single part
        :math:`p` and not a sequence, it is assumed that :math:`f_m = v_p` so that the micro moment matches :math:`v_p`.
        If this is a sequence, both ``compute_value`` and ``compute_gradient`` need to be specified.
    compute_value : `callable, optional`
        Function for computing the simulated micro value :math:`f_m(v)` (only if ``parts`` is a sequence) of the
        following form::

            compute_value(part_values) --> value

        where ``part_values`` is the array :math:`v` with as many values as there are ``parts`` and the returned
        ``value`` is the scalar :math:`f_m(v)`.

    compute_gradient : `callable, optional`
        Function for computing the gradient of the simulated micro value with respect to its parts (only required if
        ``parts`` is a sequence) of the following form::

            compute_gradient(part_values) --> gradient

        where ``part_values`` is the array :math:`v` with as many value as there are ``parts`` and the returned
        ``gradient`` is :math:`\frac{\partial f_m(v)}{\partial v}`, an array of the same shape. This is used to compute
        both analytic gradients and moment covariances.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    name: str
    value: float
    parts: Sequence[MicroPart]
    compute_value: functools.partial
    compute_gradient: functools.partial

    def __init__(
            self, name: str, value: Any, parts: Union[MicroPart, Sequence[MicroPart]],
            compute_value: Optional[Callable] = None, compute_gradient: Optional[Callable] = None) -> None:
        """Validate information to the greatest extent possible without calling the functions."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        value = np.asarray(value).flatten()
        if value.size != 1:
            raise TypeError("value must be a float.")
        if not np.isfinite(value):
            raise ValueError("value must be a finite number.")

        if isinstance(parts, MicroPart):
            parts = [parts]
            if compute_value is None:
                compute_value = default_compute_value
            if compute_gradient is None:
                compute_gradient = default_compute_gradient
        else:
            if not isinstance(parts, collections.abc.Sequence) or len(parts) < 1:
                raise TypeError("parts must be a MicroPart instance or a sequence of instances.")
            if compute_value is None:
                raise TypeError("Since parts is a sequence of MicroPart instances, compute_value must be specified.")
            if compute_gradient is None:
                raise TypeError("Since parts is a sequence of MicroPart instances, compute_gradient must be specified.")
            for p, part in enumerate(parts):
                if not isinstance(part, MicroPart):
                    raise TypeError("parts must be a MicroPart instance of a sequence of instances.")
                for part2 in parts[:p]:
                    if part == part2:
                        raise ValueError(f"There is more than one of the micro parts '{part}'.")
                    if part.name == part2.name:
                        raise ValueError(f"Micro part '{part}' has the same name as '{part2}'.")
                    if part.dataset != part2.dataset and part.dataset.name == part2.dataset.name:
                        raise ValueError(
                            f"The dataset of '{part}' is not the same instance as that of '{part2}', but the two "
                            f"datasets have the same name."
                        )
        if not callable(compute_value):
            raise ValueError("When specified, compute_value must be callable.")
        if not callable(compute_gradient):
            raise ValueError("When specified, compute_gradient must be callable.")

        self.name = name
        self.value = float(value)
        self.parts = parts
        self.compute_value = functools.partial(compute_value)
        self.compute_gradient = functools.partial(compute_gradient)

    def __str__(self) -> str:
        """Format information about the moment as a string."""
        parts_string = str(self.parts) if isinstance(self.parts, MicroPart) else "; ".join(str(p) for p in self.parts)
        return f"{self.name}: {format_number(self.value)} ({parts_string})"


def default_compute_value(part_values: Array) -> float:
    """Define the default micro value computation function for a single micro part. This needs to be at the module
    level to allow for standard multiprocessing to work.
    """
    return float(part_values)


def default_compute_gradient(part_values: Array) -> Array:
    """Define the same but for the gradient."""
    return np.ones_like(part_values)


class Moments(object):
    """Information about a sequence of micro moments."""

    micro_moments: Sequence[MicroMoment]
    micro_parts: Sequence[MicroPart]
    values: Array
    MM: int
    PM: int

    def __init__(self, micro_moments: Sequence[MicroMoment], economy: 'Economy') -> None:
        """Validate and store information about a sequence of micro moment instances."""
        if not isinstance(micro_moments, collections.abc.Sequence):
            raise TypeError("micro_moments must be a sequence of micro moment instances.")
        for m, moment in enumerate(micro_moments):
            if not isinstance(moment, MicroMoment):
                raise TypeError("micro_moments must consist only of micro moment instances.")
            for moment2 in micro_moments[:m]:
                if moment == moment2:
                    raise ValueError(f"There is more than one of the micro moment '{moment}'.")
                if moment.name == moment2.name:
                    raise ValueError(f"Micro moment '{moment}' has the same name as '{moment2}'.")

        micro_parts = []
        for moment in micro_moments:
            for part in moment.parts:
                if part not in micro_parts:
                    micro_parts.append(part)

        for p, part in enumerate(micro_parts):
            try:
                part.dataset._validate(economy)
            except Exception as exception:
                message = f"The micro dataset '{part.dataset}' is invalid because of the above exception."
                raise ValueError(message) from exception
            for part2 in micro_parts[:p]:
                if part.name == part2.name:
                    raise ValueError(f"Micro part '{part}' has the same name as '{part2}'.")
                if part.dataset != part2.dataset and part.dataset.name == part2.dataset.name:
                    raise ValueError(
                        f"The dataset of '{part}' is not the same instance as that of '{part2}', but the two "
                        f"datasets have the same name."
                    )

        self.micro_moments = micro_moments
        self.micro_parts = micro_parts
        self.values = np.c_[[m.value for m in micro_moments]]
        self.MM = len(micro_moments)
        self.PM = len(micro_parts)

    def format(self, title: str, values: Optional[Array] = None) -> str:
        """Format micro moments and their associated datasets as a string."""
        header = ["Observed"]
        if values is not None:
            header.extend(["Estimated", "Difference"])
        header.extend(["Moment", "Part", "Dataset", "Observations", "Markets"])

        data: List[List[str]] = []
        for m, moment in enumerate(self.micro_moments):
            row = [format_number(moment.value)]
            if values is not None:
                row.extend([format_number(values[m]), format_number(moment.value - values[m])])
            row.extend([
                moment.name,
                moment.parts[0].name,
                moment.parts[0].dataset.name,
                str(moment.parts[0].dataset.observations),
                moment.parts[0].dataset._format_markets(),
            ])
            data.append(row)

            for part in moment.parts[1:]:
                row = [""]
                if values is not None:
                    row.extend(["", ""])
                row.extend([
                    "",
                    part.name,
                    part.dataset.name,
                    str(part.dataset.observations),
                    part.dataset._format_markets(),
                ])
                data.append(row)

        return format_table(header, *data, title=title)
